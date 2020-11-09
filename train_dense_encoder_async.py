#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
 Pipeline to train DPR Biencoder
"""
import argparse
import csv
import glob
import logging
import math
import numpy as np
import os
import random
import sys
import time
from typing import Tuple, List, Iterable, Dict

import hydra
import torch
from omegaconf import DictConfig
from torch import Tensor as T
from torch import nn

from distributed_faiss.client import IndexClient
from distributed_faiss.index_cfg import IndexCfg
from distributed_faiss.index_state import IndexState

from dpr.data.biencoder_data import BiEncoderSample, BiEncoderPassage
from dpr.data.qa_validation import has_answer
from dense_retriever import iterate_encoded_files, generate_question_vectors
from dpr.models import init_biencoder_components
from dpr.models.biencoder import BiEncoder, BiEncoderNllLoss, BiEncoderBatch
from dpr.options import (
    add_encoder_params,
    add_training_params,
    setup_args_gpu,
    set_seed,
    add_tokenizer_params,
    set_cfg_params_from_state,
)
from dpr.utils.conf_utils import BiencoderDatasetsCfg
from dpr.utils.data_utils import ShardedDataIterator, Tensorizer, MultiSetDataIterator
from dpr.utils.dist_utils import all_gather_list
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    move_to_device,
    get_schedule_linear,
    CheckpointState,
    get_model_file,
    get_model_obj,
    load_states_from_checkpoint,
)
from dpr.utils.tokenizers import SimpleTokenizer

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


# TODO: unify with regular training pipeline
class BiEncoderTrainer(object):
    """
    BiEncoder training pipeline component. Can be used to initiate or resume training and validate the trained model
    using either binary classification's NLL loss or average rank of the question's gold passages across dataset
    provided pools of negative passages. For full IR accuracy evaluation, please see generate_dense_embeddings.py
    and dense_retriever.py CLI tools.
    """

    # TODO: switch to cfg
    def __init__(self, cfg: DictConfig):

        logger.info("Connecting to index server ...")
        self.index_client = IndexClient(cfg.index_cfg_path)
        self.index_id = "dpr"
        logger.info("Connected")

        self.shard_id = cfg.local_rank if cfg.local_rank != -1 else 0
        self.distributed_factor = cfg.distributed_world_size or 1

        logger.info("***** Initializing components for training *****")

        # if model file is specified, encoder parameters from saved state should be used for initialization
        model_file = get_model_file(cfg, cfg.checkpoint_file_name)
        saved_state = None
        if model_file:
            saved_state = load_states_from_checkpoint(model_file)
            set_cfg_params_from_state(saved_state.encoder_params, cfg)

        tensorizer, model, optimizer = init_biencoder_components(
            cfg.encoder.encoder_model_type, cfg
        )

        model, optimizer = setup_for_distributed_mode(
            model,
            optimizer,
            cfg.device,
            cfg.n_gpu,
            cfg.local_rank,
            cfg.fp16,
            cfg.fp16_opt_level,
        )
        self.biencoder = model
        self.optimizer = optimizer
        self.tensorizer = tensorizer
        self.start_epoch = 0
        self.start_batch = 0
        self.scheduler_state = None
        self.best_validation_result = None
        self.best_cp_name = None
        self.cfg = cfg
        self.ds_cfg = BiencoderDatasetsCfg(cfg)

        if saved_state:
            self._load_saved_state(saved_state)

        self.dev_iterator = None
        self.train_iterator = None

        psg_to_idx = self._get_passages_to_index()
        self.all_passages = psg_to_idx[0]
        self.passages_start_idx = psg_to_idx[1]
        self.passages_end_idx = psg_to_idx[2]

        # query/question -> new hard negatives list dictionary to use instead of those from dataset
        self.new_hn_dict = {}

    def run_train(self):
        cfg = self.cfg

        train_iterator = self.get_data_iterator(
            cfg.train.batch_size,
            True,
            shuffle=True,
            shuffle_seed=cfg.seed,
            offset=self.start_batch,
            rank=cfg.local_rank,
        )
        self.train_iterator = train_iterator

        logger.info("  Total iterations per epoch=%d", train_iterator.max_iterations)
        updates_per_epoch = (
            train_iterator.max_iterations // cfg.train.gradient_accumulation_steps
        )
        total_updates = (
            max(
                updates_per_epoch * (cfg.train.num_train_epochs - self.start_epoch - 1),
                0,
            )
            + (train_iterator.max_iterations - self.start_batch)
            // cfg.train.gradient_accumulation_steps
        )
        logger.info(" Total updates=%d", total_updates)
        warmup_steps = cfg.train.warmup_steps
        scheduler = get_schedule_linear(self.optimizer, warmup_steps, total_updates)

        if self.scheduler_state:
            logger.info("Loading scheduler state %s", self.scheduler_state)
            scheduler.load_state_dict(self.scheduler_state)

        eval_step = math.ceil(updates_per_epoch / cfg.train.eval_per_epoch)
        logger.info("  Eval step = %d", eval_step)
        logger.info("***** Training *****")

        for epoch in range(self.start_epoch, int(cfg.train.num_train_epochs)):
            logger.info("***** Epoch %d *****", epoch)
            self._train_epoch(scheduler, epoch, eval_step)

        if cfg.local_rank in [-1, 0]:
            logger.info(
                "Training finished. Best validation checkpoint %s", self.best_cp_name
            )

    def validate_and_save(self, epoch: int, iteration: int, scheduler):
        cfg = self.cfg
        # for distributed mode, save checkpoint for only one process
        save_cp = cfg.local_rank in [-1, 0]

        if epoch == cfg.val_av_rank_start_epoch:
            self.best_validation_result = None

        if epoch >= cfg.val_av_rank_start_epoch:
            validation_loss = self.validate_average_rank()
        else:
            validation_loss = self.validate_nll()

        if save_cp:
            cp_name = self._save_checkpoint(scheduler, epoch, iteration)
            logger.info("Saved checkpoint to %s", cp_name)

            if validation_loss < (self.best_validation_result or validation_loss + 1):
                self.best_validation_result = validation_loss
                self.best_cp_name = cp_name
                logger.info("New Best validation checkpoint %s", cp_name)

    def validate_nll(self) -> float:
        logger.info("NLL validation ...")
        cfg = self.cfg
        self.biencoder.eval()

        if not self.dev_iterator:
            self.dev_iterator = self.get_data_iterator(
                cfg.train.dev_batch_size, False, shuffle=False, rank=cfg.local_rank
            )
        data_iterator = self.dev_iterator

        total_loss = 0.0
        start_time = time.time()
        total_correct_predictions = 0
        num_hard_negatives = cfg.train.hard_negatives
        num_other_negatives = cfg.train.other_negatives
        log_result_step = cfg.train.log_batch_step
        batches = 0

        dataset = 0
        for i, samples_batch in enumerate(data_iterator.iterate_ds_data()):
            if isinstance(samples_batch, Tuple):
                samples_batch, dataset = samples_batch
            logger.info("Eval step: %d ,rnk=%s", i, cfg.local_rank)
            biencoder_input = BiEncoder.create_biencoder_input2(
                samples_batch,
                self.tensorizer,
                True,
                num_hard_negatives,
                num_other_negatives,
                shuffle=False,
            )
            # get the token to be used for representation selection
            ds_cfg = self.ds_cfg.train_datasets[dataset]
            rep_positions = ds_cfg.selector.get_positions(
                biencoder_input.question_ids, self.tensorizer
            )

            loss, correct_cnt = _do_biencoder_fwd_pass(
                self.biencoder,
                biencoder_input,
                self.tensorizer,
                cfg,
                encoder_type="mixed",
                rep_positions=rep_positions,
            )
            total_loss += loss.item()
            total_correct_predictions += correct_cnt
            batches += 1
            if (i + 1) % log_result_step == 0:
                logger.info(
                    "Eval step: %d , used_time=%f sec., loss=%f ",
                    i,
                    time.time() - start_time,
                    loss.item(),
                )

        total_loss = total_loss / batches
        total_samples = batches * cfg.train.dev_batch_size * self.distributed_factor
        correct_ratio = float(total_correct_predictions / total_samples)
        logger.info(
            "NLL Validation: loss = %f. correct prediction ratio  %d/%d ~  %f",
            total_loss,
            total_correct_predictions,
            total_samples,
            correct_ratio,
        )
        return total_loss

    def validate_average_rank(self) -> float:
        """
        Validates biencoder model using each question's gold passage's rank across the set of passages from the dataset.
        It generates vectors for specified amount of negative passages from each question (see --val_av_rank_xxx params)
        and stores them in RAM as well as question vectors.
        Then the similarity scores are calculted for the entire
        num_questions x (num_questions x num_passages_per_question) matrix and sorted per quesrtion.
        Each question's gold passage rank in that  sorted list of scores is averaged across all the questions.
        :return: averaged rank number
        """
        logger.info("Average rank validation ...")

        cfg = self.cfg
        self.biencoder.eval()
        distributed_factor = self.distributed_factor

        if not self.dev_iterator:
            self.dev_iterator = self.get_data_iterator(
                cfg.train.dev_batch_size, False, shuffle=False, rank=cfg.local_rank
            )
        data_iterator = self.dev_iterator

        sub_batch_size = cfg.train.val_av_rank_bsz
        sim_score_f = BiEncoderNllLoss.get_similarity_function()
        q_represenations = []
        ctx_represenations = []
        positive_idx_per_question = []

        num_hard_negatives = cfg.train.val_av_rank_hard_neg
        num_other_negatives = cfg.train.val_av_rank_other_neg

        log_result_step = cfg.train.log_batch_step
        dataset = 0
        for i, samples_batch in enumerate(data_iterator.iterate_ds_data()):
            # samples += 1
            if (
                len(q_represenations)
                > cfg.train.val_av_rank_max_qs / distributed_factor
            ):
                break

            if isinstance(samples_batch, Tuple):
                samples_batch, dataset = samples_batch

            biencoder_input = BiEncoder.create_biencoder_input2(
                samples_batch,
                self.tensorizer,
                True,
                num_hard_negatives,
                num_other_negatives,
                shuffle=False,
            )
            total_ctxs = len(ctx_represenations)
            ctxs_ids = biencoder_input.context_ids
            ctxs_segments = biencoder_input.ctx_segments
            bsz = ctxs_ids.size(0)

            # get the token to be used for representation selection
            ds_cfg = self.ds_cfg.train_datasets[dataset]
            rep_positions = ds_cfg.selector.get_positions(
                biencoder_input.question_ids, self.tensorizer
            )

            # split contexts batch into sub batches since it is supposed to be too large to be processed in one batch
            for j, batch_start in enumerate(range(0, bsz, sub_batch_size)):

                q_ids, q_segments = (
                    (biencoder_input.question_ids, biencoder_input.question_segments)
                    if j == 0
                    else (None, None)
                )

                if j == 0 and cfg.n_gpu > 1 and q_ids.size(0) == 1:
                    # if we are in DP (but not in DDP) mode, all model input tensors should have batch size >1 or 0,
                    # otherwise the other input tensors will be split but only the first split will be called
                    continue

                ctx_ids_batch = ctxs_ids[batch_start : batch_start + sub_batch_size]
                ctx_seg_batch = ctxs_segments[
                    batch_start : batch_start + sub_batch_size
                ]

                q_attn_mask = self.tensorizer.get_attn_mask(q_ids)
                ctx_attn_mask = self.tensorizer.get_attn_mask(ctx_ids_batch)
                with torch.no_grad():
                    q_dense, ctx_dense = self.biencoder(
                        q_ids,
                        q_segments,
                        q_attn_mask,
                        ctx_ids_batch,
                        ctx_seg_batch,
                        ctx_attn_mask,
                        representation_token_pos=rep_positions,
                    )

                if q_dense is not None:
                    q_represenations.extend(q_dense.cpu().split(1, dim=0))

                ctx_represenations.extend(ctx_dense.cpu().split(1, dim=0))

            batch_positive_idxs = biencoder_input.is_positive
            positive_idx_per_question.extend(
                [total_ctxs + v for v in batch_positive_idxs]
            )

            if (i + 1) % log_result_step == 0:
                logger.info(
                    "Av.rank validation: step %d, computed ctx_vectors %d, q_vectors %d",
                    i,
                    len(ctx_represenations),
                    len(q_represenations),
                )

        ctx_represenations = torch.cat(ctx_represenations, dim=0)
        q_represenations = torch.cat(q_represenations, dim=0)

        logger.info(
            "Av.rank validation: total q_vectors size=%s", q_represenations.size()
        )
        logger.info(
            "Av.rank validation: total ctx_vectors size=%s", ctx_represenations.size()
        )

        q_num = q_represenations.size(0)
        assert q_num == len(positive_idx_per_question)

        scores = sim_score_f(q_represenations, ctx_represenations)
        values, indices = torch.sort(scores, dim=1, descending=True)

        rank = 0
        for i, idx in enumerate(positive_idx_per_question):
            # aggregate the rank of the known gold passage in the sorted results for each question
            gold_idx = (indices[i] == idx).nonzero()
            rank += gold_idx.item()

        if distributed_factor > 1:
            # each node calcuated its own rank, exchange the information between node and calculate the "global" average rank
            # NOTE: the set of passages is still unique for every node
            eval_stats = all_gather_list([rank, q_num], max_size=100)
            for i, item in enumerate(eval_stats):
                remote_rank, remote_q_num = item
                if i != cfg.local_rank:
                    rank += remote_rank
                    q_num += remote_q_num

        av_rank = float(rank / q_num)
        logger.info(
            "Av.rank validation: average rank %s, total questions=%d", av_rank, q_num
        )
        return av_rank

    def get_data_iterator(
        self,
        batch_size: int,
        is_train_set: bool,
        shuffle=True,
        shuffle_seed: int = 0,
        offset: int = 0,
        rank: int = 0,
    ):

        hydra_datasets = (
            self.ds_cfg.train_datasets if is_train_set else self.ds_cfg.dev_datasets
        )
        sampling_rates = self.ds_cfg.sampling_rates

        logger.info(
            "Initializing multi task/set data %s",
            self.ds_cfg.train_datasets_names
            if is_train_set
            else self.ds_cfg.dev_datasets_names,
        )

        [ds.load_data() for ds in hydra_datasets]

        sharded_iterators = [
            ShardedDataIterator(
                ds,
                shard_id=self.shard_id,
                num_shards=self.distributed_factor,
                batch_size=batch_size,
                shuffle=shuffle,
                shuffle_seed=shuffle_seed,
                offset=offset,
            )
            for ds in hydra_datasets
        ]
        return MultiSetDataIterator(
            sharded_iterators,
            shuffle_seed,
            shuffle,
            sampling_rates=sampling_rates if is_train_set else [1],
            rank=rank,
        )

    def _get_passages_to_index(self) -> Tuple[List, int, int]:
        all_passages = []
        logger.info("Reading passages database fromm %s", self.cfg.ctx_file)
        with open(self.cfg.ctx_file) as tsvfile:
            reader = csv.reader(tsvfile, delimiter="\t")
            # file format: doc_id, doc_text, title
            all_passages.extend(
                [(row[0], row[1], row[2]) for row in reader if row[0] != "id"]
            )

        logger.info("All passages size %d", len(all_passages))

        # each node is to process its own shard of passages
        total_size = len(all_passages)
        samples_per_shard = math.ceil(total_size / self.cfg.distributed_world_size)
        shard_start_idx = self.shard_id * samples_per_shard
        shard_end_idx = min(shard_start_idx + samples_per_shard, total_size)
        return all_passages, shard_start_idx, shard_end_idx

    def _generate_embs_from_passages_it(
        self, passage_start_idx: int, passages: List
    ) -> Iterable[Tuple[np.array, list]]:
        self.biencoder.eval()
        logger.info("!!! _generate_embs_from_passages_it passages %d", len(passages))
        cfg = self.cfg
        model = get_model_obj(self.biencoder)
        encoder = model.ctx_model
        index_buffer_embs = []
        index_bufffer_size = 0
        index_buffer_meta = []

        index_bsz = cfg.index_batch_size
        dev_bsz = cfg.train.dev_batch_size
        insert_title = True

        def get_batch(buffer_embs, buffer_meta):
            # send the buffer to index
            embeddings = torch.cat(buffer_embs, dim=0).numpy()
            meta = [item for sublist in buffer_meta for item in sublist]
            return embeddings, meta

        for j, batch_start in enumerate(range(0, len(passages), dev_bsz)):
            batch_token_tensors = [
                self.tensorizer.text_to_tensor(
                    ctx[1], title=ctx[2] if insert_title else None
                )
                for ctx in passages[batch_start : batch_start + dev_bsz]
            ]
            actual_bsz = len(passages[batch_start : batch_start + dev_bsz])
            passage_start_ids = [
                passage_start_idx + batch_start + i for i in range(actual_bsz)
            ]

            ctx_ids_batch = move_to_device(
                torch.stack(batch_token_tensors, dim=0), cfg.device
            )
            ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch), cfg.device)
            ctx_attn_mask = move_to_device(
                self.tensorizer.get_attn_mask(ctx_ids_batch), cfg.device
            )
            with torch.no_grad():
                _, out, _ = encoder(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask)
            out = out.cpu()
            index_buffer_embs.append(out)
            index_buffer_meta.append(passage_start_ids)
            index_bufffer_size += out.size(0)

            if index_bufffer_size >= index_bsz:
                # send the buffer to index
                embeddings, meta = get_batch(index_buffer_embs, index_buffer_meta)
                logger.info(
                    "sending data for indexing. At batch %d. Index batch size %s",
                    j,
                    embeddings.shape,
                )
                index_buffer_embs = []
                index_bufffer_size = 0
                index_buffer_meta = []
                yield embeddings, meta

        if index_buffer_embs:
            embeddings, meta = get_batch(index_buffer_embs, index_buffer_meta)
            yield embeddings, meta

    def _build_index(
        self, shuffle_seed, emb_files: List[str] = [], load_index: bool = False
    ):
        self.biencoder.eval()

        cfg = self.cfg
        model = get_model_obj(self.biencoder)
        encoder = model.ctx_model

        index_ratio = cfg.index_ratio
        rnd = random.Random(shuffle_seed)
        torch.distributed.barrier()

        passages = self.all_passages[self.passages_start_idx : self.passages_end_idx]
        rnd.shuffle(passages)
        passages = passages[0 : int(len(passages) * index_ratio)]
        passage_start_idx = self.passages_start_idx
        logger.info("Indexing passages. passage_start_idx=%d", passage_start_idx)

        # warning: doesn't work in DP mode, supposed to be ued in DDP mode only
        index = self.index_client
        idx_cfg = IndexCfg()
        idx_cfg.dim = encoder.get_out_size()
        # tmp
        idx_cfg.train_num = max(int(len(passages) * cfg.index_train_ratio), 5000)
        logger.info("Index train num=%d", idx_cfg.train_num)
        idx_cfg.nprobe = cfg.index_nprobe
        idx_cfg.faiss_factory = cfg.index_faiss_factory
        idx_cfg.centroids = cfg.index_centroids
        index_id = self.index_id
        index.create_index(index_id, idx_cfg)

        if load_index:
            self.index_client.load_index(index_id, idx_cfg)
            total_index_data = self.index_client.get_ntotal(index_id)
            logger.info("Loaded index data %d", total_index_data)
            return

        # tmp
        log_meta = True
        if emb_files and cfg.local_rank == 0:
            logger.info("Reading embeddings from files")
            buffer = []
            index_bsz = cfg.index_batch_size

            def send_buf_data(buffer, index_client, log_meta):
                buffer_vectors = [
                    np.reshape(encoded_item[1], (1, -1)) for encoded_item in buffer
                ]
                buffer_vectors = np.concatenate(buffer_vectors, axis=0)
                meta = [int(encoded_item[0]) for encoded_item in buffer]
                # tmp:
                if log_meta:
                    logger.info("!!! meta %s", meta)
                logger.info("Sending vectors to index %s", buffer_vectors.shape)
                index_client.add_index_data(index_id, buffer_vectors, meta)

            for i, item in enumerate(iterate_encoded_files(emb_files)):
                buffer.append(item)
                if 0 < index_bsz == len(buffer):
                    send_buf_data(buffer, index, log_meta)
                    # tmp
                    log_meta = False
                    buffer = []
            if buffer:
                send_buf_data(buffer, index, log_meta)
            logger.info("Embeddings sent.")
        elif not emb_files:
            it = self._generate_embs_from_passages_it(shuffle_seed, passages)
            for i, index_batch in enumerate(it):
                embeddings, meta = index_batch
                index.add_index_data(index_id, embeddings, meta)

        logger.info(
            "Sent all data for indexing. passage_start_idx=%d. Building index ... ",
            passage_start_idx,
        )

        torch.distributed.barrier()
        if cfg.local_rank == 0:
            index.async_train(index_id)
            logger.info(
                "Sent all data for indexing. passage_start_idx=%d. Index data size %d",
                passage_start_idx,
                index.get_ntotal(index_id),
            )

    def _wait_index_ready(self):
        while self.index_client.get_state(self.index_id) != IndexState.TRAINED:
            time.sleep(10)
        logger.info(
            "Index is ready. Index data size %d",
            self.index_client.get_ntotal(self.index_id),
        )

    def _search_new_hard_negs(self) -> Dict[str, List[BiEncoderPassage]]:
        cfg = self.cfg
        self.biencoder.eval()

        datasets = self.train_iterator.get_datasets()
        query_to_hn_dict = {}
        model = get_model_obj(self.biencoder)
        q_encoder = model.question_model
        search_topk_k = cfg.index_search_topk_k
        async_max_hn = cfg.async_max_hn

        # to check the answer presence only
        tokenizer = SimpleTokenizer(**{})

        for ds in datasets:
            # requesting ALL data queries as of now
            queries, answers = ds.get_qas()
            index_bsz = 512  # cfg.index_batch_size
            n = len(queries)

            for i in range(0, n, index_bsz):
                query_batch = queries[i : i + index_bsz]
                query_batch_embs = generate_question_vectors(
                    q_encoder, self.tensorizer, query_batch, cfg.train.dev_batch_size
                )
                time0 = time.time()
                results = self.index_client.search(
                    query_batch_embs.numpy(), search_topk_k, self.index_id
                )
                logger.info("index search time: %f sec.", time.time() - time0)
                _scores, psg_ids = results

                for q_id, q in enumerate(query_batch):
                    query_result_ids = psg_ids[q_id]
                    new_hard_neg_passages = []
                    q_answers = answers[q_id]
                    for psg_id in query_result_ids:
                        # passage format: doc_id, doc_text, title
                        passage = self.all_passages[psg_id]
                        # check if passage has answer or not
                        if not has_answer(q_answers, passage[1], tokenizer, "string"):
                            new_hard_neg_passages.append(
                                BiEncoderPassage(passage[1], passage[2])
                            )

                        if len(new_hard_neg_passages) > async_max_hn:
                            break
                    query_to_hn_dict[q] = new_hard_neg_passages
        return query_to_hn_dict

    def _train_epoch(self, scheduler, epoch: int, eval_step: int):

        cfg = self.cfg
        rolling_train_loss = 0.0
        epoch_loss = 0
        epoch_correct_predictions = 0
        # index_rebuild_step = cfg.index_rebuild_updates
        # TODO: introduce global update variable and use for index step calc
        index_rebuild_epoch = cfg.index_rebuild_epochs

        log_result_step = cfg.train.log_batch_step
        rolling_loss_step = cfg.train.train_rolling_loss_step
        num_hard_negatives = cfg.train.hard_negatives
        num_other_negatives = cfg.train.other_negatives

        epoch_batches = self.train_iterator.max_iterations
        data_iteration = 0

        # tmp:
        logger.info("!!! new_hn_dict size %d", len(self.new_hn_dict))
        seed = cfg.seed
        dataset = 0

        if epoch == 0 or (
            (epoch + 1) % index_rebuild_epoch == 0 and epoch < cfg.index_max_epoch
        ):
            logger.info(
                "rank=%d, Index re/build: Epoch: %d Step: %d/%d",
                cfg.local_rank,
                epoch,
                data_iteration,
                epoch_batches,
            )
            self.index_client.drop_index(self.index_id)
            emb_files = cfg.emb_files
            if emb_files:
                emb_files = glob.glob(emb_files)
            self._build_index(
                cfg.seed,
                emb_files=(emb_files if epoch == 0 else None),
                load_index=False,
            )
            self._wait_index_ready()
            # self.index_client.save_index(self.index_id)
            self.new_hn_dict = self._search_new_hard_negs()
            logger.info("!!! new_hn_dict size %d", len(self.new_hn_dict))

        self.biencoder.train()
        for i, samples_batch in enumerate(
            self.train_iterator.iterate_ds_data(epoch=epoch)
        ):
            if isinstance(samples_batch, Tuple):
                samples_batch, dataset = samples_batch

            ds_cfg = self.ds_cfg.train_datasets[dataset]
            special_token = ds_cfg.special_token
            shuffle_positives = ds_cfg.shuffle_positives
            logger.info(" shuffle_positives %s", shuffle_positives)

            if self.new_hn_dict:
                # replace hard negatives with the new ones
                self._replace_hns(samples_batch)

            # to be able to resume shuffled ctx- pools
            data_iteration = self.train_iterator.get_iteration()
            # random.seed(seed + epoch + data_iteration)

            biencoder_batch = BiEncoder.create_biencoder_input2(
                samples_batch,
                self.tensorizer,
                True,
                num_hard_negatives,
                num_other_negatives,
                shuffle=True,
                shuffle_positives=shuffle_positives,
                query_token=special_token,
            )

            # get the token to be used for representation selection
            rep_positions = ds_cfg.selector.get_positions(
                biencoder_batch.question_ids, self.tensorizer
            )

            loss, correct_cnt = _do_biencoder_fwd_pass(
                self.biencoder,
                biencoder_batch,
                self.tensorizer,
                cfg,
                encoder_type="mixed",
                rep_positions=rep_positions,
            )

            epoch_correct_predictions += correct_cnt
            epoch_loss += loss.item()
            rolling_train_loss += loss.item()

            if cfg.fp16:
                from apex import amp

                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                if cfg.train.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(self.optimizer), cfg.train.max_grad_norm
                    )
            else:
                loss.backward()
                if cfg.train.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.biencoder.parameters(), cfg.train.max_grad_norm
                    )

            if (i + 1) % cfg.train.gradient_accumulation_steps == 0:
                self.optimizer.step()
                scheduler.step()
                self.biencoder.zero_grad()

            if i % log_result_step == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch: %d: Step: %d/%d, loss=%f, lr=%f",
                    epoch,
                    data_iteration,
                    epoch_batches,
                    loss.item(),
                    lr,
                )

            if (i + 1) % rolling_loss_step == 0:
                logger.info("Train batch %d", data_iteration)
                latest_rolling_train_av_loss = rolling_train_loss / rolling_loss_step
                logger.info(
                    "Avg. loss per last %d batches: %f",
                    rolling_loss_step,
                    latest_rolling_train_av_loss,
                )
                rolling_train_loss = 0.0

            if data_iteration % eval_step == 0:
                logger.info(
                    "rank=%d, Validation: Epoch: %d Step: %d/%d",
                    cfg.local_rank,
                    epoch,
                    data_iteration,
                    epoch_batches,
                )
                self.validate_and_save(
                    epoch, self.train_iterator.get_iteration(), scheduler
                )
                self.biencoder.train()

        logger.info("Epoch finished on %d", cfg.local_rank)
        epoch_loss = (epoch_loss / epoch_batches) if epoch_batches > 0 else 0
        logger.info("Av Loss per epoch=%f", epoch_loss)
        logger.info("epoch total correct predictions=%d", epoch_correct_predictions)

        self.validate_and_save(epoch, data_iteration, scheduler)

    def _replace_hns(self, samples: List[BiEncoderSample]):
        for s in samples:
            q = s.query
            if q not in self.new_hn_dict:
                continue
            q_new_hns = self.new_hn_dict[q]
            same_hn = 0
            for passage in q_new_hns:
                if any(passage.text == hn.text for hn in s.hard_negative_passages):
                    same_hn += 1

            s.hard_negative_passages = q_new_hns
            """
            logger.info(
                "!!! s.hard_negative_passages new %d. Overlap with prev %d",
                len(s.hard_negative_passages),
                same_hn,
            )
            """

    def _save_checkpoint(self, scheduler, epoch: int, offset: int) -> str:
        cfg = self.cfg
        model_to_save = get_model_obj(self.biencoder)
        cp = os.path.join(
            cfg.output_dir, cfg.checkpoint_file_name + "." + str(epoch)
        )  # + ('.' + str(offset) if offset > 0 else ''))
        # TODO
        # tmp
        meta_params = {}  # get_encoder_params_state(args)

        state = CheckpointState(
            model_to_save.state_dict(),
            self.optimizer.state_dict(),
            scheduler.state_dict(),
            offset,
            epoch,
            meta_params,
        )
        torch.save(state._asdict(), cp)
        logger.info("Saved checkpoint at %s", cp)
        return cp

    def _load_saved_state(self, saved_state: CheckpointState):
        epoch = saved_state.epoch
        offset = saved_state.offset
        if offset == 0:  # epoch has been completed
            epoch += 1
        logger.info("Loading checkpoint @ batch=%s and epoch=%s", offset, epoch)
        # tmp
        # self.start_epoch = epoch
        # self.start_batch = offset

        model_to_load = get_model_obj(self.biencoder)
        logger.info("Loading saved model state ...")
        model_to_load.load_state_dict(
            saved_state.model_dict
        )  # set strict=False if you use extra projection

        # tmp
        # if saved_state.optimizer_dict:
        #    logger.info("Loading saved optimizer state ...")
        #    self.optimizer.load_state_dict(saved_state.optimizer_dict)

        # if saved_state.scheduler_dict:
        #    self.scheduler_state = saved_state.scheduler_dict


def _calc_loss(
    args,
    loss_function,
    local_q_vector,
    local_ctx_vectors,
    local_positive_idxs,
    local_hard_negatives_idxs: list = None,
) -> Tuple[T, bool]:
    """
    Calculates In-batch negatives schema loss and supports to run it in DDP mode by exchanging the representations
    across all the nodes.
    """
    distributed_world_size = args.distributed_world_size or 1
    if distributed_world_size > 1:
        q_vector_to_send = (
            torch.empty_like(local_q_vector).cpu().copy_(local_q_vector).detach_()
        )
        ctx_vector_to_send = (
            torch.empty_like(local_ctx_vectors).cpu().copy_(local_ctx_vectors).detach_()
        )

        global_question_ctx_vectors = all_gather_list(
            [
                q_vector_to_send,
                ctx_vector_to_send,
                local_positive_idxs,
                local_hard_negatives_idxs,
            ],
            max_size=args.global_loss_buf_sz,
        )

        global_q_vector = []
        global_ctxs_vector = []

        # ctxs_per_question = local_ctx_vectors.size(0)
        positive_idx_per_question = []
        hard_negatives_per_question = []

        total_ctxs = 0

        for i, item in enumerate(global_question_ctx_vectors):
            q_vector, ctx_vectors, positive_idx, hard_negatives_idxs = item

            if i != args.local_rank:
                global_q_vector.append(q_vector.to(local_q_vector.device))
                global_ctxs_vector.append(ctx_vectors.to(local_q_vector.device))
                positive_idx_per_question.extend([v + total_ctxs for v in positive_idx])
                hard_negatives_per_question.extend(
                    [[v + total_ctxs for v in l] for l in hard_negatives_idxs]
                )
            else:
                global_q_vector.append(local_q_vector)
                global_ctxs_vector.append(local_ctx_vectors)
                positive_idx_per_question.extend(
                    [v + total_ctxs for v in local_positive_idxs]
                )
                hard_negatives_per_question.extend(
                    [[v + total_ctxs for v in l] for l in local_hard_negatives_idxs]
                )
            total_ctxs += ctx_vectors.size(0)

        global_q_vector = torch.cat(global_q_vector, dim=0)
        global_ctxs_vector = torch.cat(global_ctxs_vector, dim=0)

    else:
        global_q_vector = local_q_vector
        global_ctxs_vector = local_ctx_vectors
        positive_idx_per_question = local_positive_idxs
        hard_negatives_per_question = local_hard_negatives_idxs

    loss, is_correct = loss_function.calc(
        global_q_vector,
        global_ctxs_vector,
        positive_idx_per_question,
        hard_negatives_per_question,
    )

    return loss, is_correct


def _do_biencoder_fwd_pass(
    model: nn.Module,
    input: BiEncoderBatch,
    tensorizer: Tensorizer,
    cfg,
    encoder_type: str,
    rep_positions=0,
) -> Tuple[torch.Tensor, int]:
    # logger.info('encoder_type %s', encoder_type)
    input = BiEncoderBatch(**move_to_device(input._asdict(), cfg.device))

    q_attn_mask = tensorizer.get_attn_mask(input.question_ids)
    ctx_attn_mask = tensorizer.get_attn_mask(input.context_ids)
    if model.training:
        model_out = model(
            input.question_ids,
            input.question_segments,
            q_attn_mask,
            input.context_ids,
            input.ctx_segments,
            ctx_attn_mask,
            encoder_type=encoder_type,
            representation_token_pos=rep_positions,
        )
    else:
        with torch.no_grad():
            model_out = model(
                input.question_ids,
                input.question_segments,
                q_attn_mask,
                input.context_ids,
                input.ctx_segments,
                ctx_attn_mask,
                encoder_type,
                representation_token_pos=rep_positions,
            )

    local_q_vector, local_ctx_vectors = model_out

    loss_function = BiEncoderNllLoss()

    loss, is_correct = _calc_loss(
        cfg,
        loss_function,
        local_q_vector,
        local_ctx_vectors,
        input.is_positive,
        input.hard_negatives,
    )

    is_correct = is_correct.sum().item()

    if cfg.n_gpu > 1:
        loss = loss.mean()
    if cfg.train.gradient_accumulation_steps > 1:
        loss = loss / cfg.train.gradient_accumulation_steps
    return loss, is_correct


@hydra.main(config_path="conf", config_name="biencoder_async_train_cfg")
def main(cfg: DictConfig):
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_training_params(parser)
    add_tokenizer_params(parser)

    # bi-encoder specific training features
    # parser.add_argument("--eval_per_epoch", default=1, type=int,
    #                    help="How many times it evaluates on dev set per epoch and saves a checkpoint")

    parser.add_argument(
        "--global_loss_buf_sz",
        type=int,
        default=150000,
        help='Buffer size for distributed mode representations al gather operation. \
                                Increase this if you see errors like "encoded data exceeds max_size ..."',
    )

    parser.add_argument("--fix_ctx_encoder", action="store_true")
    parser.add_argument("--shuffle_positive_ctx", action="store_true")

    # input/output src params
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model checkpoints will be written or resumed from",
    )

    # data handling parameters
    parser.add_argument(
        "--hard_negatives",
        default=1,
        type=int,
        help="amount of hard negative ctx per question",
    )
    parser.add_argument(
        "--other_negatives",
        default=0,
        type=int,
        help="amount of 'other' negative ctx per question",
    )
    parser.add_argument(
        "--train_files_upsample_rates",
        type=str,
        help="list of up-sample rates per each train file. Example: [1,2,1]",
    )

    # parameters for Av.rank validation method
    parser.add_argument(
        "--val_av_rank_start_epoch",
        type=int,
        default=10000,
        help="Av.rank validation: the epoch from which to enable this validation",
    )
    parser.add_argument(
        "--val_av_rank_hard_neg",
        type=int,
        default=30,
        help="Av.rank validation: how many hard negatives to take from each question pool",
    )
    parser.add_argument(
        "--val_av_rank_other_neg",
        type=int,
        default=30,
        help="Av.rank validation: how many 'other' negatives to take from each question pool",
    )
    parser.add_argument(
        "--val_av_rank_bsz",
        type=int,
        default=128,
        help="Av.rank validation: batch size to process passages",
    )
    parser.add_argument(
        "--val_av_rank_max_qs",
        type=int,
        default=10000,
        help="Av.rank validation: max num of questions",
    )
    parser.add_argument(
        "--checkpoint_file_name",
        type=str,
        default="dpr_biencoder",
        help="Checkpoints file prefix",
    )

    args, _argv = parser.parse_known_args()

    if cfg.train.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                cfg.train.gradient_accumulation_steps
            )
        )

    if cfg.output_dir is not None:
        os.makedirs(cfg.output_dir, exist_ok=True)

    cfg = setup_args_gpu(cfg)
    set_seed(cfg)

    logger.info("CFG (after gpu  configuration):")
    logger.info("%s", cfg.pretty())

    trainer = BiEncoderTrainer(cfg)

    if cfg.train_datasets and len(cfg.train_datasets) > 0:
        trainer.run_train()
    elif cfg.model_file and cfg.dev_datasets:
        logger.info(
            "No train files are specified. Run 2 types of validation for specified model file"
        )
        trainer.validate_nll()
        trainer.validate_average_rank()
    else:
        logger.warning(
            "Neither train_file or (model_file & dev_file) parameters are specified. Nothing to do."
        )


if __name__ == "__main__":

    logger.info("Sys.argv: %s", sys.argv)
    hydra_formatted_args = []

    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--") :])
        else:
            hydra_formatted_args.append(arg)
    logger.info("Hydra formatted Sys.argv: %s", hydra_formatted_args)
    sys.argv = hydra_formatted_args

    main()
