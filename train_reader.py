#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
 Pipeline to train the reader model on top of the retriever results
"""

import argparse
import collections
import glob
import json
import logging
import os
from collections import defaultdict
from typing import List

import numpy as np
import torch

from dpr.data.qa_validation import exact_match_score
from dpr.data.reader_data import (
    ReaderSample,
    get_best_spans,
    SpanPrediction,
    convert_retriever_results,
)
from dpr.models import init_reader_components
from dpr.models.reader import create_reader_input, ReaderBatch, compute_loss
from dpr.options import (
    add_encoder_params,
    setup_args_gpu,
    set_seed,
    add_training_params,
    add_reader_preprocessing_params,
    set_encoder_params_from_state,
    get_encoder_params_state,
    add_tokenizer_params,
    print_args,
)
from dpr.utils.data_utils import (
    ShardedDataIterator,
    read_serialized_data_from_files,
    Tensorizer,
)
from dpr.utils.model_utils import (
    get_schedule_linear,
    load_states_from_checkpoint,
    move_to_device,
    CheckpointState,
    get_model_file,
    setup_for_distributed_mode,
    get_model_obj,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

ReaderQuestionPredictions = collections.namedtuple(
    "ReaderQuestionPredictions", ["id", "predictions", "gold_answers"]
)


class ReaderTrainer(object):
    def __init__(self, args):
        self.args = args

        self.shard_id = args.local_rank if args.local_rank != -1 else 0
        self.distributed_factor = args.distributed_world_size or 1

        logger.info("***** Initializing components for training *****")

        model_file = get_model_file(self.args, self.args.checkpoint_file_name)
        saved_state = None
        if model_file:
            saved_state = load_states_from_checkpoint(model_file)
            set_encoder_params_from_state(saved_state.encoder_params, args)

        tensorizer, reader, optimizer = init_reader_components(
            args.encoder_model_type, args
        )

        reader, optimizer = setup_for_distributed_mode(
            reader,
            optimizer,
            args.device,
            args.n_gpu,
            args.local_rank,
            args.fp16,
            args.fp16_opt_level,
        )
        self.reader = reader
        self.optimizer = optimizer
        self.tensorizer = tensorizer
        self.start_epoch = 0
        self.start_batch = 0
        self.scheduler_state = None
        self.best_validation_result = None
        self.best_cp_name = None
        if saved_state:
            self._load_saved_state(saved_state)

    def get_data_iterator(
        self,
        path: str,
        batch_size: int,
        is_train: bool,
        shuffle=True,
        shuffle_seed: int = 0,
        offset: int = 0,
    ) -> ShardedDataIterator:
        data_files = glob.glob(path)
        logger.info("Data files: %s", data_files)
        if not data_files:
            raise RuntimeError("No Data files found")
        preprocessed_data_files = self._get_preprocessed_files(data_files, is_train)
        data = read_serialized_data_from_files(preprocessed_data_files)

        iterator = ShardedDataIterator(
            data,
            shard_id=self.shard_id,
            num_shards=self.distributed_factor,
            batch_size=batch_size,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            offset=offset,
        )

        # apply deserialization hook
        iterator.apply(lambda sample: sample.on_deserialize())
        return iterator

    def run_train(self):
        args = self.args

        train_iterator = self.get_data_iterator(
            args.train_file,
            args.batch_size,
            True,
            shuffle=True,
            shuffle_seed=args.seed,
            offset=self.start_batch,
        )

        num_train_epochs = args.num_train_epochs - self.start_epoch

        logger.info("Total iterations per epoch=%d", train_iterator.max_iterations)
        updates_per_epoch = (
            train_iterator.max_iterations // args.gradient_accumulation_steps
        )
        total_updates = updates_per_epoch * num_train_epochs - self.start_batch
        logger.info(" Total updates=%d", total_updates)

        warmup_steps = args.warmup_steps
        scheduler = get_schedule_linear(
            self.optimizer, warmup_steps=warmup_steps, training_steps=total_updates
        )
        if self.scheduler_state:
            logger.info("Loading scheduler state %s", self.scheduler_state)
            scheduler.load_state_dict(self.scheduler_state)

        eval_step = args.eval_step
        logger.info("  Eval step = %d", eval_step)
        logger.info("***** Training *****")

        global_step = self.start_epoch * updates_per_epoch + self.start_batch

        for epoch in range(self.start_epoch, int(args.num_train_epochs)):
            logger.info("***** Epoch %d *****", epoch)
            global_step = self._train_epoch(
                scheduler, epoch, eval_step, train_iterator, global_step
            )

        if args.local_rank in [-1, 0]:
            logger.info(
                "Training finished. Best validation checkpoint %s", self.best_cp_name
            )

        return

    def validate_and_save(self, epoch: int, iteration: int, scheduler):
        args = self.args
        # in distributed DDP mode, save checkpoint for only one process
        save_cp = args.local_rank in [-1, 0]
        reader_validation_score = self.validate()

        if save_cp:
            cp_name = self._save_checkpoint(scheduler, epoch, iteration)
            logger.info("Saved checkpoint to %s", cp_name)

            if reader_validation_score < (self.best_validation_result or 0):
                self.best_validation_result = reader_validation_score
                self.best_cp_name = cp_name
                logger.info("New Best validation checkpoint %s", cp_name)

    def validate(self):
        logger.info("Validation ...")
        args = self.args
        self.reader.eval()
        data_iterator = self.get_data_iterator(
            args.dev_file, args.dev_batch_size, False, shuffle=False
        )

        log_result_step = args.log_batch_step
        all_results = []

        eval_top_docs = args.eval_top_docs
        for i, samples_batch in enumerate(data_iterator.iterate_data()):
            input = create_reader_input(
                self.tensorizer.get_pad_id(),
                samples_batch,
                args.passages_per_question_predict,
                args.sequence_length,
                args.max_n_answers,
                is_train=False,
                shuffle=False,
            )

            input = ReaderBatch(**move_to_device(input._asdict(), args.device))
            attn_mask = self.tensorizer.get_attn_mask(input.input_ids)

            with torch.no_grad():
                start_logits, end_logits, relevance_logits = self.reader(
                    input.input_ids, attn_mask
                )

            batch_predictions = self._get_best_prediction(
                start_logits,
                end_logits,
                relevance_logits,
                samples_batch,
                passage_thresholds=eval_top_docs,
            )

            all_results.extend(batch_predictions)

            if (i + 1) % log_result_step == 0:
                logger.info("Eval step: %d ", i)

        ems = defaultdict(list)

        for q_predictions in all_results:
            gold_answers = q_predictions.gold_answers
            span_predictions = (
                q_predictions.predictions
            )  # {top docs threshold -> SpanPrediction()}
            for (n, span_prediction) in span_predictions.items():
                em_hit = max(
                    [
                        exact_match_score(span_prediction.prediction_text, ga)
                        for ga in gold_answers
                    ]
                )
                ems[n].append(em_hit)
        em = 0
        for n in sorted(ems.keys()):
            em = np.mean(ems[n])
            logger.info("n=%d\tEM %.2f" % (n, em * 100))

        if args.prediction_results_file:
            self._save_predictions(args.prediction_results_file, all_results)

        return em

    def _train_epoch(
        self,
        scheduler,
        epoch: int,
        eval_step: int,
        train_data_iterator: ShardedDataIterator,
        global_step: int,
    ):
        args = self.args
        rolling_train_loss = 0.0
        epoch_loss = 0
        log_result_step = args.log_batch_step
        rolling_loss_step = args.train_rolling_loss_step

        self.reader.train()
        epoch_batches = train_data_iterator.max_iterations

        for i, samples_batch in enumerate(
            train_data_iterator.iterate_data(epoch=epoch)
        ):

            data_iteration = train_data_iterator.get_iteration()

            # enables to resume to exactly same train state
            if args.fully_resumable:
                np.random.seed(args.seed + global_step)
                torch.manual_seed(args.seed + global_step)
                if args.n_gpu > 0:
                    torch.cuda.manual_seed_all(args.seed + global_step)

            input = create_reader_input(
                self.tensorizer.get_pad_id(),
                samples_batch,
                args.passages_per_question,
                args.sequence_length,
                args.max_n_answers,
                is_train=True,
                shuffle=True,
            )

            loss = self._calc_loss(input)

            epoch_loss += loss.item()
            rolling_train_loss += loss.item()

            if args.fp16:
                from apex import amp

                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(self.optimizer), args.max_grad_norm
                    )
            else:
                loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.reader.parameters(), args.max_grad_norm
                    )

            global_step += 1

            if (i + 1) % args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                scheduler.step()
                self.reader.zero_grad()

            if global_step % log_result_step == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch: %d: Step: %d/%d, global_step=%d, lr=%f",
                    epoch,
                    data_iteration,
                    epoch_batches,
                    global_step,
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

            if global_step % eval_step == 0:
                logger.info(
                    "Validation: Epoch: %d Step: %d/%d",
                    epoch,
                    data_iteration,
                    epoch_batches,
                )
                self.validate_and_save(
                    epoch, train_data_iterator.get_iteration(), scheduler
                )
                self.reader.train()

        epoch_loss = (epoch_loss / epoch_batches) if epoch_batches > 0 else 0
        logger.info("Av Loss per epoch=%f", epoch_loss)
        return global_step

    def _save_checkpoint(self, scheduler, epoch: int, offset: int) -> str:
        args = self.args
        model_to_save = get_model_obj(self.reader)
        cp = os.path.join(
            args.output_dir,
            args.checkpoint_file_name
            + "."
            + str(epoch)
            + ("." + str(offset) if offset > 0 else ""),
        )

        meta_params = get_encoder_params_state(args)

        state = CheckpointState(
            model_to_save.state_dict(),
            self.optimizer.state_dict(),
            scheduler.state_dict(),
            offset,
            epoch,
            meta_params,
        )
        torch.save(state._asdict(), cp)
        return cp

    def _load_saved_state(self, saved_state: CheckpointState):
        epoch = saved_state.epoch
        offset = saved_state.offset
        if offset == 0:  # epoch has been completed
            epoch += 1
        logger.info("Loading checkpoint @ batch=%s and epoch=%s", offset, epoch)
        self.start_epoch = epoch
        self.start_batch = offset

        model_to_load = get_model_obj(self.reader)
        if saved_state.model_dict:
            logger.info("Loading model weights from saved state ...")
            model_to_load.load_state_dict(saved_state.model_dict)

        logger.info("Loading saved optimizer state ...")
        if saved_state.optimizer_dict:
            self.optimizer.load_state_dict(saved_state.optimizer_dict)
        self.scheduler_state = saved_state.scheduler_dict

    def _get_best_prediction(
        self,
        start_logits,
        end_logits,
        relevance_logits,
        samples_batch: List[ReaderSample],
        passage_thresholds: List[int] = None,
    ) -> List[ReaderQuestionPredictions]:

        args = self.args
        max_answer_length = args.max_answer_length
        questions_num, passages_per_question = relevance_logits.size()

        _, idxs = torch.sort(
            relevance_logits,
            dim=1,
            descending=True,
        )

        batch_results = []
        for q in range(questions_num):
            sample = samples_batch[q]

            non_empty_passages_num = len(sample.passages)
            nbest = []
            for p in range(passages_per_question):
                passage_idx = idxs[q, p].item()
                if (
                    passage_idx >= non_empty_passages_num
                ):  # empty passage selected, skip
                    continue
                reader_passage = sample.passages[passage_idx]
                sequence_ids = reader_passage.sequence_ids
                sequence_len = sequence_ids.size(0)
                # assuming question & title information is at the beginning of the sequence
                passage_offset = reader_passage.passage_offset

                p_start_logits = start_logits[q, passage_idx].tolist()[
                    passage_offset:sequence_len
                ]
                p_end_logits = end_logits[q, passage_idx].tolist()[
                    passage_offset:sequence_len
                ]

                ctx_ids = sequence_ids.tolist()[passage_offset:]
                best_spans = get_best_spans(
                    self.tensorizer,
                    p_start_logits,
                    p_end_logits,
                    ctx_ids,
                    max_answer_length,
                    passage_idx,
                    relevance_logits[q, passage_idx].item(),
                    top_spans=10,
                )
                nbest.extend(best_spans)
                if len(nbest) > 0 and not passage_thresholds:
                    break

            if passage_thresholds:
                passage_rank_matches = {}
                for n in passage_thresholds:
                    curr_nbest = [pred for pred in nbest if pred.passage_index < n]
                    passage_rank_matches[n] = curr_nbest[0]
                predictions = passage_rank_matches
            else:
                if len(nbest) == 0:
                    predictions = {
                        passages_per_question: SpanPrediction("", -1, -1, -1, "")
                    }
                else:
                    predictions = {passages_per_question: nbest[0]}
            batch_results.append(
                ReaderQuestionPredictions(sample.question, predictions, sample.answers)
            )
        return batch_results

    def _calc_loss(self, input: ReaderBatch) -> torch.Tensor:
        args = self.args
        input = ReaderBatch(**move_to_device(input._asdict(), args.device))
        attn_mask = self.tensorizer.get_attn_mask(input.input_ids)
        questions_num, passages_per_question, _ = input.input_ids.size()

        if self.reader.training:
            # start_logits, end_logits, rank_logits = self.reader(input.input_ids, attn_mask)
            loss = self.reader(
                input.input_ids,
                attn_mask,
                input.start_positions,
                input.end_positions,
                input.answers_mask,
            )

        else:
            # TODO: remove?
            with torch.no_grad():
                start_logits, end_logits, rank_logits = self.reader(
                    input.input_ids, attn_mask
                )

            loss = compute_loss(
                input.start_positions,
                input.end_positions,
                input.answers_mask,
                start_logits,
                end_logits,
                rank_logits,
                questions_num,
                passages_per_question,
            )
        if args.n_gpu > 1:
            loss = loss.mean()
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        return loss

    def _get_preprocessed_files(
        self,
        data_files: List,
        is_train: bool,
    ):

        serialized_files = [file for file in data_files if file.endswith(".pkl")]
        if serialized_files:
            return serialized_files
        assert len(data_files) == 1, "Only 1 source file pre-processing is supported."

        # data may have been serialized and cached before, try to find ones from same dir
        def _find_cached_files(path: str):
            dir_path, base_name = os.path.split(path)
            base_name = base_name.replace(".json", "")
            out_file_prefix = os.path.join(dir_path, base_name)
            out_file_pattern = out_file_prefix + "*.pkl"
            return glob.glob(out_file_pattern), out_file_prefix

        serialized_files, out_file_prefix = _find_cached_files(data_files[0])
        if serialized_files:
            logger.info("Found preprocessed files. %s", serialized_files)
            return serialized_files

        gold_passages_src = None
        if self.args.gold_passages_src:
            gold_passages_src = (
                self.args.gold_passages_src
                if is_train
                else self.args.gold_passages_src_dev
            )
            assert os.path.exists(
                gold_passages_src
            ), "Please specify valid gold_passages_src/gold_passages_src_dev"
        logger.info(
            "Data are not preprocessed for reader training. Start pre-processing ..."
        )

        # start pre-processing and save results
        def _run_preprocessing(tensorizer: Tensorizer):
            # temporarily disable auto-padding to save disk space usage of serialized files
            tensorizer.set_pad_to_max(False)
            serialized_files = convert_retriever_results(
                is_train,
                data_files[0],
                out_file_prefix,
                gold_passages_src,
                self.tensorizer,
                num_workers=self.args.num_workers,
            )
            tensorizer.set_pad_to_max(True)
            return serialized_files

        if self.distributed_factor > 1:
            # only one node in DDP model will do pre-processing
            if self.args.local_rank in [-1, 0]:
                serialized_files = _run_preprocessing(self.tensorizer)
                torch.distributed.barrier()
            else:
                torch.distributed.barrier()
                serialized_files = _find_cached_files(data_files[0])
        else:
            serialized_files = _run_preprocessing(self.tensorizer)

        return serialized_files

    def _save_predictions(
        self, out_file: str, prediction_results: List[ReaderQuestionPredictions]
    ):
        logger.info("Saving prediction results to  %s", out_file)
        with open(out_file, "w", encoding="utf-8") as output:
            save_results = []
            for r in prediction_results:
                save_results.append(
                    {
                        "question": r.id,
                        "gold_answers": r.gold_answers,
                        "predictions": [
                            {
                                "top_k": top_k,
                                "prediction": {
                                    "text": span_pred.prediction_text,
                                    "score": span_pred.span_score,
                                    "relevance_score": span_pred.relevance_score,
                                    "passage_idx": span_pred.passage_index,
                                    "passage": self.tensorizer.to_string(
                                        span_pred.passage_token_ids
                                    ),
                                },
                            }
                            for top_k, span_pred in r.predictions.items()
                        ],
                    }
                )
            output.write(json.dumps(save_results, indent=4) + "\n")


def main():
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_training_params(parser)
    add_tokenizer_params(parser)
    add_reader_preprocessing_params(parser)

    # reader specific params
    parser.add_argument(
        "--max_n_answers",
        default=10,
        type=int,
        help="Max amount of answer spans to marginalize per singe passage",
    )
    parser.add_argument(
        "--passages_per_question",
        type=int,
        default=2,
        help="Total amount of positive and negative passages per question",
    )
    parser.add_argument(
        "--passages_per_question_predict",
        type=int,
        default=50,
        help="Total amount of positive and negative passages per question for evaluation",
    )
    parser.add_argument(
        "--max_answer_length",
        default=10,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--eval_top_docs",
        nargs="+",
        type=int,
        help="top retrival passages thresholds to analyze prediction results for",
    )
    parser.add_argument("--checkpoint_file_name", type=str, default="dpr_reader")
    parser.add_argument(
        "--prediction_results_file",
        type=str,
        help="path to a file to write prediction results to",
    )

    # training parameters
    parser.add_argument(
        "--eval_step",
        default=2000,
        type=int,
        help="batch steps to run validation and save checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model checkpoints will be written to",
    )

    parser.add_argument(
        "--fully_resumable",
        action="store_true",
        help="Enables resumable mode by specifying global step dependent random seed before shuffling "
        "in-batch data",
    )

    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    setup_args_gpu(args)
    set_seed(args)
    print_args(args)

    trainer = ReaderTrainer(args)

    if args.train_file is not None:
        trainer.run_train()
    elif args.dev_file:
        logger.info("No train files are specified. Run validation.")
        trainer.validate()
    else:
        logger.warning(
            "Neither train_file or (model_file & dev_file) parameters are specified. Nothing to do."
        )


if __name__ == "__main__":
    main()
