import logging
import random

import numpy as np
import torch
from typing import List
from torch import Tensor as T
from torch import nn


from dpr.data.speech_data import BiEncoderMixedSample
from dpr.models.biencoder import BiEncoder, BiEncoderBatch

from dpr.models.hf_models import (
    HFBertEncoder,
    get_optimizer,
    get_bert_tensorizer,
    get_hf_model_param_grouping,
    get_optimizer_grouped,
    get_wav2vec_encoder,
)

from dpr.models.fairseq_models import Wav2Vec2Encoder, HubertEncoder
from dpr.utils.data_utils import Tensorizer

logger = logging.getLogger(__name__)


def get_audio_mixed_biencoder_components(cfg, inference_only: bool = False, **kwargs):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    logger.info("Initializing Mixed Encoder")

    if cfg.encoder.pretrained_wav2vec_model_cfg:
        question_encoder = get_wav2vec_encoder(
            cfg.encoder.pretrained_wav2vec_model_cfg,
            cfg.encoder.wav2vec_max_audio_t,
            cfg.encoder.wav2_vec_extra_proj_dim,
            cfg.encoder.wav2vec_dropout,
            cfg.encoder.wav2vec_use_activation,
        )

    elif cfg.encoder.wav2vec_cp_file:  # fairseq model

        if cfg.encoder.encoder_model_type == "mixed_hf_bert_wav2vec":
            audio_cls = Wav2Vec2Encoder
        elif cfg.encoder.encoder_model_type == "mixed_hf_bert_hubert":
            audio_cls = HubertEncoder
        else:
            raise ValueError(f"{cfg.encoder.encoder_model_type} is not supported.")

        question_encoder = audio_cls(
            cfg.encoder.wav2vec_cp_file,
            cfg.encoder.wav2vec_apply_mask,
            cfg.encoder.wav2vec_max_audio_t,
            use_tanh=cfg.encoder.wav2vec_use_activation,
            dropout=cfg.encoder.wav2vec_dropout,
        )
    else:
        raise RuntimeError("Either pretrained_wav2vec_model_cfg or wav2vec_cp_file should be defined")

    ctx_encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs,
    )

    fix_ctx_encoder = cfg.fix_ctx_encoder if hasattr(cfg, "fix_ctx_encoder") else False

    biencoder = MixedBiEncoder(question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder)
    tensorizer = get_bert_tensorizer(cfg)

    if not hasattr(cfg, "train"):
        return tensorizer, biencoder, None

    lr = cfg.train.learning_rate
    if cfg.encoder.audio_encoder_lr_factor != 0:
        groups = get_hf_model_param_grouping(biencoder.ctx_model, weight_decay=cfg.train.weight_decay)
        q_groups = get_hf_model_param_grouping(biencoder.question_model, weight_decay=cfg.train.weight_decay)
        for g in q_groups:
            g["lr"] = lr * cfg.encoder.audio_encoder_lr_factor
            logger.info("Setting lr=%s for wav2vec encoder param group", g["lr"])
            groups.append(g)
        optimizer = get_optimizer_grouped(groups, learning_rate=lr, adam_eps=cfg.train.adam_eps)
    else:
        optimizer = (
            get_optimizer(
                biencoder,
                learning_rate=lr,
                adam_eps=cfg.train.adam_eps,
                weight_decay=cfg.train.weight_decay,
            )
            if not inference_only
            else None
        )

    return tensorizer, biencoder, optimizer


# TODO reduce code copy-paste
class MixedBiEncoder(BiEncoder):
    def create_biencoder_input(
        self,
        samples: List[BiEncoderMixedSample],
        tensorizer: Tensorizer,
        insert_title: bool,
        num_hard_negatives: int = 0,
        num_other_negatives: int = 0,
        shuffle: bool = True,
        shuffle_positives: bool = False,
        hard_neg_fallback: bool = True,
        query_token: str = None,
    ) -> BiEncoderBatch:
        """
        Creates a batch of the biencoder training tuple.
        :param samples: list of BiEncoderSample-s to create the batch for
        :param tensorizer: components to create model input tensors from a text sequence
        :param insert_title: enables title insertion at the beginning of the context sequences
        :param num_hard_negatives: amount of hard negatives per question (taken from samples' pools)
        :param num_other_negatives: amount of other negatives per question (taken from samples' pools)
        :param shuffle: shuffles negative passages pools
        :param shuffle_positives: shuffles positive passages pools
        :return: BiEncoderBatch tuple
        """

        question_tensors = []
        ctx_tensors = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []

        max_audio_vector_len = 0
        for sample in samples:
            # ctx+ & [ctx-] composition
            # as of now, take the first(gold) ctx+ only

            if shuffle and shuffle_positives:
                positive_ctxs = sample.positive_passages
                positive_ctx = positive_ctxs[np.random.choice(len(positive_ctxs))]
            else:
                positive_ctx = sample.positive_passages[0]

            neg_ctxs = sample.negative_passages
            hard_neg_ctxs = sample.hard_negative_passages
            question_tensor = sample.query

            # TODO: introduce seed based rnd
            if shuffle:
                random.shuffle(neg_ctxs)
                random.shuffle(hard_neg_ctxs)

            if hard_neg_fallback and len(hard_neg_ctxs) == 0:
                hard_neg_ctxs = neg_ctxs[0:num_hard_negatives]

            neg_ctxs = neg_ctxs[0:num_other_negatives]
            hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]

            all_ctxs = [positive_ctx] + neg_ctxs + hard_neg_ctxs
            hard_negatives_start_idx = 1
            hard_negatives_end_idx = 1 + len(hard_neg_ctxs)

            current_ctxs_len = len(ctx_tensors)

            sample_ctxs_tensors = [
                tensorizer.text_to_tensor(ctx.text, title=ctx.title if (insert_title and ctx.title) else None)
                for ctx in all_ctxs
            ]

            ctx_tensors.extend(sample_ctxs_tensors)
            positive_ctx_indices.append(current_ctxs_len)
            hard_neg_ctx_indices.append(
                [
                    i
                    for i in range(
                        current_ctxs_len + hard_negatives_start_idx,
                        current_ctxs_len + hard_negatives_end_idx,
                    )
                ]
            )
            question_tensors.append(question_tensor)
            max_audio_vector_len = max(max_audio_vector_len, question_tensor.size(1))

        # TODO: _pad_to_len move to utils
        from dpr.models.reader import _pad_to_len

        question_tensors = [_pad_to_len(q.squeeze(0), 0, max_audio_vector_len) for q in question_tensors]

        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_tensors], dim=0)
        questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0)

        ctx_segments = torch.zeros_like(ctxs_tensor)
        question_segments = torch.zeros_like(questions_tensor)

        return BiEncoderBatch(
            questions_tensor,
            question_segments,
            ctxs_tensor,
            ctx_segments,
            positive_ctx_indices,
            hard_neg_ctx_indices,
            "question",
        )
