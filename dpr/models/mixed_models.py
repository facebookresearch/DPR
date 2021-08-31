import logging
import random
from typing import Tuple, List

import numpy as np
import torch

from dpr.data.speech_data import BiEncoderMixedSample
from dpr.models.biencoder import BiEncoderBatch, BiEncoder

from dpr.models.hf_models import (
    HFBertEncoder,
    get_hf_model_param_grouping,
    get_optimizer_grouped,
    get_wav2vec_encoder,
    get_hubert_encoder,
    get_bert_tensorizer_p,
)
from dpr.utils.data_utils import Tensorizer

logger = logging.getLogger(__name__)


def get_audio_mixed_biencoder_components(cfg, inference_only: bool = False, **kwargs):
    logger.info("Initializing Mixed Encoder")

    question_encoder = get_query_encoder(cfg)
    ctx_encoder, tensorizer = get_ctx_encoder(cfg)

    fix_ctx_encoder = cfg.fix_ctx_encoder if hasattr(cfg, "fix_ctx_encoder") else False

    if cfg.encoder.q_encoder_type == "hf-quantized":
        biencoder = BiEncoder(question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder)
    else:
        # del question_encoder.masked_spec_embed
        logger.info("!!! question_encoder state %s", question_encoder.state_dict().keys())
        biencoder = MixedBiEncoder(question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder)

    if not hasattr(cfg, "train"):  # eval mode
        return tensorizer, biencoder, None

    # init optimizer
    lr = cfg.train.learning_rate
    if cfg.encoder.optimizer == "hf-adam":
        if cfg.encoder.q_audio_encoder_lr_factor != 0:
            groups = get_hf_model_param_grouping(biencoder.ctx_model, weight_decay=cfg.train.weight_decay)
            q_groups = get_hf_model_param_grouping(biencoder.question_model, weight_decay=cfg.train.weight_decay)
            for g in q_groups:
                g["lr"] = lr * cfg.encoder.q_audio_encoder_lr_factor
                logger.info("Setting lr=%s for wav2vec encoder param group", g["lr"])
                groups.append(g)
        else:
            groups = get_hf_model_param_grouping(biencoder, weight_decay=cfg.train.weight_decay)
        optimizer = get_optimizer_grouped(groups, learning_rate=lr, adam_eps=cfg.train.adam_eps)
    else:
        from dpr.models.fairseq_models import get_fairseq_adamw_optimizer

        optimizer = get_fairseq_adamw_optimizer(biencoder, cfg)

    return tensorizer, biencoder, optimizer


def get_query_encoder(cfg):
    # TODO: unify initialization
    if cfg.encoder.q_encoder_type == "hf-hubert" and cfg.encoder.q_encoder_model_cfg:  # HF-based
        query_encoder = get_hubert_encoder(
            cfg.encoder.q_encoder_model_cfg,
            cfg.encoder.q_max_audio_t,
            cfg.encoder.q_projection_dim,
            cfg.encoder.q_dropout,
            cfg.encoder.q_use_activation,
            cfg.encoder.q_output_layer,
        )
    elif cfg.encoder.q_encoder_type == "hf-wav2vec" and cfg.encoder.q_encoder_model_cfg:  # HF-based
        query_encoder = get_wav2vec_encoder(
            cfg.encoder.q_encoder_model_cfg,
            cfg.encoder.q_max_audio_t,
            cfg.encoder.q_projection_dim,
            cfg.encoder.q_dropout,
            cfg.encoder.q_use_activation,
            cfg.encoder.q_output_layer,
        )
    elif cfg.encoder.q_encoder_type == "hf-quantized":
        assert cfg.encoder.q_encoder_cp_file, (
            "q_encoder_cp_file should be set to point to quantization pre-trained " "checkpoint"
        )
        query_encoder = HFBertEncoder.init_encoder(
            cfg.encoder.q_encoder_model_cfg,
            projection_dim=cfg.encoder.q_projection_dim,
            dropout=cfg.encoder.q_dropout,
            pretrained=True,
        )
        cp_state = torch.load(cfg.encoder.q_encoder_cp_file)
        key_shift = len("bert.")
        logger.info("Loading pre-trained q-encoder weights from %s", cfg.encoder.q_encoder_cp_file)
        cp_state = {k[key_shift:]: v for k, v in cp_state.items() if k.startswith("bert.")}
        query_encoder.load_state_dict(cp_state, strict=False)

    elif cfg.encoder.q_encoder_cp_file:  # Fairseq based
        from dpr.models.fairseq_models import (
            Wav2Vec2Encoder,
            HubertEncoder,
        )

        if cfg.encoder.q_encoder_type == "fairseq-wav2vec":
            audio_cls = Wav2Vec2Encoder
        elif cfg.encoder.q_encoder_type == "fairseq-hubert":
            audio_cls = HubertEncoder
        else:
            raise ValueError(f"{cfg.encoder.q_encoder_type} is not supported.")

        query_encoder = audio_cls(
            cfg.encoder.q_encoder_cp_file,
            cfg.encoder.q_wav2vec_apply_mask,
            cfg.encoder.q_max_audio_t,
            use_tanh=cfg.encoder.q_use_activation,
            dropout=cfg.encoder.q_dropout,
            output_layer=cfg.encoder.q_output_layer,
        )
    else:
        raise RuntimeError("Either q_encoder_model_cfg or q_encoder_cp_file should be defined")
    return query_encoder


def get_ctx_encoder(cfg) -> Tuple[torch.nn.Module, Tensorizer]:
    # TODO: unify initialization
    if cfg.encoder.ctx_encoder_type == "hf-bert":  # HF-based

        ctx_encoder = HFBertEncoder.init_encoder(
            cfg.encoder.ctx_model_cfg,
            projection_dim=cfg.encoder.ctx_projection_dim,
            dropout=cfg.encoder.ctx_dropout,
            pretrained=cfg.encoder.ctx_pretrained,
        )
        tensorizer = get_bert_tensorizer_p(
            cfg.encoder.ctx_model_cfg, cfg.encoder.ctx_sequence_length, cfg.do_lower_case, cfg.special_tokens
        )

        # TODO: separate method to init tokenizer & tensorizer
        if cfg.encoder.q_encoder_type == "hf-quantized":
            # TODO: move to a sub-routine
            #  --------------------------------------------
            new_tokens_prefix = "w2v"
            new_tokens = ["[" + new_tokens_prefix + str(i) + "]" for i in range(100)]
            from dpr.models.hf_models import _add_special_tokens

            _add_special_tokens(tensorizer.tokenizer, new_tokens)
            # ----------------------------------------------------------------------------

    elif cfg.encoder.ctx_encoder_type == "fairseq-roberta":  # Fairseq based
        from dpr.models.fairseq_models import get_roberta_encoder_components

        ctx_encoder, tensorizer = get_roberta_encoder_components(
            cfg.encoder.ctx_pretrained_file,
            cfg.encoder.ctx_model_cfg,
            cfg.do_lower_case,
            cfg.encoder.ctx_sequence_length,
        )

    else:
        raise RuntimeError("encoder.ctx_encoder_type should be defined")
    return ctx_encoder, tensorizer


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
