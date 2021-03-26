#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Encoder model wrappers based on Fairseq code
"""

import logging
from typing import Tuple

import torch
from fairseq.models.roberta.hub_interface import RobertaHubInterface
from fairseq.models.roberta.model import RobertaModel as FaiseqRobertaModel
from torch import Tensor as T
from torch import nn

import fairseq
from dpr.models.hf_models import get_roberta_tensorizer
from fairseq.optim.adam import FairseqAdam
from .biencoder import BiEncoder

logger = logging.getLogger(__name__)


def get_roberta_biencoder_components(args, inference_only: bool = False, **kwargs):
    question_encoder = RobertaEncoder.from_pretrained(args.pretrained_file)
    ctx_encoder = RobertaEncoder.from_pretrained(args.pretrained_file)
    biencoder = BiEncoder(question_encoder, ctx_encoder)
    optimizer = (
        get_fairseq_adamw_optimizer(biencoder, args) if not inference_only else None
    )

    tensorizer = get_roberta_tensorizer(args)

    return tensorizer, biencoder, optimizer


def get_fairseq_adamw_optimizer(model: nn.Module, args):
    setattr(args, "lr", [args.learning_rate])
    return FairseqAdam(args, model.parameters()).optimizer


class RobertaEncoder(nn.Module):
    def __init__(self, fairseq_roberta_hub: RobertaHubInterface):
        super(RobertaEncoder, self).__init__()
        self.fairseq_roberta = fairseq_roberta_hub

    @classmethod
    def from_pretrained(cls, pretrained_dir_path: str):
        model = FaiseqRobertaModel.from_pretrained(pretrained_dir_path)
        return cls(model)

    def forward(
        self, input_ids: T, token_type_ids: T, attention_mask: T
    ) -> Tuple[T, ...]:
        roberta_out = self.fairseq_roberta.extract_features(input_ids)
        cls_out = roberta_out[:, 0, :]
        return roberta_out, cls_out, None

    def get_out_size(self):
        raise NotImplementedError


class Wav2Vec2Encoder(nn.Module):
    def __init__(
        self,
        cp_file: str,
        apply_mask: bool,
        max_audio_t: int,
        use_tanh: bool = True,
        dropout: float = 0.0,
    ):
        super(Wav2Vec2Encoder, self).__init__()
        state = fairseq.checkpoint_utils.load_checkpoint_to_cpu(cp_file)
        w2v_args = state["args"]
        task = fairseq.tasks.setup_task(w2v_args)
        model = task.build_model(w2v_args)
        model.load_state_dict(state["model"], strict=True)
        logger.info(
            "Initialized Wav2Vec2Encoder model as %s, from cp=%s, use_tanh=%s, dropout=%s",
            type(model),
            cp_file,
            use_tanh,
            dropout,
        )
        if isinstance(model, fairseq.models.wav2vec.wav2vec2.Wav2Vec2Model):
            self.wav2vec_model = model
            hidden_size = model.post_extract_proj.out_features
        else:
            self.wav2vec_model = model.w2v_encoder.w2v_model
            hidden_size = self.wav2vec_model.post_extract_proj.out_features

        self.max_audio_t = max_audio_t * hidden_size
        logger.info("Wav2Vec2Encoder max_audio_t %s", self.max_audio_t)

        self.dense = nn.Linear(self.max_audio_t, hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

        self.apply_mask = apply_mask
        self.use_tanh = use_tanh

        # TODO: remove after debug
        self.tmp_long_audio_samples = 0

    def forward(
        self,
        input_ids: T,
        _token_type_ids: T,
        attention_mask: T,
        representation_token_pos=0,
        output_layer=None
    ) -> Tuple[T, ...]:
        mask = self.apply_mask and self.training

        # TODO: remove after debug
        torch.cuda.ipc_collect()

        wav2vec_out, pad_mask = self.wav2vec_model.extract_features(
            input_ids, padding_mask=attention_mask, mask=mask
        )

        B, T, C = wav2vec_out.size()

        flat_encoded_out = wav2vec_out.reshape(B, -1)
        if T > self.max_audio_t:
            logger.warning("T>max_audio_t: %d>%d", T, self.max_audio_t)

        # TODO: make a util method
        def pad_to_len(
            seq,
            max_len,
        ):
            s_len = seq.size(0)
            # TODO: remove after debug
            if s_len > max_len:
                self.tmp_long_audio_samples += 1
                if self.tmp_long_audio_samples % 100 == 0:
                    logger.info(
                        "tmp_long_audio_samples %s", self.tmp_long_audio_samples
                    )

                return seq[0:max_len]
            r = torch.cat(
                [
                    seq,
                    torch.Tensor()
                    .new_full((max_len - s_len,), 0, dtype=torch.float)
                    .to(flat_encoded_out.device),
                ],
                dim=0,
            )
            return r

        flat_encoded_out = torch.cat(
            [
                pad_to_len(flat_encoded_out[i], self.max_audio_t).view(1, -1)
                for i in range(B)
            ],
            dim=0,
        )

        pooled_output = self.dense(flat_encoded_out)

        if self.use_tanh:
            pooled_output = self.activation(pooled_output)

        if self.training:
            pooled_output = self.dropout(pooled_output)

        return None, pooled_output, None

    def get_out_size(self):
        return self.wav2vec_model.post_extract_proj.out_features


class HubertEncoder(nn.Module):
    def __init__(
        self,
        cp_file: str,
        apply_mask: bool,
        max_audio_t: int,
        use_tanh: bool = True,
        dropout: float = 0.0,
    ):
        super(HubertEncoder, self).__init__()
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_file])
        self.model = models[0]
        logger.info(
            "Initialized HubertEncoder model as %s, from cp=%s, use_tanh=%s, dropout=%s",
            type(self.model),
            cp_file,
            use_tanh,
            dropout,
        )
        self.hidden_size = self.model.post_extract_proj.out_features

        self.max_audio_t = max_audio_t
        logger.info("HubertEncoder max_audio_t %s", self.max_audio_t)

        self.dense = nn.Linear(self.max_audio_t, self.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

        self.apply_mask = apply_mask
        self.use_tanh = use_tanh

        # TODO: remove after debug
        self.tmp_long_audio_samples = 0

    def forward(
        self,
        input_ids: T,
        _token_type_ids: T,
        padding_mask: T,
        representation_token_pos=0,
        output_layer=None
    ) -> Tuple[T, ...]:
        mask = self.apply_mask and self.training
        
        # TODO: remove after debug
        torch.cuda.ipc_collect()

        features, padding_mask = self.model.extract_features(
            input_ids, padding_mask=padding_mask, mask=mask, output_layer=output_layer
        )

        bsz, seq_len, feature_dim = features.size()
        assert self.hidden_size == feature_dim

        flat_encoded_out = features.reshape(bsz, -1)
        if seq_len > self.max_audio_t:
            logger.warning("Audio length exceeds > max_audio_t: %d>%d", T, self.max_audio_t)

        # TODO: make a util method
        def pad_to_len(
            seq,
            max_len,
        ):
            s_len = seq.size(0)
            # TODO: remove after debug
            if s_len > max_len:
                self.tmp_long_audio_samples += 1
                if self.tmp_long_audio_samples % 100 == 0:
                    logger.info(
                        "tmp_long_audio_samples %s", self.tmp_long_audio_samples
                    )

                return seq[0:max_len]
            r = torch.cat(
                [
                    seq,
                    torch.Tensor()
                    .new_full((max_len - s_len,), 0, dtype=torch.float)
                    .to(flat_encoded_out.device),
                ],
                dim=0,
            )
            return r

        flat_encoded_out = torch.cat(
            [
                pad_to_len(flat_encoded_out[i], self.max_audio_t).view(1, -1)
                for i in range(bsz)
            ],
            dim=0,
        )

        pooled_output = self.dense(flat_encoded_out)

        if self.use_tanh:
            pooled_output = self.activation(pooled_output)

        if self.training:
            pooled_output = self.dropout(pooled_output)

        return None, pooled_output, None

    def get_out_size(self):
        return self.hidden_size
