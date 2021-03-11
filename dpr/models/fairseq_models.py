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
import fairseq

from typing import Tuple
from fairseq.models.roberta.hub_interface import RobertaHubInterface
from fairseq.models.roberta.model import RobertaModel as FaiseqRobertaModel
from fairseq.optim.adam import FairseqAdam
from torch import Tensor as T
from torch import nn

from dpr.models.hf_models import get_roberta_tensorizer
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
    ):
        super(Wav2Vec2Encoder, self).__init__()
        state = fairseq.checkpoint_utils.load_checkpoint_to_cpu(cp_file)
        w2v_args = state["args"]
        task = fairseq.tasks.setup_task(w2v_args)
        model = task.build_model(w2v_args)
        model.load_state_dict(state["model"], strict=True)
        logger.info("Initialized Wav2Vec2Encoder model as %s", type(model))
        self.wav2vec_model = model.w2v_encoder.w2v_model
        hidden_size = model.projmodel.w2v_encoder.proj.out_features

        self.max_audio_t = max_audio_t
        self.dense = nn.Linear(max_audio_t, hidden_size)
        self.activation = nn.Tanh()
        self.apply_mask = apply_mask

    def forward(
        self, input_ids: T, _token_type_ids: T, attention_mask: T
    ) -> Tuple[T, ...]:
        mask = self.apply_mask and self.training
        wav2vec_out = self.wav2vec_model.extract_features(
            self, input_ids, padding_mask=attention_mask, mask=mask
        )
        encoded_out = wav2vec_out["x"]  # B x T x C
        B, T, C = encoded_out.size()

        logger.info("Wav2Vec2Encoder out %s", encoded_out.size())

        flat_encoded_out = encoded_out.view(B, -1)
        if T > self.max_audio_t:
            logger.warning("T>max_audio_t: %d>%d", T, self.max_audio_t)

        pooled_output = self.dense(flat_encoded_out)
        pooled_output = self.activation(pooled_output)
        logger.info("Wav2Vec2Encoder pooled out %s", pooled_output.size())
        return pooled_output
