#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Encoder model wrappers based on HuggingFace code
"""

import logging
from typing import Tuple, List

import torch
import transformers
from torch import Tensor as T
from torch import nn

if transformers.__version__.startswith("4"):
    from transformers import BertConfig, BertModel
    from transformers import AdamW
    from transformers import BertTokenizer
    from transformers import RobertaTokenizer
    from transformers import Wav2Vec2Model, Wav2Vec2Config
else:
    from transformers.modeling_bert import BertConfig, BertModel
    from transformers.optimization import AdamW
    from transformers.tokenization_bert import BertTokenizer
    from transformers.tokenization_roberta import RobertaTokenizer

    # from transformers import Wav2Vec2Model, Wav2Vec2Config  # will fail

from dpr.models.biencoder import BiEncoder
from dpr.utils.data_utils import Tensorizer
from .reader import Reader

logger = logging.getLogger(__name__)


def get_bert_biencoder_components(cfg, inference_only: bool = False, **kwargs):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    question_encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )
    ctx_encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )

    fix_ctx_encoder = cfg.encoder.fix_ctx_encoder if hasattr(cfg.encoder, "fix_ctx_encoder") else False

    biencoder = BiEncoder(question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder)

    optimizer = (
        get_optimizer(
            biencoder,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_bert_tensorizer(cfg)
    return tensorizer, biencoder, optimizer


def get_bert_reader_components(cfg, inference_only: bool = False, **kwargs):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )

    hidden_size = encoder.config.hidden_size
    reader = Reader(encoder, hidden_size)

    optimizer = (
        get_optimizer(
            reader,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_bert_tensorizer(cfg)
    return tensorizer, reader, optimizer


# TODO: unify tensorizer init methods
def get_bert_tensorizer(cfg):
    sequence_length = cfg.encoder.sequence_length
    pretrained_model_cfg = cfg.encoder.pretrained_model_cfg

    tokenizer = get_bert_tokenizer(pretrained_model_cfg, do_lower_case=cfg.do_lower_case)
    if cfg.special_tokens:
        _add_special_tokens(tokenizer, cfg.special_tokens)

    return BertTensorizer(tokenizer, sequence_length)


def get_bert_tensorizer_p(
    pretrained_model_cfg: str, sequence_length: int, do_lower_case: bool = True, special_tokens: List[str] = []
):
    tokenizer = get_bert_tokenizer(pretrained_model_cfg, do_lower_case=do_lower_case)
    if special_tokens:
        _add_special_tokens(tokenizer, special_tokens)

    return BertTensorizer(tokenizer, sequence_length)


def _add_special_tokens(tokenizer, special_tokens):
    logger.info("Adding special tokens %s", special_tokens)
    logger.info("!!! tokenizer=%s", type(tokenizer))
    special_tokens_num = len(special_tokens)
    # TODO: this is a hack-y logic that uses some private tokenizer structure which can be changed in HF code
    assert special_tokens_num < 101
    unused_ids = [tokenizer.vocab["[unused{}]".format(i)] for i in range(special_tokens_num)]
    logger.info("Utilizing the following unused token ids %s", unused_ids)

    for idx, id in enumerate(unused_ids):
        old_token = "[unused{}]".format(idx)
        del tokenizer.vocab[old_token]
        new_token = special_tokens[idx]
        tokenizer.vocab[new_token] = id
        tokenizer.ids_to_tokens[id] = new_token
        # logging.info("new token %s id=%s", new_token, id)

    tokenizer.additional_special_tokens = list(special_tokens)
    logger.info("additional_special_tokens %s", tokenizer.additional_special_tokens)
    logger.info("all_special_tokens_extended: %s", tokenizer.all_special_tokens_extended)
    logger.info("additional_special_tokens_ids: %s", tokenizer.additional_special_tokens_ids)
    logger.info("all_special_tokens %s", tokenizer.all_special_tokens)

    logger.info("!!! test tokenize %s", tokenizer.tokenize("[CLS] [w2v60] [w2v19] [w2v46][SEP]does"))
    enc = tokenizer.encode("[CLS] [w2v60] [w2v19] [w2v46] [w2v24][SEP] does")
    logger.info("!!! test encode %s", enc)
    logger.info("!!! test decode %s", tokenizer.decode(enc))

    # for st in special_tokens:
    #    logging.info("Special token=%s id=%s", st, tokenizer.convert_tokens_to_ids([st]))


def get_roberta_tensorizer(pretrained_model_cfg: str, do_lower_case: bool, sequence_length: int):
    tokenizer = get_roberta_tokenizer(pretrained_model_cfg, do_lower_case=do_lower_case)
    return RobertaTensorizer(tokenizer, sequence_length)


def get_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-5,
    adam_eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    optimizer_grouped_parameters = get_hf_model_param_grouping(model, weight_decay)
    return get_optimizer_grouped(optimizer_grouped_parameters, learning_rate, adam_eps)


def get_hf_model_param_grouping(
    model: nn.Module,
    weight_decay: float = 0.0,
):
    no_decay = ["bias", "LayerNorm.weight"]

    return [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]


def get_optimizer_grouped(
    optimizer_grouped_parameters: List,
    learning_rate: float = 1e-5,
    adam_eps: float = 1e-8,
) -> torch.optim.Optimizer:

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    return optimizer


def get_bert_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    return BertTokenizer.from_pretrained(pretrained_cfg_name, do_lower_case=do_lower_case)


def get_roberta_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    # still uses HF code for tokenizer since they are the same
    return RobertaTokenizer.from_pretrained(pretrained_cfg_name, do_lower_case=do_lower_case)


def get_wav2vec_encoder(
    pretrained_model: str,
    max_audio_t: int,
    extra_proj_d: int,
    final_drop: float,
    use_activation: bool,
    output_layer: int,
):
    encoder = Wav2Vec2HFEncoder.init_encoder(
        pretrained_model, max_audio_t, extra_proj_d, final_drop, use_activation, output_layer=output_layer
    )
    return encoder


class HFBertEncoder(BertModel):
    def __init__(self, config, project_dim: int = 0):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.encode_proj = nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        self.init_weights()

    @classmethod
    def init_encoder(
        cls, cfg_name: str, projection_dim: int = 0, dropout: float = 0.1, pretrained: bool = True, **kwargs
    ) -> BertModel:
        logger.info("Initializing HF BERT Encoder. cfg_name=%s", cfg_name)
        cfg = BertConfig.from_pretrained(cfg_name if cfg_name else "bert-base-uncased")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout

        if pretrained:
            return cls.from_pretrained(cfg_name, config=cfg, project_dim=projection_dim, **kwargs)
        else:
            return HFBertEncoder(cfg, project_dim=projection_dim)

    def forward(
        self,
        input_ids: T,
        token_type_ids: T,
        attention_mask: T,
        representation_token_pos=0,
    ) -> Tuple[T, ...]:

        out = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        # HF >4.0 version support
        if transformers.__version__.startswith("4") and isinstance(
            out,
            transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions,
        ):
            sequence_output = out.last_hidden_state
            pooled_output = None
            hidden_states = out.hidden_states

        elif self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = out
        else:
            hidden_states = None
            out = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
            sequence_output, pooled_output = out

        if isinstance(representation_token_pos, int):
            pooled_output = sequence_output[:, representation_token_pos, :]
        else:  # treat as a tensor
            bsz = sequence_output.size(0)
            assert representation_token_pos.size(0) == bsz, "query bsz={} while representation_token_pos bsz={}".format(
                bsz, representation_token_pos.size(0)
            )
            pooled_output = torch.stack([sequence_output[i, representation_token_pos[i, 1], :] for i in range(bsz)])

        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    # TODO: make a super class for all encoders
    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size


class Wav2Vec2HFEncoder(Wav2Vec2Model):
    def __init__(
        self, config, max_audio_t: int, project_dim, final_dropout: float, use_activation: bool, output_layer: int = -1
    ):
        Wav2Vec2Model.__init__(self, config)
        hidden_size = config.hidden_size

        self.max_audio = max_audio_t * hidden_size
        logger.info(
            "Wav2Vec2HFEncoder: max_audio_t %s, project_dim %s, dropout %s, use_activation %s, output_layer %d",
            self.max_audio,
            project_dim,
            final_dropout,
            use_activation,
            output_layer,
        )

        self.dense = nn.Linear(self.max_audio, hidden_size) if project_dim != 0 else None
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(final_dropout)

        # self.apply_mask = apply_mask
        self.use_activation = use_activation
        self.init_weights()

        self.output_layer = output_layer

    @classmethod
    def init_encoder(
        cls,
        cfg_name: str,
        max_audio_t: int,
        projection_dim: int,
        dropout: float,
        use_activation: bool,
        pretrained: bool = True,
        output_layer: int = -1,
    ) -> Wav2Vec2Model:
        logger.info("Initializing HF Wav2Vec2Model Encoder. cfg_name=%s", cfg_name)
        cfg_name = cfg_name if cfg_name else "facebook/wav2vec2-base-960h"
        cfg = Wav2Vec2Config.from_pretrained(cfg_name)
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout

        if pretrained:
            return cls.from_pretrained(
                cfg_name,
                max_audio_t=max_audio_t,
                config=cfg,
                project_dim=projection_dim,
                final_dropout=dropout,
                use_activation=use_activation,
                output_layer=output_layer,
            )
        else:
            return Wav2Vec2HFEncoder(cfg, max_audio_t, projection_dim, dropout, use_activation, output_layer)

    def forward(
        self,
        input_ids: T,
        _token_type_ids: T,
        attention_mask: T,
        representation_token_pos=0,
    ) -> Tuple[T, ...]:

        wav2vec_out = super().forward(input_ids, output_hidden_states=True)  # attention_mask=attention_mask
        wav2vec_out = (
            wav2vec_out.last_hidden_state
            if self.output_layer in [None, -1]
            else wav2vec_out.hidden_states[self.output_layer]
        )
        B, T, C = wav2vec_out.size()

        if self.dense:
            flat_encoded_out = wav2vec_out.reshape(B, -1)
            if flat_encoded_out.size(1) > self.max_audio:
                logger.warning("TxC>max_audio_t: %d>%d", flat_encoded_out.size(1), self.max_audio)

            # TODO: make a util method
            def pad_to_len(
                seq,
                max_len,
            ):
                s_len = seq.size(0)
                if s_len > max_len:
                    return seq[0:max_len]
                r = torch.cat(
                    [
                        seq,
                        torch.Tensor().new_full((max_len - s_len,), 0, dtype=torch.float).to(flat_encoded_out.device),
                    ],
                    dim=0,
                )
                return r

            flat_encoded_out = torch.cat(
                [pad_to_len(flat_encoded_out[i], self.max_audio).view(1, -1) for i in range(B)],
                dim=0,
            )
            wav2vec_out = self.dense(flat_encoded_out)

            if self.use_activation:
                wav2vec_out = self.activation(wav2vec_out)

            if self.training:
                wav2vec_out = self.dropout(wav2vec_out)
        else:
            wav2vec_out = wav2vec_out[:, representation_token_pos, :]

        return None, wav2vec_out, None

    def get_out_size(self):
        return self.config.hidden_size


class BertTensorizer(Tensorizer):
    def __init__(self, tokenizer: BertTokenizer, max_length: int, pad_to_max: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max

    def text_to_tensor(
        self,
        text: str,
        title: str = None,
        add_special_tokens: bool = True,
        apply_max_len: bool = True,
    ):
        text = text.strip()
        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        # TODO: move max len to methods params?

        if title:
            token_ids = self.tokenizer.encode(
                title,
                text_pair=text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length if apply_max_len else 10000,
                pad_to_max_length=False,
                truncation=True,
            )
        else:
            token_ids = self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length if apply_max_len else 10000,
                pad_to_max_length=False,
                truncation=True,
            )

        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (seq_len - len(token_ids))
        if len(token_ids) >= seq_len:
            token_ids = token_ids[0:seq_len] if apply_max_len else token_ids
            token_ids[-1] = self.tokenizer.sep_token_id

        return torch.tensor(token_ids)

    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    def get_attn_mask(self, tokens_tensor: T) -> T:
        return tokens_tensor != self.get_pad_id()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad

    def get_token_id(self, token: str) -> int:
        return self.tokenizer.vocab[token]


class RobertaTensorizer(BertTensorizer):
    def __init__(self, tokenizer, max_length: int, pad_to_max: bool = True):
        super(RobertaTensorizer, self).__init__(tokenizer, max_length, pad_to_max=pad_to_max)
