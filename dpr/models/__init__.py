#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib

from transformers import HfArgumentParser, TrainingArguments

from coil.arguments import COILDataArguments, COILModelArguments
from utils.retriever_utils import to_namedtuple

"""
 'Router'-like set of methods for component initialization with lazy imports 
"""


def init_hf_bert_biencoder(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError("Please install transformers lib")
    from .hf_models import get_bert_biencoder_components

    return get_bert_biencoder_components(args, **kwargs)


def init_hf_bert_question_encoder(args, **kwargs):
    tensorizer, biencoder, optimizer = init_hf_bert_biencoder(args, **kwargs)
    return tensorizer, biencoder.question_model, optimizer


def init_hf_coil_question_encoder(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError("Please install transformers lib")

    # update args from default
    parser = HfArgumentParser((COILModelArguments, COILDataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_dict(dict(args.encoder))
    params = {
        "model_args": model_args,
        "data_args": data_args,
        "training_args": training_args
    }
    encoder_args = to_namedtuple(params)

    from coil.models.coil_model import get_coil_question_encoder_components
    return get_coil_question_encoder_components(encoder_args, **kwargs)


def init_hf_bert_reader(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError("Please install transformers lib")
    from .hf_models import get_bert_reader_components

    return get_bert_reader_components(args, **kwargs)


def init_pytext_bert_biencoder(args, **kwargs):
    if importlib.util.find_spec("pytext") is None:
        raise RuntimeError("Please install pytext lib")
    from .pytext_models import get_bert_biencoder_components

    return get_bert_biencoder_components(args, **kwargs)


def init_fairseq_roberta_biencoder(args, **kwargs):
    if importlib.util.find_spec("fairseq") is None:
        raise RuntimeError("Please install fairseq lib")
    from .fairseq_models import get_roberta_biencoder_components

    return get_roberta_biencoder_components(args, **kwargs)


def init_fairseq_roberta_question_encoder(args, **kwargs):
    tensorizer, biencoder, optimizer = init_fairseq_roberta_biencoder(args, **kwargs)
    return tensorizer, biencoder.question_model, optimizer


def init_hf_bert_tenzorizer(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError("Please install transformers lib")
    from .hf_models import get_bert_tensorizer

    return get_bert_tensorizer(args)


def init_hf_roberta_tenzorizer(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError("Please install transformers lib")
    from .hf_models import get_roberta_tensorizer

    return get_roberta_tensorizer(args)


BIENCODER_INITIALIZERS = {
    "hf_bert": init_hf_bert_biencoder,
    "pytext_bert": init_pytext_bert_biencoder,
    "fairseq_roberta": init_fairseq_roberta_biencoder,
}

QUESTION_ENCODER_INITIALIZERS = {
    "hf_bert": init_hf_bert_question_encoder,
    "fairseq_roberta": init_fairseq_roberta_question_encoder,
    "coil": init_hf_coil_question_encoder,
}

READER_INITIALIZERS = {
    "hf_bert": init_hf_bert_reader,
}

TENSORIZER_INITIALIZERS = {
    "hf_bert": init_hf_bert_tenzorizer,
    "hf_roberta": init_hf_roberta_tenzorizer,
    "pytext_bert": init_hf_bert_tenzorizer,  # using HF's code as of now
    "fairseq_roberta": init_hf_roberta_tenzorizer,  # using HF's code as of now
}


def init_comp(initializers_dict, type, args, **kwargs):
    if type in initializers_dict:
        return initializers_dict[type](args, **kwargs)
    else:
        raise RuntimeError("unsupported model type: {}".format(type))


def init_biencoder_components(encoder_type: str, args, **kwargs):
    return init_comp(BIENCODER_INITIALIZERS, encoder_type, args, **kwargs)


def init_question_encoder_components(encoder_type: str, args, **kwargs):
    return init_comp(QUESTION_ENCODER_INITIALIZERS, encoder_type, args, **kwargs)


def init_reader_components(encoder_type: str, args, **kwargs):
    return init_comp(READER_INITIALIZERS, encoder_type, args, **kwargs)


def init_tenzorizer(encoder_type: str, args, **kwargs):
    return init_comp(TENSORIZER_INITIALIZERS, encoder_type, args, **kwargs)
