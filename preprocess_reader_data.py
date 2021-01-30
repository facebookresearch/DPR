#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Reader data preprocessor command line tool
"""
import argparse
import logging

from dpr.data.reader_data import convert_retriever_results
from dpr.models import init_tenzorizer
from dpr.options import (
    print_args,
    add_encoder_params,
    add_reader_preprocessing_params,
    add_tokenizer_params,
)

logger = logging.getLogger()


def main(args):
    tensorizer = init_tenzorizer(args.encoder_model_type, args)

    # disable auto-padding to save disk space of serialized files
    tensorizer.set_pad_to_max(False)

    convert_retriever_results(
        args.is_train_set,
        args.retriever_results,
        args.out_file,
        args.gold_passages_src,
        tensorizer,
        args.num_workers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_reader_preprocessing_params(parser)

    parser.add_argument(
        "--is_train_set",
        action="store_true",
        help="If true, the data will be binarised for train model usage (split into ctx+ and ctx- \
                        and with answer spans selected)",
    )
    parser.add_argument(
        "--retriever_results",
        required=True,
        type=str,
        help="File with retriever results file(json format)",
    )
    parser.add_argument(
        "--out_file",
        required=True,
        type=str,
        help="The file to write serialized results to",
    )

    args = parser.parse_args()

    print_args(args)

    main(args)
