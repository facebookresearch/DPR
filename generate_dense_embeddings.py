#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool that produces embeddings for a large documents base based on the pretrained ctx & question encoders
 Supposed to be used in a 'sharded' way to speed up the process.
"""
import csv
import logging
import os
import pickle

import argparse
import numpy as np
import pathlib
import torch
from torch import nn
from typing import List, Tuple, Dict

from dpr.data.biencoder_data import JsonLTablesQADataset
from dpr.data.tables import read_nq_tables_jsonl, Table
from dpr.models import init_biencoder_components
from dpr.options import add_encoder_params, setup_args_gpu, print_args, set_encoder_params_from_state, \
    add_tokenizer_params, add_cuda_params
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint, move_to_device

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


def gen_ctx_vectors(ctx_rows: List[Tuple[object, str, str]], model: nn.Module, tensorizer: Tensorizer,
                    insert_title: bool = True) -> List[Tuple[object, np.array]]:
    n = len(ctx_rows)
    bsz = args.batch_size
    total = 0
    results = []
    for j, batch_start in enumerate(range(0, n, bsz)):

        batch_token_tensors = [tensorizer.text_to_tensor(ctx[1], title=ctx[2] if insert_title else None) for ctx in
                               ctx_rows[batch_start:batch_start + bsz]]

        ctx_ids_batch = move_to_device(torch.stack(batch_token_tensors, dim=0), args.device)
        ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch), args.device)
        ctx_attn_mask = move_to_device(tensorizer.get_attn_mask(ctx_ids_batch), args.device)
        with torch.no_grad():
            _, out, _ = model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask)
        out = out.cpu()

        ctx_ids = [r[0] for r in ctx_rows[batch_start:batch_start + bsz]]
        extra_info = []
        if len(ctx_rows[0]) > 3:
            extra_info = [r[3:] for r in ctx_rows[batch_start:batch_start + bsz]]

        assert len(ctx_ids) == out.size(0)

        total += len(ctx_ids)

        # TODO: refactor to avoid 'if'
        if extra_info:
            results.extend([
                (ctx_ids[i], out[i].view(-1).numpy(), *extra_info[i])
                for i in range(out.size(0))
            ])
        else:
            results.extend([
                (ctx_ids[i], out[i].view(-1).numpy())
                for i in range(out.size(0))
            ])

        if total % 10 == 0:
            logger.info('Encoded passages %d', total)

    return results


def split_tables_to_chunks(tables_dict: Dict[str, Table], max_table_len: int, split_type: str) -> List[
    Tuple[int, str, str, int]]:
    tables_as_dicts = [t.to_dpr_json() for k, t in tables_dict.items()]

    chunk_id_to_table_id = []
    chunks = []
    chunk_id = 0
    for i, t in enumerate(tables_as_dicts):
        if split_type == 'type2':
            table_chunks = JsonLTablesQADataset.split_table2(t, max_table_len)
        else:
            table_chunks = JsonLTablesQADataset.split_table(t, max_table_len)
        title = t['caption']
        for c in table_chunks:
            # chunk id , text, title, external_id
            chunks.append((chunk_id, c, title, i))
            chunk_id += 1
        chunk_id_to_table_id += [i] * len(table_chunks)
        if i % 1000 == 0:
            logger.info('Splitted %d tables to %d chunks', i, len(chunks))
    return chunks


def main(args):
    saved_state = load_states_from_checkpoint(args.model_file)
    set_encoder_params_from_state(saved_state.encoder_params, args)
    print_args(args)

    tensorizer, encoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)

    encoder = encoder.ctx_model

    encoder, _ = setup_for_distributed_mode(encoder, None, args.device, args.n_gpu,
                                            args.local_rank,
                                            args.fp16,
                                            args.fp16_opt_level)
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info('Loading saved model state ...')
    logger.debug('saved model keys =%s', saved_state.model_dict.keys())

    prefix_len = len('ctx_model.')
    ctx_state = {key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if
                 key.startswith('ctx_model.')}
    model_to_load.load_state_dict(ctx_state)

    logger.info('reading data from file=%s', args.ctx_file)

    rows = []

    if args.new_chunks:
        with open(args.ctx_file, "rt", newline="") as fin:
            reader = csv.DictReader(fin, delimiter="\t")
            # file format: doc_id, doc_text, title
            rows.extend([(row['id'], row['text'], row['wikipedia_title']) for row in reader])
    elif args.tables_as_passages:
        logger.info('Parsing Tables ctx db')
        tables_dict = read_nq_tables_jsonl(args.ctx_file)
        rows.extend(split_tables_to_chunks(tables_dict, args.tables_chunk_sz))
    else:
        with open(args.ctx_file) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            # file format: doc_id, doc_text, title
            rows.extend([(row[0], row[1], row[2]) for row in reader if row[0] != 'id'])

    shard_size = int(len(rows) / args.num_shards)
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size

    logger.info('Producing encodings for passages range: %d to %d (out of total %d)', start_idx, end_idx, len(rows))
    rows = rows[start_idx:end_idx]

    data = gen_ctx_vectors(rows, encoder, tensorizer, True)

    file = args.out_file + '_' + str(args.shard_id)
    pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
    logger.info('Writing results to %s' % file)
    with open(file, mode='wb') as f:
        pickle.dump(data, f)

    logger.info('Total passages processed %d. Written to %s', len(data), file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument('--ctx_file', type=str, default=None, help='Path to passages set .tsv file')
    parser.add_argument('--out_file', required=True, type=str, default=None,
                        help='output .tsv file path to write results to ')
    parser.add_argument('--shard_id', type=int, default=0, help="Number(0-based) of data shard to process")
    parser.add_argument('--num_shards', type=int, default=1, help="Total amount of data shards")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for the passage encoder forward pass")
    parser.add_argument('--new_chunks', action='store_true')
    parser.add_argument('--tables_as_passages', action='store_true')

    # tmp param
    # parser.add_argument('--enable_st_selector', action='store_true')
    parser.add_argument('--special_tokens', type=str, default="['[START_ENT]','[END_ENT]']", )
    parser.add_argument('--tables_chunk_sz', type=int, default=100, )

    args = parser.parse_args()

    assert args.model_file, 'Please specify --model_file checkpoint to init model weights'

    setup_args_gpu(args)

    main(args)
