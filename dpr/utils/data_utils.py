#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for general purpose data processing
"""

import json
import logging
import math
import pickle
import random
from typing import List, Iterator, Callable, Tuple

import torch
from torch import Tensor as T

logger = logging.getLogger()


def read_serialized_data_from_files(paths: List[str]) -> List:
    results = []
    for i, path in enumerate(paths):
        with open(path, "rb") as reader:
            logger.info('Reading file %s', path)
            data = pickle.load(reader)
            results.extend(data)
            logger.info('Aggregated data size: {}'.format(len(results)))
    logger.info('Total data size: {}'.format(len(results)))
    return results


def read_data_from_json_files(paths: List[str], upsample_rates: List = None) -> List:
    results = []
    if upsample_rates is None:
        upsample_rates = [1] * len(paths)

    assert len(upsample_rates) == len(paths), 'up-sample rates parameter doesn\'t match input files amount'

    for i, path in enumerate(paths):
        with open(path, 'r', encoding="utf-8") as f:
            logger.info('Reading file %s' % path)
            data = json.load(f)
            upsample_factor = int(upsample_rates[i])
            data = data * upsample_factor
            results.extend(data)
            logger.info('Aggregated data size: {}'.format(len(results)))
    return results


class ShardedDataIterator(object):
    """
    General purpose data iterator to be used for Pytorch's DDP mode where every node should handle its own part of
    the data.
    Instead of cutting data shards by their min size, it sets the amount of iterations by the maximum shard size.
    It fills the extra sample by just taking first samples in a shard.
    It can also optionally enforce identical batch size for all iterations (might be useful for DP mode).
    """

    def __init__(self, data: torch.utils.data.Dataset, shard_id: int = 0, num_shards: int = 1, batch_size: int = 1,
                 shuffle=True,
                 shuffle_seed: int = 0, offset: int = 0,
                 strict_batch_size: bool = False
                 ):

        self.data = data
        total_size = len(data)

        self.shards_num = max(num_shards, 1)
        self.shard_id = max(shard_id, 0)

        samples_per_shard = math.ceil(total_size / self.shards_num)

        self.shard_start_idx = self.shard_id * samples_per_shard

        self.shard_end_idx = min(self.shard_start_idx + samples_per_shard, total_size)

        if strict_batch_size:
            self.max_iterations = math.ceil(samples_per_shard / batch_size)
        else:
            self.max_iterations = int(samples_per_shard / batch_size)

        logger.info(
            'samples_per_shard=%d, shard_start_idx=%d, shard_end_idx=%d, max_iterations=%d', samples_per_shard,
            self.shard_start_idx,
            self.shard_end_idx,
            self.max_iterations)

        self.iteration = offset  # to track in-shard iteration status
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.shuffle_seed = shuffle_seed
        self.strict_batch_size = strict_batch_size

    def total_data_len(self) -> int:
        return len(self.data)

    def iterations_num(self) -> int:
        return self.max_iterations - self.iteration

    # TODO remove
    def iterate_data(self, epoch: int = 0) -> Iterator[List]:

        # tmp: always assumes it is JsonQADataset object
        data = self.data.data

        if self.shuffle:
            # to be able to resume, same shuffling should be used when starting from a failed/stopped iteration
            epoch_rnd = random.Random(self.shuffle_seed + epoch)
            epoch_rnd.shuffle(data)

        # if resuming iteration somewhere in the middle of epoch, one needs to adjust max_iterations
        max_iterations = self.max_iterations - self.iteration

        shard_samples = data[self.shard_start_idx:self.shard_end_idx]
        for i in range(self.iteration * self.batch_size, len(shard_samples), self.batch_size):
            items = shard_samples[i:i + self.batch_size]
            if self.strict_batch_size and len(items) < self.batch_size:
                logger.debug('Extending batch to max size')
                items.extend(shard_samples[0:self.batch_size - len(items)])
            self.iteration += 1
            yield items

        # some shards may done iterating while the others are at the last batch. Just return the first batch
        while self.iteration < max_iterations:
            logger.debug('Fulfilling non complete shard='.format(self.shard_id))
            self.iteration += 1
            batch = shard_samples[0:self.batch_size]
            yield batch

        logger.debug('Finished iterating, iteration={}, shard={}'.format(self.iteration, self.shard_id))
        # reset the iteration status
        self.iteration = 0

    def get_iteration(self) -> int:
        return self.iteration

    def apply(self, visitor_func: Callable):
        for sample in self.data:
            visitor_func(sample)

    def iterate_ds_data(self, epoch: int = 0) -> Iterator[List]:
        indices = list(range(len(self.data)))

        if self.shuffle:
            # to be able to resume, same shuffling should be used when starting from a failed/stopped iteration
            epoch_rnd = random.Random(self.shuffle_seed + epoch)
            epoch_rnd.shuffle(indices)

        # if resuming iteration somewhere in the middle of epoch, one needs to adjust max_iterations
        max_iterations = self.max_iterations - self.iteration

        shard_indices = indices[self.shard_start_idx:self.shard_end_idx]

        for i in range(self.iteration * self.batch_size, len(shard_indices), self.batch_size):
            items_idxs = shard_indices[i:i + self.batch_size]
            if self.strict_batch_size and len(items_idxs) < self.batch_size:
                logger.debug('Extending batch to max size')
                items_idxs.extend(shard_indices[0:self.batch_size - len(items)])
            self.iteration += 1
            items = [self.data[idx] for idx in items_idxs]
            yield items

        # some shards may done iterating while the others are at the last batch. Just return the first batch
        while self.iteration < max_iterations:
            logger.debug('Fulfilling non complete shard='.format(self.shard_id))
            self.iteration += 1
            items_idxs = shard_indices[0:self.batch_size]
            items = [self.data[idx] for idx in items_idxs]
            yield items

        logger.info('Finished iterating, iteration={}, shard={}'.format(self.iteration, self.shard_id))
        # reset the iteration status
        self.iteration = 0


class MultiSetDataIterator(object):
    """
    Iterator over multiple data sources. Useful when all samples form a single batch should be from the same dataset.
    """

    def __init__(self, datasets: List[ShardedDataIterator], shuffle_seed: int = 0, shuffle=True,
                 sampling_rates: List = [],
                 ):
        self.iterables = datasets
        data_lengths = [it.total_data_len() for it in datasets]
        self.total_data = sum(data_lengths)
        logger.info('Multi set data sizes %s', data_lengths)
        logger.info('Multi set total data %s', self.total_data)
        logger.info('Multi set sampling_rates %s', sampling_rates)
        self.shuffle_seed = shuffle_seed
        self.shuffle = shuffle
        self.iteration = 0

        if sampling_rates:
            self.max_iterations = sum(int(ds.max_iterations * sampling_rates[i]) for i, ds in enumerate(datasets))
        else:
            self.max_iterations = sum(ds.max_iterations for ds in datasets)
        logger.info('Multi set max_iterations %s', self.max_iterations)
        self.sampling_rates = sampling_rates

    def total_data_len(self) -> int:
        return self.total_data

    def iterate_ds_data(self, epoch: int = 0) -> Iterator[Tuple[List, int]]:

        data_lengths = [it.iterations_num() for it in self.iterables]

        # iterations_num = sum(data_lengths)

        logger.info('Multi set iteration: lengths per set: %s', data_lengths)
        logger.info('Multi set iteration: iteration per set: %s', [it.iteration for it in self.iterables])

        data_src_indices = []
        for source, source_len in enumerate(data_lengths):
            sampling_factor = 1
            if self.sampling_rates:
                sampling_factor = self.sampling_rates[source]
            logger.info('Multi set iteration: source %d, batches to be taken: %s', source, int(source_len * sampling_factor))
            data_src_indices.extend([source] * int(source_len * sampling_factor))

        if self.shuffle:
            # to be able to resume, same shuffling should be used when starting from a failed/stopped iteration
            epoch_rnd = random.Random(self.shuffle_seed + epoch)
            epoch_rnd.shuffle(data_src_indices)

        iterators = [it.iterate_ds_data(epoch=epoch) for it in self.iterables]

        for i, source_idx in enumerate(data_src_indices):
            it = iterators[source_idx]
            next_item = next(it, None)
            if next_item is not None:
                self.iteration += 1
                yield (next_item, source_idx)

        # tmp
        logger.info('Multi set iteration finished: iteration per set: %s', [it.iteration for it in self.iterables])
        [next(it, None) for it in iterators]

        # TODO: clear iterators in some non-hacky way
        for it in self.iterables:
            it.iteration = 0

        # reset the iteration status
        self.iteration = 0

    def get_iteration(self) -> int:
        return self.iteration


def normalize_question(question: str) -> str:
    if question[-1] == '?':
        question = question[:-1]
    return question


class Tensorizer(object):
    """
    Component for all text to model input data conversions and related utility methods
    """

    # Note: title, if present, is supposed to be put before text (i.e. optional title + document body)
    def text_to_tensor(self, text: str, title: str = None, add_special_tokens: bool = True, apply_max_len: bool =True):
        raise NotImplementedError

    def get_pair_separator_ids(self) -> T:
        raise NotImplementedError

    def get_pad_id(self) -> int:
        raise NotImplementedError

    def get_attn_mask(self, tokens_tensor: T):
        raise NotImplementedError

    def is_sub_word_id(self, token_id: int):
        raise NotImplementedError

    def to_string(self, token_ids, skip_special_tokens=True):
        raise NotImplementedError

    def set_pad_to_max(self, pad: bool):
        raise NotImplementedError

    def get_token_id(self, token: str) -> int:
        raise NotImplementedError
