import collections
import csv
import glob
import logging
import numpy as np
import os
import random
from typing import Dict, List, Tuple

import hydra
import jsonlines
import torch
from omegaconf import DictConfig
from torch import Tensor as T

from dpr.utils.data_utils import read_data_from_json_files, Tensorizer

logger = logging.getLogger(__name__)

BiEncoderPassage = collections.namedtuple('BiEncoderPassage', ['text', 'title', ])


class BiEncoderSample(object):
    query: str
    positive_passages: List[BiEncoderPassage]
    negative_passages: List[BiEncoderPassage]
    hard_negative_passages: List[BiEncoderPassage]


class RepTokenSelector(object):
    def get_positions(self, input_ids: T, tenzorizer: Tensorizer):
        raise NotImplementedError


class RepStaticPosTokenSelector(RepTokenSelector):
    def __init__(self, static_position: int = 0):
        self.static_position = static_position

    def get_positions(self, input_ids: T, tenzorizer: Tensorizer):
        return self.static_position


class RepSpecificTokenSelector(RepTokenSelector):
    def __init__(self, token: str = '[CLS]'):
        self.token = token
        self.token_id = None

    def get_positions(self, input_ids: T, tenzorizer: Tensorizer):
        if not self.token_id:
            self.token_id = tenzorizer.get_token_id(self.token)
        token_indexes = (input_ids == self.token_id).nonzero()
        # check if all samples in input_ids has index presence and out a default value otherwise
        bsz = input_ids.size(0)
        if bsz == token_indexes.size(0):
            return token_indexes

        token_indexes_result = []
        found_idx_cnt = 0
        for i in range(bsz):
            if found_idx_cnt < token_indexes.size(0) and token_indexes[found_idx_cnt][
                0] == i:  # this samples has the special token
                token_indexes_result.append(token_indexes[found_idx_cnt])
                found_idx_cnt += 1
            else:
                logger.warning('missing special token %s', input_ids[i])

                token_indexes_result.append(
                    torch.tensor([i, 0]))  # setting 0-th token, i.e. CLS for BERT as the special one
        token_indexes_result = torch.stack(token_indexes_result, dim=0)
        return token_indexes_result


class Dataset(torch.utils.data.Dataset):
    def __init__(self, selector: DictConfig):
        # TODO: move to conf_utils
        self.selector = hydra.utils.instantiate(selector)

    def __getitem__(self, index) -> BiEncoderSample:
        raise NotImplementedError


class JsonQADataset(Dataset):
    def __init__(self, data_file_pattern: str, selector: DictConfig):
        super().__init__(selector)
        self.data_files = glob.glob(data_file_pattern)
        self.data = []

    def load_data(self):
        data = read_data_from_json_files(self.data_files)
        # filter those without positive ctx
        self.data = [r for r in data if len(r['positive_ctxs']) > 0]
        logger.info('Total cleaned data size: {}'.format(len(self.data)))

    def __getitem__(self, index) -> BiEncoderSample:
        json_sample = self.data[index]
        r = BiEncoderSample()
        r.query = json_sample['question']
        r.positive_passages = [BiEncoderPassage(ctx['text'], ctx['title']) for ctx in json_sample['positive_ctxs']]
        r.negative_passages = [BiEncoderPassage(ctx['text'], ctx['title']) for ctx in
                               json_sample['negative_ctxs']]
        r.hard_negative_passages = [BiEncoderPassage(ctx['text'], ctx['title']) for ctx in
                                    json_sample['hard_negative_ctxs']]
        return r

    def __len__(self):
        return len(self.data)


class JsonLTablesQADataset(Dataset):
    def __init__(self, data_file_pattern: str, selector: DictConfig, is_train_set: bool,
                 shuffle_positives: bool = False,
                 max_negatives: int = 1, seed: int = 0, max_len=200
                 ):
        super().__init__(selector)
        self.data_files = glob.glob(data_file_pattern)
        self.data = []
        self.shuffle_positives = shuffle_positives
        self.is_train_set = is_train_set
        self.max_negatives = max_negatives
        self.rnd = random.Random(seed)
        self.max_len = max_len

    def load_data(self):
        data = []
        for path in self.data_files:
            with jsonlines.open(path, mode='r') as jsonl_reader:
                data += [jline for jline in jsonl_reader]

        # filter those without positive ctx
        self.data = [r for r in data if len(r['positive_ctxs']) > 0]
        logger.info('Total cleaned data size: {}'.format(len(self.data)))

    def __getitem__(self, index) -> BiEncoderSample:

        #logger.info('!!! get table item')
        json_sample = self.data[index]
        r = BiEncoderSample()
        r.query = json_sample['question']
        positive_ctxs = json_sample['positive_ctxs']
        hard_negative_ctxs = json_sample['hard_negative_ctxs']

        if self.shuffle_positives:
            self.rnd.shuffle(positive_ctxs)

        if self.is_train_set:
            self.rnd.shuffle(hard_negative_ctxs)

        #logger.info('!!! positive_ctxs %s', len(positive_ctxs))

        positive_ctxs = positive_ctxs[0:1]
        hard_negative_ctxs = hard_negative_ctxs[0:self.max_negatives]

        #logger.info('!!! hard_negative_ctxs %s', len(hard_negative_ctxs))

        r.positive_passages = [BiEncoderPassage(self._linearize_table(ctx, True), ctx['caption']) for ctx in
                               positive_ctxs]
        r.negative_passages = []
        r.hard_negative_passages = [BiEncoderPassage(self._linearize_table(ctx, False), ctx['caption']) for ctx in
                                    hard_negative_ctxs]
        return r

    def __len__(self):
        return len(self.data)

    def _linearize_table(self, t: dict, is_positive: bool) -> List[str]:
        rows = t['rows']
        selected_rows = set()
        rows_linearized = []
        total_words_len = 0

        # get the first non empty row as the "header"
        for i, r in enumerate(rows):
            row_lin, row_len = self._linearize_row(r)
            if len(row_lin) > 1:  # TODO: change to checking cell value tokens
                selected_rows.add(i)
                rows_linearized.append(row_lin)
                total_words_len += row_len
                break

        # split to chunks
        if is_positive:
            row_idx_with_answers = [ap[0] for ap in t['answer_pos']]
            #logger.info('!!! row_idx_with_answers %s', row_idx_with_answers)

            if self.shuffle_positives:
                self.rnd.shuffle(row_idx_with_answers)
            for i in row_idx_with_answers:
                if i not in selected_rows:
                    row_lin, row_len = self._linearize_row(rows[i])
                    selected_rows.add(i)
                    rows_linearized.append(row_lin)
                    total_words_len += row_len
                if total_words_len >= self.max_len:
                    break

        if total_words_len < self.max_len:  # append random rows

            if self.is_train_set:
                rows_indexes = np.random.permutation(range(len(rows)))
            else:
                rows_indexes = [*range(len(rows))]

            for i in rows_indexes:
                if i not in selected_rows:
                    row_lin, row_len = self._linearize_row(rows[i])
                    if len(row_lin) > 1:  # TODO: change to checking cell value tokens
                        selected_rows.add(i)
                        rows_linearized.append(row_lin)
                        total_words_len += row_len
                    if total_words_len >= self.max_len:
                        break

        linearized_str = ''
        for r in rows_linearized:
            linearized_str += (r + '\n')

        #logger.info('!!! selected_rows %s', selected_rows)
        #logger.info('!!! total_words_len %s', total_words_len)
        #logger.info('!!! positive linearized_str %s', linearized_str)

        return linearized_str

    def _linearize_row(self, row: dict) -> Tuple[str, int]:
        cell_values = [c['value'] for c in row['columns']]
        total_words = sum(len(c.split(' ')) for c in cell_values)
        return ', '.join([c['value'] for c in row['columns']]), total_words


class TRECDataset(Dataset):
    def __init__(self, data_dir: str, passages_filename: str, queries_filename: str, qrels_filename: str,
                 selector: DictConfig):
        super().__init__(selector)

        self.passages_file = os.path.join(data_dir, passages_filename)
        self.queries_file = os.path.join(data_dir, queries_filename)
        self.qidpidtriples_file = os.path.join(data_dir, qrels_filename)
        assert os.path.isfile(self.passages_file)
        assert os.path.isfile(self.queries_file)
        assert os.path.isfile(self.qidpidtriples_file)

        self.passages_dict = {}
        self.queries = []
        self.qrels = []

    def load_data(self):
        logger.info('Reading TREC dataset')
        logger.info('Reading passages list from: %s', self.passages_file)
        self.passages_dict = _load_tsv_as_dict(self.passages_file)
        logger.info('Total TREC passages: %d', len(self.passages_dict))

        logger.info('Reading queries from: %s', self.queries_file)
        self.queries = _load_tsv_as_dict(self.queries_file)
        logger.info('Total TREC queries: %d', len(self.queries))

        logger.info('Reading triplets data from: %s', self.qidpidtriples_file)
        self.qrels = _load_qrels(self.qidpidtriples_file)
        logger.info('Total TREC samples: %d', len(self.qrels))

    def __getitem__(self, index) -> BiEncoderSample:
        qrel = self.qrels[index]
        r = BiEncoderSample()
        r.query = self.queries[qrel[0]]
        r.positive_passages = [BiEncoderPassage(self.passages_dict[pid], None) for pid in qrel[1]]
        r.hard_negative_passages = [BiEncoderPassage(self.passages_dict[pid], None) for pid in qrel[2]]
        r.negative_passages = []
        return r

    def __len__(self):
        return len(self.qrels)


def _load_tsv_as_dict(ctx_file: str) -> Dict[int, str]:
    docs = {}
    with open(ctx_file) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t', )
        # file format: id, text
        for row in reader:
            if row[0] != 'id':
                docs[int(row[0])] = row[1]
    return docs


def _load_qrels(qrels_file: str, max_passages_per_q: int = 30) -> List[Tuple[int, List[int], List[int]]]:
    """
    Loads TREC's (Triples QID PID) Format
    :param qrels_file:
    :param max_passages_per_q:
    :return: result tuple format: qid, positive pid-s, negative pid-s
    """

    result = {}
    with open(qrels_file) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t', )
        # file format: qid, positive pid, negative pid
        for row in reader:
            if row[0] != 'id':
                qid = int(row[0])

                q_info = result.get(qid, ([], []))

                if len(q_info[0]) < max_passages_per_q:
                    pid = int(row[1])
                    if pid not in q_info[0]:  # assuming positive pid is usually ~ 1 sample
                        q_info[0].append(pid)

                if len(q_info[1]) < max_passages_per_q:
                    pid = int(row[2])
                    q_info[1].append(pid)  # assuming negatives are unique per question
                result[qid] = q_info

            # tmp:
            # if len(result)>100:
            #    break

    # convert to list
    return [(k, v[0], v[1]) for k, v in result.items()]
