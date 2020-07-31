import collections
import csv
import glob
import logging
import os
from typing import Dict, List, Tuple

import torch

from dpr.utils.data_utils import read_data_from_json_files

logger = logging.getLogger(__name__)

BiEncoderPassage = collections.namedtuple('BiEncoderPassage', ['text', 'title', ])


class BiEncoderSample(object):
    query: str
    positive_passages: List[BiEncoderPassage]
    negative_passages: List[BiEncoderPassage]
    hard_negative_passages: List[BiEncoderPassage]


class JsonQADataset(torch.utils.data.Dataset):
    def __init__(self, data_file_pattern: str):
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


class TRECDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, passages_filename: str, queries_filename: str, qrels_filename: str):
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
            if len(result)>100:
                break

    # convert to list
    return [(k, v[0], v[1]) for k, v in result.items()]
