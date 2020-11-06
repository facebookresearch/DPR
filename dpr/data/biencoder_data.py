import collections
import csv
import glob
import logging
import os
import random
from typing import Dict, List, Tuple

import hydra
import jsonlines
import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor as T

from dpr.data.tables import Table
from dpr.utils.data_utils import read_data_from_json_files, Tensorizer

logger = logging.getLogger(__name__)
BiEncoderPassage = collections.namedtuple("BiEncoderPassage", ["text", "title"])


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
    def __init__(self, token: str = "[CLS]"):
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
            if (
                found_idx_cnt < token_indexes.size(0)
                and token_indexes[found_idx_cnt][0] == i
            ):
                # this samples has the special token
                token_indexes_result.append(token_indexes[found_idx_cnt])
                found_idx_cnt += 1
            else:
                logger.warning("missing special token %s", input_ids[i])

                token_indexes_result.append(
                    torch.tensor([i, 0]).to(input_ids.device)
                )  # setting 0-th token, i.e. CLS for BERT as the special one
        token_indexes_result = torch.stack(token_indexes_result, dim=0)
        return token_indexes_result


DEFAULT_SELECTOR = RepStaticPosTokenSelector()


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        selector: DictConfig = None,
        special_token: str = None,
        shared_encoder: bool = False,
        shuffle_positives: bool = False,
        query_special_suffix: str = None,
    ):
        if selector:
            self.selector = hydra.utils.instantiate(selector)
        else:
            self.selector = DEFAULT_SELECTOR
        self.special_token = special_token
        self.shared_encoder = shared_encoder
        self.shuffle_positives = shuffle_positives
        self.query_special_suffix = query_special_suffix

    def __getitem__(self, index) -> BiEncoderSample:
        raise NotImplementedError

    def _process_query(self, query: str):
        if self.query_special_suffix and not query.endswith(self.query_special_suffix):
            query += self.query_special_suffix
        return query


class JsonQADataset(Dataset):
    def __init__(
        self,
        data_file_pattern: str,
        selector: DictConfig = None,
        special_token: str = None,
        shared_encoder: bool = False,
        shuffle_positives: bool = False,
        normalize: bool = False,
        query_special_suffix: str = None,
    ):
        super().__init__(
            selector,
            special_token=special_token,
            shared_encoder=shared_encoder,
            shuffle_positives=shuffle_positives,
            query_special_suffix=query_special_suffix,
        )
        self.data_files = glob.glob(data_file_pattern)
        self.data = []
        self.normalize = normalize

    def load_data(self):
        data = read_data_from_json_files(self.data_files)
        # filter those without positive ctx
        self.data = [r for r in data if len(r["positive_ctxs"]) > 0]
        logger.info("Total cleaned data size: {}".format(len(self.data)))

    def __getitem__(self, index) -> BiEncoderSample:
        json_sample = self.data[index]
        r = BiEncoderSample()
        r.query = self._process_query(json_sample["question"])
        r.positive_passages = [
            BiEncoderPassage(ctx["text"], ctx["title"])
            for ctx in json_sample["positive_ctxs"]
        ]
        r.negative_passages = (
            [
                BiEncoderPassage(ctx["text"], ctx["title"])
                for ctx in json_sample["negative_ctxs"]
            ]
            if "negative_ctxs" in json_sample
            else []
        )
        if "hard_negative_ctxs" in json_sample:
            r.hard_negative_passages = [
                BiEncoderPassage(ctx["text"], ctx["title"])
                for ctx in json_sample["hard_negative_ctxs"]
            ]
        else:
            r.hard_negative_passages = []

        # tmp experiment
        if self.normalize:
            r.positive_passages = [
                BiEncoderPassage(normalize_kilt_passage(p.text), p.title)
                for p in r.positive_passages
            ]
            r.negative_passages = [
                BiEncoderPassage(normalize_kilt_passage(p.text), p.title)
                for p in r.negative_passages
            ]
            r.hard_negative_passages = [
                BiEncoderPassage(normalize_kilt_passage(p.text), p.title)
                for p in r.hard_negative_passages
            ]
        return r

    def __len__(self):
        return len(self.data)

    def get_qas(self) -> Tuple[List[str], List[str]]:
        return [s["question"] for s in self.data], [s["answers"] for s in self.data]

    def get_qas_range(
        self, start_idx: int, end_idx: int
    ) -> Tuple[List[str], List[str]]:
        return (
            [s["question"] for s in self.data[start_idx:end_idx]],
            [s["answers"] for s in self.data[start_idx:end_idx]],
        )


class JsonLTablesQADataset(Dataset):
    def __init__(
        self,
        data_file_pattern: str,
        is_train_set: bool,
        selector: DictConfig = None,
        shuffle_positives: bool = False,
        max_negatives: int = 1,
        seed: int = 0,
        max_len=100,
        split_type: str = "type1",
    ):
        super().__init__(selector, shuffle_positives=shuffle_positives)
        self.data_files = glob.glob(data_file_pattern)
        self.data = []
        self.is_train_set = is_train_set
        self.max_negatives = max_negatives
        self.rnd = random.Random(seed)
        self.max_len = max_len
        self.linearize_func = JsonLTablesQADataset.get_lin_func(split_type)

    def load_data(self):
        data = []
        for path in self.data_files:
            with jsonlines.open(path, mode="r") as jsonl_reader:
                data += [jline for jline in jsonl_reader]

        # filter those without positive ctx
        self.data = [r for r in data if len(r["positive_ctxs"]) > 0]
        logger.info("Total cleaned data size: {}".format(len(self.data)))

    def __getitem__(self, index) -> BiEncoderSample:
        json_sample = self.data[index]
        r = BiEncoderSample()
        r.query = json_sample["question"]
        positive_ctxs = json_sample["positive_ctxs"]
        hard_negative_ctxs = json_sample["hard_negative_ctxs"]

        if self.shuffle_positives:
            self.rnd.shuffle(positive_ctxs)

        if self.is_train_set:
            self.rnd.shuffle(hard_negative_ctxs)
        positive_ctxs = positive_ctxs[0:1]
        hard_negative_ctxs = hard_negative_ctxs[0 : self.max_negatives]

        r.positive_passages = [
            BiEncoderPassage(self.linearize_func(self, ctx, True), ctx["caption"])
            for ctx in positive_ctxs
        ]
        r.negative_passages = []
        r.hard_negative_passages = [
            BiEncoderPassage(self.linearize_func(self, ctx, False), ctx["caption"])
            for ctx in hard_negative_ctxs
        ]
        return r

    def __len__(self):
        return len(self.data)

    @classmethod
    def get_lin_func(cls, split_type: str):
        f = {
            "type1": JsonLTablesQADataset._linearize_table,
            "type2": JsonLTablesQADataset._linearize_table2,
        }
        return f[split_type]

    @classmethod
    def split_table(cls, t: dict, max_length: int):
        rows = t["rows"]
        header = None
        header_len = 0
        start_row = 0

        # get the first non empty row as the "header"
        for i, r in enumerate(rows):
            row_lin, row_len = JsonLTablesQADataset._linearize_row(r)
            if len(row_lin) > 1:  # TODO: change to checking cell value tokens
                header = row_lin
                header_len += row_len
                start_row = i
                break

        chunks = []
        current_rows = [header]
        current_len = header_len

        for i in range(start_row + 1, len(rows)):
            row_lin, row_len = JsonLTablesQADataset._linearize_row(rows[i])
            if len(row_lin) > 1:  # TODO: change to checking cell value tokens
                current_rows.append(row_lin)
                current_len += row_len
            if current_len >= max_length:
                # linearize chunk
                linearized_str = "\n".join(current_rows) + "\n"
                chunks.append(linearized_str)
                current_rows = [header]
                current_len = header_len
                # logger.info('!!! chunk %s', linearized_str)

        if len(current_rows) > 1:
            linearized_str = "\n".join(current_rows) + "\n"
            chunks.append(linearized_str)
            # logger.info('!!! chunk %s', linearized_str)
        return chunks

    @classmethod
    def split_table2(cls, t: dict, max_length: int) -> List[str]:
        rows = t["rows"]
        header_id = 0

        # get the first non empty row as the "header"
        for i, r in enumerate(rows):
            row_lin, row_len = JsonLTablesQADataset._linearize_row(r)
            if len(row_lin) > 1:  # TODO: change to checking cell value tokens
                header_id = i
                break

        # logger.info("!!! table %s", t)

        chunks = []
        current_rows = []
        current_len = 0

        lin_f = JsonLTablesQADataset._linearize_row2
        for i in range(header_id + 1, len(rows)):
            row_lin, row_len = lin_f(rows[i], rows[header_id])
            if len(row_lin) > 1:  # TODO: change to checking cell value tokens
                current_rows.append(row_lin)
                current_len += row_len
            if current_len >= max_length:
                # linearize chunk
                linearized_str = "\n".join(current_rows) + "\n"
                chunks.append(linearized_str)
                current_rows = []
                current_len = 0
                # logger.info("!!! chunk %s", linearized_str)

        if len(current_rows) > 1:
            linearized_str = "\n".join(current_rows) + "\n"
            chunks.append(linearized_str)
            # logger.info("!!! chunk %s", linearized_str)

        if len(chunks) == 0:
            row = rows[header_id]
            result = ["row {} ".format(row["row"])]
            result += [c["value"] for c in row["columns"] if c["value"] != ""]
            row_lin = "; ".join(result) + "\n"
            chunks.append(row_lin)
        logger.info("!!! chunks %d", len(chunks))
        return chunks

    def _linearize_table(self, t: dict, is_positive: bool) -> List[str]:
        rows = t["rows"]
        selected_rows = set()
        rows_linearized = []
        total_words_len = 0

        # get the first non empty row as the "header"
        for i, r in enumerate(rows):
            row_lin, row_len = JsonLTablesQADataset._linearize_row(r)
            if len(row_lin) > 1:  # TODO: change to checking cell value tokens
                selected_rows.add(i)
                rows_linearized.append(row_lin)
                total_words_len += row_len
                break

        # split to chunks
        if is_positive:
            row_idx_with_answers = [ap[0] for ap in t["answer_pos"]]

            if self.shuffle_positives:
                self.rnd.shuffle(row_idx_with_answers)
            for i in row_idx_with_answers:
                if i not in selected_rows:
                    row_lin, row_len = JsonLTablesQADataset._linearize_row(rows[i])
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
                    row_lin, row_len = JsonLTablesQADataset._linearize_row(rows[i])
                    if len(row_lin) > 1:  # TODO: change to checking cell value tokens
                        selected_rows.add(i)
                        rows_linearized.append(row_lin)
                        total_words_len += row_len
                    if total_words_len >= self.max_len:
                        break

        linearized_str = ""
        for r in rows_linearized:
            linearized_str += r + "\n"

        # logger.info('!!! selected_rows %s', selected_rows)
        # logger.info('!!! total_words_len %s', total_words_len)
        # logger.info('!!! positive linearized_str %s', linearized_str)

        return linearized_str

    def _linearize_table2(self, t: dict, is_positive: bool) -> List[str]:
        rows = t["rows"]
        selected_rows = set()
        rows_linearized = []
        total_words_len = 0

        # logger.info("!!! table %s", t)

        # get the first non empty row as the "header"

        header_row_id = 0
        for i, r in enumerate(rows):
            row_lin, row_len = JsonLTablesQADataset._linearize_row(r)
            if len(row_lin) > 1:  # TODO: change to checking cell value tokens
                header_row_id = i
                selected_rows.add(i)
                break

        # split to chunks
        if is_positive:
            row_idx_with_answers = [ap[0] for ap in t["answer_pos"]]

            if self.shuffle_positives:
                self.rnd.shuffle(row_idx_with_answers)
            for i in row_idx_with_answers:
                if i not in selected_rows:
                    row_lin, row_len = JsonLTablesQADataset._linearize_row2(
                        rows[i], rows[header_row_id]
                    )
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
                    row_lin, row_len = JsonLTablesQADataset._linearize_row2(
                        rows[i], rows[header_row_id]
                    )
                    if len(row_lin) > 1:  # TODO: change to checking cell value tokens
                        selected_rows.add(i)
                        rows_linearized.append(row_lin)
                        total_words_len += row_len
                    if total_words_len >= self.max_len:
                        break

        linearized_str = ""
        for r in rows_linearized:
            linearized_str += r + "\n"

        # logger.info("!!! selected_rows %s", selected_rows)
        # logger.info("!!! total_words_len %s", total_words_len)
        # logger.info("!!! positive linearized_str %s", linearized_str)

        return linearized_str

    @classmethod
    def _linearize_row(cls, row: dict) -> Tuple[str, int]:
        cell_values = [c["value"] for c in row["columns"]]
        total_words = sum(len(c.split(" ")) for c in cell_values)
        return ", ".join([c["value"] for c in row["columns"]]), total_words

    @classmethod
    def _linearize_row2(cls, row: dict, header: dict) -> Tuple[str, int]:
        header_cells = [c["value"] for c in header["columns"]]
        cell_values = [c["value"] for c in row["columns"]]
        h = len(header_cells)
        n = len(cell_values)
        result = ["row {} ".format(row["row"])]
        for i in range(n):
            if cell_values[i] == "":
                continue
            if i < h:
                result += [" ".join([header_cells[i], "is", cell_values[i]])]
            else:
                result += [cell_values[i]]
        total_words = len(result)
        return "; ".join(result), total_words


class TRECDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        passages_filename: str,
        queries_filename: str,
        qrels_filename: str,
        selector: DictConfig,
        special_token: str = None,
    ):
        super().__init__(selector, special_token=special_token)
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
        logger.info("Reading TREC dataset")
        logger.info("Reading passages list from: %s", self.passages_file)
        self.passages_dict = _load_tsv_as_dict(self.passages_file)
        logger.info("Total TREC passages: %d", len(self.passages_dict))

        logger.info("Reading queries from: %s", self.queries_file)
        self.queries = _load_tsv_as_dict(self.queries_file)
        logger.info("Total TREC queries: %d", len(self.queries))

        logger.info("Reading triplets data from: %s", self.qidpidtriples_file)
        self.qrels = _load_qrels(self.qidpidtriples_file)
        logger.info("Total TREC samples: %d", len(self.qrels))

    def __getitem__(self, index) -> BiEncoderSample:
        qrel = self.qrels[index]
        r = BiEncoderSample()
        r.query = self._process_query(self.queries[qrel[0]])
        r.positive_passages = [
            BiEncoderPassage(self.passages_dict[pid], None) for pid in qrel[1]
        ]
        r.hard_negative_passages = [
            BiEncoderPassage(self.passages_dict[pid], None) for pid in qrel[2]
        ]
        r.negative_passages = []
        return r

    def __len__(self):
        return len(self.qrels)


def _load_tsv_as_dict(ctx_file: str) -> Dict[int, str]:
    docs = {}
    with open(ctx_file) as tsvfile:
        reader = csv.reader(tsvfile, delimiter="\t")
        # file format: id, text
        for row in reader:
            if row[0] != "id":
                docs[int(row[0])] = row[1]
    return docs


def _load_qrels(
    qrels_file: str, max_passages_per_q: int = 30
) -> List[Tuple[int, List[int], List[int]]]:
    """
    Loads TREC's (Triples QID PID) Format
    :param qrels_file:
    :param max_passages_per_q:
    :return: result tuple format: qid, positive pid-s, negative pid-s
    """

    result = {}
    with open(qrels_file) as tsvfile:
        reader = csv.reader(tsvfile, delimiter="\t")
        # file format: qid, positive pid, negative pid
        for row in reader:
            if row[0] != "id":
                qid = int(row[0])
                q_info = result.get(qid, ([], []))
                if len(q_info[0]) < max_passages_per_q:
                    pid = int(row[1])
                    if (
                        pid not in q_info[0]
                    ):  # assuming positive pid is usually ~ 1 sample
                        q_info[0].append(pid)
                if len(q_info[1]) < max_passages_per_q:
                    pid = int(row[2])
                    q_info[1].append(pid)  # assuming negatives are unique per question
                result[qid] = q_info
    # convert to list
    return [(k, v[0], v[1]) for k, v in result.items()]


def split_tables_to_chunks(
    tables_dict: Dict[str, Table], max_table_len: int, split_type: str = "type1"
) -> List[Tuple[int, str, str, int]]:
    tables_as_dicts = [t.to_dpr_json() for k, t in tables_dict.items()]
    chunks = []
    chunk_id = 0
    for i, t in enumerate(tables_as_dicts):
        if split_type == "type2":
            table_chunks = JsonLTablesQADataset.split_table2(t, max_table_len)
        else:
            table_chunks = JsonLTablesQADataset.split_table(t, max_table_len)
        title = t["caption"]
        for c in table_chunks:
            # chunk id , text, title, external_id
            chunks.append((chunk_id, c, title, i))
            chunk_id += 1
        if i % 1000 == 0:
            logger.info("Splitted %d tables to %d chunks", i, len(chunks))
    return chunks


def normalize_kilt_passage(ctx_text: str):
    ctx_text = ctx_text.replace("\n", " ")
    return ctx_text
