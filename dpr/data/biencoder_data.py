import collections
import glob
import logging
import os
import random
from typing import Dict, List, Tuple

import jsonlines
import numpy as np
from omegaconf import DictConfig

from dpr.data.tables import Table
from dpr.utils.data_utils import read_data_from_json_files, Dataset

logger = logging.getLogger(__name__)
BiEncoderPassage = collections.namedtuple("BiEncoderPassage", ["text", "title"])


def get_dpr_files(source_name) -> List[str]:
    if os.path.exists(source_name) or glob.glob(source_name):
        return glob.glob(source_name)
    else:
        # try to use data downloader
        from dpr.data.download_data import download
        return download(source_name)


class BiEncoderSample(object):
    query: str
    positive_passages: List[BiEncoderPassage]
    negative_passages: List[BiEncoderPassage]
    hard_negative_passages: List[BiEncoderPassage]


class JsonQADataset(Dataset):
    def __init__(
        self,
        file: str,
        selector: DictConfig = None,
        special_token: str = None,
        encoder_type: str = None,
        shuffle_positives: bool = False,
        normalize: bool = False,
        query_special_suffix: str = None,
        # tmp: for cc-net results only
        exclude_gold: bool = False,
    ):
        super().__init__(
            selector,
            special_token=special_token,
            encoder_type=encoder_type,
            shuffle_positives=shuffle_positives,
            query_special_suffix=query_special_suffix,
        )
        self.file = file
        self.data_files = []
        self.normalize = normalize
        self.exclude_gold = exclude_gold

    def calc_total_data_len(self):
        if not self.data:
            logger.info("Loading all data")
            self._load_all_data()
        return len(self.data)

    def load_data(self, start_pos: int = -1, end_pos: int = -1):
        if not self.data:
            self._load_all_data()
        if start_pos >= 0 and end_pos >= 0:
            logger.info("Selecting subset range from %d to %d", start_pos, end_pos)
            self.data = self.data[start_pos:end_pos]

    def _load_all_data(self):
        self.data_files = get_dpr_files(self.file)
        logger.info("Data files: %s", self.data_files)
        data = read_data_from_json_files(self.data_files)
        # filter those without positive ctx
        self.data = [r for r in data if len(r["positive_ctxs"]) > 0]
        logger.info("Total cleaned data size: %d", len(self.data))

    def __getitem__(self, index) -> BiEncoderSample:
        json_sample = self.data[index]
        r = BiEncoderSample()
        r.query = self._process_query(json_sample["question"])

        positive_ctxs = json_sample["positive_ctxs"]
        if self.exclude_gold:
            ctxs = [ctx for ctx in positive_ctxs if "score" in ctx]
            if ctxs:
                positive_ctxs = ctxs

        negative_ctxs = json_sample["negative_ctxs"] if "negative_ctxs" in json_sample else []
        hard_negative_ctxs = json_sample["hard_negative_ctxs"] if "hard_negative_ctxs" in json_sample else []

        for ctx in positive_ctxs + negative_ctxs + hard_negative_ctxs:
            if "title" not in ctx:
                ctx["title"] = None

        def create_passage(ctx: dict):
            return BiEncoderPassage(
                normalize_passage(ctx["text"]) if self.normalize else ctx["text"],
                ctx["title"],
            )

        r.positive_passages = [create_passage(ctx) for ctx in positive_ctxs]
        r.negative_passages = [create_passage(ctx) for ctx in negative_ctxs]
        r.hard_negative_passages = [create_passage(ctx) for ctx in hard_negative_ctxs]
        return r


class JsonlQADataset(JsonQADataset):
    def __init__(
        self,
        file: str,
        selector: DictConfig = None,
        special_token: str = None,
        encoder_type: str = None,
        shuffle_positives: bool = False,
        normalize: bool = False,
        query_special_suffix: str = None,
        # tmp: for cc-net results only
        exclude_gold: bool = False,
        total_data_size: int = -1,
    ):
        super().__init__(
            file,
            selector,
            special_token,
            encoder_type,
            shuffle_positives,
            normalize,
            query_special_suffix,
            exclude_gold,
        )
        self.total_data_size = total_data_size
        self.data_files = get_dpr_files(self.file)
        logger.info("Data files: %s", self.data_files)

    def calc_total_data_len(self):
        # TODO: optimize jsonl file read & results caching
        if self.total_data_size < 0:
            logger.info("Calculating data size")
            for file in self.data_files:
                with jsonlines.open(file, mode="r") as jsonl_reader:
                    for _ in jsonl_reader:
                        self.total_data_size += 1
        logger.info("total_data_size=%d", self.total_data_size)
        return self.total_data_size

    def load_data(self, start_pos: int = -1, end_pos: int = -1):
        if self.data:
            return
        logger.info("Jsonl loading subset range from %d to %d", start_pos, end_pos)
        if start_pos < 0 and end_pos < 0:
            for file in self.data_files:
                with jsonlines.open(file, mode="r") as jsonl_reader:
                    self.data.extend([l for l in jsonl_reader])
            return

        global_sample_id = 0
        for file in self.data_files:
            if global_sample_id >= end_pos:
                break
            with jsonlines.open(file, mode="r") as jsonl_reader:
                for jline in jsonl_reader:
                    if start_pos <= global_sample_id < end_pos:
                        self.data.append(jline)
                    if global_sample_id >= end_pos:
                        break
                    global_sample_id += 1
        logger.info("Jsonl loaded data size %d ", len(self.data))


def normalize_passage(ctx_text: str):
    ctx_text = ctx_text.replace("\n", " ").replace("’", "'")
    if ctx_text.startswith('"'):
        ctx_text = ctx_text[1:]
    if ctx_text.endswith('"'):
        ctx_text = ctx_text[:-1]
    return ctx_text


class Cell:
    def __init__(self):
        self.value_tokens: List[str] = []
        self.type: str = ""
        self.nested_tables: List[Table] = []

    def __str__(self):
        return " ".join(self.value_tokens)

    def to_dpr_json(self, cell_idx: int):
        r = {"col": cell_idx}
        r["value"] = str(self)
        return r


class Row:
    def __init__(self):
        self.cells: List[Cell] = []

    def __str__(self):
        return "| ".join([str(c) for c in self.cells])

    def visit(self, tokens_function, row_idx: int):
        for i, c in enumerate(self.cells):
            if c.value_tokens:
                tokens_function(c.value_tokens, row_idx, i)

    def to_dpr_json(self, row_idx: int):
        r = {"row": row_idx}
        r["columns"] = [c.to_dpr_json(i) for i, c in enumerate(self.cells)]
        return r


class Table(object):
    def __init__(self, caption=""):
        self.caption = caption
        self.body: List[Row] = []
        self.key = None
        self.gold_match = False

    def __str__(self):
        table_str = "<T>: {}\n".format(self.caption)
        table_str += " rows:\n"
        for i, r in enumerate(self.body):
            table_str += " row #{}: {}\n".format(i, str(r))

        return table_str

    def get_key(self) -> str:
        if not self.key:
            self.key = str(self)
        return self.key

    def visit(self, tokens_function, include_caption: bool = False) -> bool:
        if include_caption:
            tokens_function(self.caption, -1, -1)
        for i, r in enumerate(self.body):
            r.visit(tokens_function, i)

    def to_dpr_json(self):
        r = {
            "caption": self.caption,
            "rows": [r.to_dpr_json(i) for i, r in enumerate(self.body)],
        }
        if self.gold_match:
            r["gold_match"] = 1
        return r


class NQTableParser(object):
    def __init__(self, tokens, is_html_mask, title):
        self.tokens = tokens
        self.is_html_mask = is_html_mask
        self.max_idx = len(self.tokens)
        self.all_tables = []

        self.current_table: Table = None
        self.tables_stack = collections.deque()
        self.title = title

    def parse(self) -> List[Table]:
        self.all_tables = []
        self.tables_stack = collections.deque()

        for i in range(self.max_idx):

            t = self.tokens[i]

            if not self.is_html_mask[i]:
                # cell content
                self._on_content(t)
                continue

            if "<Table" in t:
                self._on_table_start()
            elif t == "</Table>":
                self._on_table_end()
            elif "<Tr" in t:
                self._onRowStart()
            elif t == "</Tr>":
                self._onRowEnd()
            elif "<Td" in t or "<Th" in t:
                self._onCellStart()
            elif t in ["</Td>", "</Th>"]:
                self._on_cell_end()

        return self.all_tables

    def _on_table_start(self):
        caption = self.title
        parent_table = self.current_table
        if parent_table:
            self.tables_stack.append(parent_table)

            caption = parent_table.caption
            if parent_table.body and parent_table.body[-1].cells:
                current_cell = self.current_table.body[-1].cells[-1]
                caption += " | " + " ".join(current_cell.value_tokens)

        t = Table()
        t.caption = caption
        self.current_table = t
        self.all_tables.append(t)

    def _on_table_end(self):
        t = self.current_table
        if t:
            if self.tables_stack:  # t is a nested table
                self.current_table = self.tables_stack.pop()
                if self.current_table.body:
                    current_cell = self.current_table.body[-1].cells[-1]
                    current_cell.nested_tables.append(t)
        else:
            logger.error("table end without table object")

    def _onRowStart(self):
        self.current_table.body.append(Row())

    def _onRowEnd(self):
        pass

    def _onCellStart(self):
        current_row = self.current_table.body[-1]
        current_row.cells.append(Cell())

    def _on_cell_end(self):
        pass

    def _on_content(self, token):
        if self.current_table.body:
            current_row = self.current_table.body[-1]
            current_cell = current_row.cells[-1]
            current_cell.value_tokens.append(token)
        else:  # tokens outside of row/cells. Just append to the table caption.
            self.current_table.caption += " " + token


def read_nq_tables_jsonl(path: str) -> Dict[str, Table]:
    tables_with_issues = 0
    single_row_tables = 0
    nested_tables = 0
    regular_tables = 0
    total_tables = 0
    total_rows = 0
    tables_dict = {}

    with jsonlines.open(path, mode="r") as jsonl_reader:
        for jline in jsonl_reader:
            tokens = jline["tokens"]

            if "( hide ) This section has multiple issues" in " ".join(tokens):
                tables_with_issues += 1
                continue

            mask = jline["html_mask"]
            # page_url = jline["doc_url"]
            title = jline["title"]
            p = NQTableParser(tokens, mask, title)
            tables = p.parse()

            # table = parse_table(tokens, mask)

            nested_tables += len(tables[1:])

            for t in tables:
                total_tables += 1

                # calc amount of non empty rows
                non_empty_rows = sum([1 for r in t.body if r.cells and any([True for c in r.cells if c.value_tokens])])

                if non_empty_rows <= 1:
                    single_row_tables += 1
                else:
                    regular_tables += 1
                    total_rows += len(t.body)

                    if t.get_key() not in tables_dict:
                        tables_dict[t.get_key()] = t

            if len(tables_dict) % 1000 == 0:
                logger.info("tables_dict %d", len(tables_dict))

    logger.info("regular tables %d", regular_tables)
    logger.info("tables_with_issues %d", tables_with_issues)
    logger.info("single_row_tables %d", single_row_tables)
    logger.info("nested_tables %d", nested_tables)
    return tables_dict


def get_table_string_for_answer_check(table: Table):  # this doesn't use caption
    table_text = ""
    for r in table.body:
        table_text += " . ".join([" ".join(c.value_tokens) for c in r.cells])
    table_text += " . "
    return table_text


# TODO: inherit from Jsonl
class JsonLTablesQADataset(Dataset):
    def __init__(
        self,
        file: str,
        is_train_set: bool,
        selector: DictConfig = None,
        shuffle_positives: bool = False,
        max_negatives: int = 1,
        seed: int = 0,
        max_len=100,
        split_type: str = "type1",
    ):
        super().__init__(selector, shuffle_positives=shuffle_positives)
        self.data_files = glob.glob(file)
        self.data = []
        self.is_train_set = is_train_set
        self.max_negatives = max_negatives
        self.rnd = random.Random(seed)
        self.max_len = max_len
        self.linearize_func = JsonLTablesQADataset.get_lin_func(split_type)

    def load_data(self, start_pos: int = -1, end_pos: int = -1):
        # TODO: use JsonlX super class load_data() ?
        data = []
        for path in self.data_files:
            with jsonlines.open(path, mode="r") as jsonl_reader:
                data += [jline for jline in jsonl_reader]
        # filter those without positive ctx
        self.data = [r for r in data if len(r["positive_ctxs"]) > 0]
        logger.info("Total cleaned data size: {}".format(len(self.data)))
        if start_pos >= 0 and end_pos >= 0:
            logger.info("Selecting subset range from %d to %d", start_pos, end_pos)
            self.data = self.data[start_pos:end_pos]

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
            BiEncoderPassage(self.linearize_func(self, ctx, True), ctx["caption"]) for ctx in positive_ctxs
        ]
        r.negative_passages = []
        r.hard_negative_passages = [
            BiEncoderPassage(self.linearize_func(self, ctx, False), ctx["caption"]) for ctx in hard_negative_ctxs
        ]
        return r

    @classmethod
    def get_lin_func(cls, split_type: str):
        f = {
            "type1": JsonLTablesQADataset._linearize_table,
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

        if len(current_rows) > 1:
            linearized_str = "\n".join(current_rows) + "\n"
            chunks.append(linearized_str)
        return chunks

    def _linearize_table(self, t: dict, is_positive: bool) -> str:
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

        return linearized_str

    @classmethod
    def _linearize_row(cls, row: dict) -> Tuple[str, int]:
        cell_values = [c["value"] for c in row["columns"]]
        total_words = sum(len(c.split(" ")) for c in cell_values)
        return ", ".join([c["value"] for c in row["columns"]]), total_words


def split_tables_to_chunks(
    tables_dict: Dict[str, Table], max_table_len: int, split_type: str = "type1"
) -> List[Tuple[int, str, str, int]]:
    tables_as_dicts = [t.to_dpr_json() for k, t in tables_dict.items()]
    chunks = []
    chunk_id = 0
    for i, t in enumerate(tables_as_dicts):
        # TODO: support other types
        assert split_type == "type1"
        table_chunks = JsonLTablesQADataset.split_table(t, max_table_len)
        title = t["caption"]
        for c in table_chunks:
            # chunk id , text, title, external_id
            chunks.append((chunk_id, c, title, i))
            chunk_id += 1
        if i % 1000 == 0:
            logger.info("Splitted %d tables to %d chunks", i, len(chunks))
    return chunks
