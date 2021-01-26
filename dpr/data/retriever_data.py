import collections
import csv
import hydra
import json
import jsonlines
import logging
import pickle
import torch

from typing import Dict, List
from omegaconf import DictConfig

from dpr.data.biencoder_data import (
    BiEncoderPassage,
    split_tables_to_chunks,
    normalize_passage,
    normalize_question,
)
from dpr.data.tables import read_nq_tables_jsonl

logger = logging.getLogger(__name__)
QASample = collections.namedtuple("QuerySample", ["query", "id", "answers"])
TableChunk = collections.namedtuple("TableChunk", ["text", "title", "table_id"])


class QASrc(torch.utils.data.Dataset):
    def __init__(
        self,
        selector: DictConfig = None,
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        self.data = None
        self.selector = hydra.utils.instantiate(selector) if selector else None
        self.special_query_token = special_query_token
        self.query_special_suffix = query_special_suffix

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def _process_question(self, question: str):
        # as of now, always normalize query
        question = normalize_question(question)
        if self.query_special_suffix and not question.endswith(
            self.query_special_suffix
        ):
            question += self.query_special_suffix
        return question


class CsvQASrc(QASrc):
    def __init__(
        self,
        file: str,
        question_col: int = 0,
        answers_col: int = 1,
        id_col: int = -1,
        selector: DictConfig = None,
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(selector, special_query_token, query_special_suffix)
        self.question_col = question_col
        self.answers_col = answers_col
        self.id_col = id_col
        self.file = file

    def load_data(self):
        data = []
        with open(self.file) as ifile:
            reader = csv.reader(ifile, delimiter="\t")
            for row in reader:
                question = row[self.question_col]
                answers = eval(row[self.answers_col])
                id = None
                if self.id_col >= 0:
                    id = row[self.id_col]
                data.append(QASample(self._process_question(question), id, answers))
        self.data = data


class JsonlQASrc(QASrc):
    def __init__(
        self,
        file: str,
        selector: DictConfig = None,
        question_attr: str = "question",
        answers_attr: str = "answers",
        id_attr: str = "id",
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(selector, special_query_token, query_special_suffix)
        self.question_attr = question_attr
        self.answers_attr = answers_attr
        self.id_attr = id_attr
        self.file = file

    def load_data(self):
        data = []
        with jsonlines.open(self.file, mode="r") as jsonl_reader:
            for jline in jsonl_reader:
                question = jline[self.question_attr]
                answers = jline[self.answers_attr] if self.answers_attr in jline else []
                id = None
                if self.id_attr in jline:
                    id = jline[self.id_attr]
                data.append(QASample(self._process_question(question), id, answers))
        self.data = data


class KiltCsvQASrc(CsvQASrc):
    def __init__(
        self,
        file: str,
        kilt_gold_file: str,
        question_col: int = 0,
        answers_col: int = 1,
        id_col: int = -1,
        selector: DictConfig = None,
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(
            file,
            question_col,
            answers_col,
            id_col,
            selector,
            special_query_token,
            query_special_suffix,
        )
        self.kilt_gold_file = kilt_gold_file


class KiltJsonlQASrc(JsonlQASrc):
    def __init__(
        self,
        file: str,
        kilt_gold_file: str,
        question_attr: str = "input",
        answers_attr: str = "answer",
        id_attr: str = "id",
        selector: DictConfig = None,
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(
            file,
            selector,
            question_attr,
            answers_attr,
            id_attr,
            special_query_token,
            query_special_suffix,
        )
        self.kilt_gold_file = kilt_gold_file

    def load_data(self):
        data = []
        with jsonlines.open(self.file, mode="r") as jsonl_reader:
            for jline in jsonl_reader:
                question = jline[self.question_attr]
                out = jline["output"]
                answers = [o["answer"] for o in out if "answer" in o]
                id = None
                if self.id_attr in jline:
                    id = jline[self.id_attr]
                data.append(QASample(self._process_question(question), id, answers))
        self.data = data


# TODO: super class for CtxSrc ?
class CsvCtxSrc(object):
    def __init__(
        self,
        file: str,
        id_col: int = 0,
        text_col: int = 1,
        title_col: int = 2,
        id_prefix: str = None,
        normalize: bool = False,
    ):
        self.text_col = text_col
        self.title_col = title_col
        self.id_col = id_col
        self.file = file
        self.id_prefix = id_prefix
        self.normalize = normalize

    def load_data_to(self, ctxs: Dict[object, BiEncoderPassage]):
        with open(self.file) as ifile:
            reader = csv.reader(ifile, delimiter="\t")
            for row in reader:
                if row[self.id_col] != "id":
                    if self.id_prefix:
                        sample_id = self.id_prefix + str(row[self.id_col])
                    else:
                        sample_id = row[self.id_col]
                    passage = row[self.text_col]
                    if self.normalize:
                        passage = normalize_passage(passage)
                    ctxs[sample_id] = BiEncoderPassage(passage, row[self.title_col])


class KiltCsvCtxSrc(CsvCtxSrc):
    def __init__(
        self,
        file: str,
        mapping_file: str,
        id_col: int = 0,
        text_col: int = 1,
        title_col: int = 2,
        id_prefix: str = None,
        normalize: bool = False,
    ):
        super().__init__(
            file, id_col, text_col, title_col, id_prefix, normalize=normalize
        )
        self.mapping_file = mapping_file

    def convert_to_kilt(self, kilt_gold_file, dpr_output, kilt_out_file):
        logger.info("Converting to KILT format file: %s", dpr_output)

        with open(dpr_output, "rt") as fin:
            dpr_output = json.load(fin)

        with jsonlines.open(kilt_gold_file, "r") as reader:
            kilt_gold_file = list(reader)
        assert len(kilt_gold_file) == len(dpr_output)
        map_path = self.mapping_file
        with open(map_path, "rb") as fin:
            mapping = pickle.load(fin)

        with jsonlines.open(kilt_out_file, mode="w") as writer:
            for dpr_entry, kilt_gold_entry in zip(dpr_output, kilt_gold_file):
                assert dpr_entry["question"] == kilt_gold_entry["input"]
                provenance = []
                for ctx in dpr_entry["ctxs"]:
                    wikipedia_id, end_paragraph_id = mapping[int(ctx["id"])]
                    provenance.append(
                        {
                            "wikipedia_id": wikipedia_id,
                            "end_paragraph_id": end_paragraph_id,
                        }
                    )
                kilt_entry = {
                    "id": kilt_gold_entry["id"],
                    "input": dpr_entry["question"],
                    "output": [{"provenance": provenance}],
                }
                writer.write(kilt_entry)

        logger.info("Saved KILT formatted results to: %s", kilt_out_file)


class JsonlTablesCtxSrc(object):
    def __init__(
        self,
        file: str,
        tables_chunk_sz: int = 100,
        split_type: str = "type1",
        id_prefix: str = None,
    ):
        self.tables_chunk_sz = tables_chunk_sz
        self.split_type = split_type
        self.file = file
        self.id_prefix = id_prefix

    def load_data_to(self, ctxs: Dict):
        docs = {}
        logger.info("Parsing Tables data from: %s", self.file)
        tables_dict = read_nq_tables_jsonl(self.file)
        table_chunks = split_tables_to_chunks(
            tables_dict, self.tables_chunk_sz, split_type=self.split_type
        )
        for chunk in table_chunks:
            sample_id = self.id_prefix + str(chunk[0])
            docs[sample_id] = TableChunk(chunk[1], chunk[2], chunk[3])
        logger.info("Loaded %d tables chunks", len(docs))
        ctxs.update(docs)


class ToyCtxSrc(object):
    def __init__(
        self,
    ):
        self.id_prefix = ""
        self.normalize = False
        self.data = [
            (
                "This makes shortwave radio one of the most robust means of communications, which can be disrupted only by interference or bad ionospheric conditions. Modern digital transmission modes such as MFSK and Olivia are even more robust, allowing successful reception of signals well below the noise floor of a conventional receiver. Shortwave radio's benefits are sometimes regarded as being outweighed by its drawbacks, including: BULLET::::- In most Western countries, shortwave radio ownership is usually limited to true enthusiasts, since most new standard radios do not receive the shortwave band. Therefore, Western audiences are limited. BULLET::::- In the developed world, shortwave reception is very difficult in urban areas because of excessive noise from switched-mode power adapters, fluorescent or LED light sources, internet modems and routers",
                "Shortwave radio",
            ),
            (
                "Shortwave broadcast radio uses digital transmission modes such as MFSK and Olivia, allowing successful reception of signals well below the noise floor of a conventional receiver. This makes shortwave radio one of the most robust means of communications, which can be disrupted only by interference or bad ionospheric conditions. Shortwave radio's benefits are sometimes regarded as being outweighed by its drawbacks, including: BULLET::::- In most Western countries, shortwave radio ownership is usually limited to true enthusiasts, since most new standard radios do not receive the shortwave band. Therefore, Western audiences are limited. BULLET::::- In the developed world, shortwave reception is very difficult in urban areas because of excessive noise from switched-mode power adapters, fluorescent or LED light sources, internet modems and routers",
                "Shortwave radio",
            ),
            (
                "Shortwave radio can benefit from modern digital transmission modes such as MFSK and Olivia which are robust, allowing successful reception of signals well below the noise floor of a conventional receiver. Shortwave radio's benefits are sometimes regarded as being outweighed by its drawbacks, including: BULLET::::- In most Western countries, shortwave radio ownership is usually limited to true enthusiasts, since most new standard radios do not receive the shortwave band. Therefore, Western audiences are limited. BULLET::::- In the developed world, shortwave reception is very difficult in urban areas because of excessive noise from switched-mode power adapters, fluorescent or LED light sources, internet modems and routers",
                "Shortwave radio",
            ),
            (
                "Shortwave radio is one of the most robust means of communications since it can use modern digital transmission modes such as MFSK and Olivia, allowing successful reception of signals well below the noise floor of a conventional receiver. Shortwave radio's benefits are sometimes regarded as being outweighed by its drawbacks, including: BULLET::::- In most Western countries, shortwave radio ownership is usually limited to true enthusiasts, since most new standard radios do not receive the shortwave band. Therefore, Western audiences are limited. BULLET::::- In the developed world, shortwave reception is very difficult in urban areas because of excessive noise from switched-mode ",
                "Shortwave radio",
            ),
            (
                "Heavy showers coming from pre-monsoonal convective clouds mainly in the form of squall lines also known as the north easterlies formed mainly as a result of the interactions of the two dominant airmasses in Nigeria known as the Maritime tropical (south westerlies) and the Continental tropical (north easterlies). It begins in central Nigeria while the Monsoons from the south atlantic ocean arrives in central Nigeria in July bringing with it high humidity, heavy cloud cover and heavy rainfall which can be daily occurrence lasting till September when the monsoons gradually begin retreating southward to the southern part of Nigeria.",
                "Geography of Nigeria",
            ),
            (
                "Heavy showers coming from pre-monsoonal convective clouds mainly in the form of squall lines also known as the north easterlies formed mainly as a result of the interactions of the two dominant airmasses in Nigeria known as the Maritime tropical (south westerlies) and the Continental tropical (north easterlies). It begins in central Nigeria when south westerlies arrives in central Nigeria in July bringing with it high humidity, heavy cloud cover and heavy rainfall which can be daily occurrence lasting till September when the monsoons gradually begin retreating southward to the southern part of Nigeria. Rainfall totals in central Nigeria",
                "Geography of Nigeria",
            ),
            (
                "Heavy showers coming from pre-monsoonal convective clouds mainly in the form of squall lines also known as the north easterlies formed mainly as a result of the interactions of the two dominant airmasses in Nigeria known as the Maritime tropical (south westerlies) and the Continental tropical (north easterlies). South west winds  arrives in central Nigeria in July bringing with it high humidity, heavy cloud cover and heavy rainfall which can be daily occurrence lasting till September when the monsoons gradually begin retreating southward to the southern part of Nigeria. Rainfall totals in central Nigeria varies from 1,100 mm (43.3 in) in the lowlands of the river Niger Benue trough",
                "Geography of Nigeria",
            ),
            (
                "Heavy showers coming from pre-monsoonal convective clouds mainly in the form of squall lines also known as the north easterlies formed mainly as a result of the interactions of the two dominant airmasses in Nigeria known as the Maritime tropical (south westerlies) and the Continental tropical (north easterlies). South west airmasses  arrives in central Nigeria in July bringing with it high humidity, heavy cloud cover and heavy rainfall which can be daily occurrence lasting till September when the monsoons gradually begin retreating southward to the southern part of Nigeria. Rainfall totals in central Nigeria varies from 1,100 mm (43.3 in) in the lowlands of the river Niger Benue trough",
                "Geography of Nigeria",
            ),
            (
                "South west airmasses  arrives in central Nigeria in July bringing with it high humidity, heavy cloud cover and heavy rainfall which can be daily occurrence lasting till September when the monsoons gradually begin retreating southward to the southern part of Nigeria. Heavy showers coming from pre-monsoonal convective clouds mainly in the form of squall lines also known as the north easterlies formed mainly as a result of the interactions of the two dominant airmasses in Nigeria known as the Maritime tropical (south westerlies) and the Continental tropical (north easterlies). Rainfall totals in central Nigeria varies from 1,100 mm (43.3 in) in the lowlands of the river Niger Benue trough",
                "Geography of Nigeria",
            ),
            (
                "Thus, the right to property is no longer a fundamental right, though it is still a constitutional right. If the government appears to have acted unfairly, the action can be challenged in a court of law by aggrieved citizens. The liberalisation of the economy and the government's initiative to set up special economic zones has led to many protests by farmers and have led to calls for the reinstatement of the fundamental right to private property. Supreme Court had sent a notice to the government questioning why the right should not be brought back, but in 2010, the Court rejected the PIL.",
                "Fundamental rights in India",
            ),
            (
                "Furthermore, the aggrieved person would also have no right to move the court under Article 32 due to the right to property no longer being a fundamental right, though it would still be a constitutional one. If the government appears to have acted unfairly, the action can be challenged in a court of law by aggrieved citizens. The liberalisation of the economy and the government's initiative to set up special economic zones has led to many protests by farmers and have led to calls for the reinstatement of the fundamental right to private property. The Supreme Court has sent a notice to the government questioning why the right",
                "Fundamental rights in India",
            ),
            (
                "Since the right to property is no longer being a fundamental right, the aggrieved person would also have no right to move the court . But the right to property is still a constitutional right. The liberalisation of the economy and the government's initiative to set up special economic zones has led to many protests by farmers and have led to calls for the reinstatement of the fundamental right to private property. Supreme Court had sent a notice to the government questioning why the right should not be brought back, but in 2010, the Court rejected the PIL.",
                "Fundamental rights in India",
            ),
            (
                'The right to property according to the constitution of india is a is no longer being a fundamental right but is still a constitutional right.The provisions relating to the right to property were changed a number of times. The 44th Amendment of 1978 removed the right to property from the list of fundamental rights.[50] A new provision, Article 300-A, was added to the constitution, which provided that "no person shall be deprived of his property save by authority of law". Thus, if a legislator made a law depriving a person of his property, there would be no obligation on the part of the State to pay anything as compensation.',
                "Fundamental rights in India",
            ),
        ]

    def load_data_to(self, ctxs: Dict[object, BiEncoderPassage]):
        for i, s in enumerate(self.data):
            tokens = s[0].split()
            tokens = tokens[0:100]
            passage = " ".join(tokens)
            logger.info("passage %s", passage)
            logger.info("title %s", s[1])
            ctxs[i] = BiEncoderPassage(passage, s[1])
