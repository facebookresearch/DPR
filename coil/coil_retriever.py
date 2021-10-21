import os
from collections import defaultdict, namedtuple
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from coil.models.coil_model import COIL
from coil.utils.dataset import EncodeDataset
from utils.data_utils import Tensorizer
from retrievers.abstract_dense_retriever import DenseRetriever

try:
    from coil.retriever_ext import scatter as c_scatter
except ImportError:
    raise ImportError(
        "Cannot import scatter module."
        " Make sure you have compiled the retriever extension."
    )

COILQuestionRepresentations = namedtuple(
    "COILQuestionRepresentations",
    ["tok_reps", "offsets", "cls_reps"],
)

COILIndexShard = namedtuple(
    "COILIndexShard",
    [
        "all_ivl_scatter_maps",
        "all_shard_scatter_maps",
        "doc_cls_reps",
        "cls_ex_ids",
        "tok_id_2_reps",
    ],
)


class COILIndex:
    """
    Hosts the document index and enable search from shards
    """

    def __init__(self, shard_path, num_shard: int, buffer_size: int = 1000):
        """
        Converts shard path into list of paths to each shard
        """
        self.shard_paths = []
        for i in range(int(num_shard)):
            self.shard_paths.append(os.path.join(shard_path, f"shard_{i:02d}"))

        self.shard_indexes = []
        self.buffer_size = buffer_size

    def init_index(self, vector_size):
        self.vector_size = vector_size

    @staticmethod
    def dict_2_float(dd):
        for k in dd:
            dd[k] = dd[k].float()

        return dd

    def load_shards_into_index(self, vector_files):
        if vector_files:
            self.shard_paths.extend(vector_files)

        for doc_shard in self.shard_paths:
            all_ivl_scatter_maps = torch.load(
                os.path.join(doc_shard, "ivl_scatter_maps.pt")
            )
            all_shard_scatter_maps = torch.load(
                os.path.join(doc_shard, "shard_scatter_maps.pt")
            )
            tok_id_2_reps = torch.load(os.path.join(doc_shard, "tok_reps.pt"))
            doc_cls_reps = torch.load(os.path.join(doc_shard, "cls_reps.pt")).float()
            cls_ex_ids = torch.load(os.path.join(doc_shard, "cls_ex_ids.pt"))
            tok_id_2_reps = COILIndex.dict_2_float(tok_id_2_reps)

            self.shard_indexes.append(
                COILIndexShard(
                    all_ivl_scatter_maps=all_ivl_scatter_maps,
                    all_shard_scatter_maps=all_shard_scatter_maps,
                    doc_cls_reps=doc_cls_reps,
                    cls_ex_ids=cls_ex_ids,
                    tok_id_2_reps=tok_id_2_reps,
                )
            )

    def search_top_k(
        self, question_vector: COILQuestionRepresentations, top_docs: int = 1000
    ):
        all_results = []
        for index in self.shard_indexes:
            all_results.extend(
                self.search_top_k_from_shard(question_vector, index, top_docs)
            )

        sorted_results = sorted(all_results, key=lambda x: x[1], reverse=True)
        return sorted_results[:top_docs]

    def search_top_k_from_shard(
        self,
        question_vector: COILQuestionRepresentations,
        index_shard: COILIndexShard,
        top_docs: int = 100,
    ) -> List[Tuple[List[object], List[float]]]:
        """
        Searches query in one particular index shard
        Args:
            question_vector: COILQuestionRepresentations
            index_shard: COILIndexShard
            top_docs: top k document from the shard

        Returns:
            [(doc_id, score) * num_queries]
        """
        query_cls_reps = question_vector.cls_reps
        all_query_offsets = question_vector.offsets
        query_tok_reps = question_vector.tok_reps

        doc_cls_reps = index_shard.doc_cls_reps
        tok_id_2_reps = index_shard.tok_id_2_reps
        all_ivl_scatter_maps = index_shard.all_ivl_scatter_maps
        all_shard_scatter_maps = index_shard.all_shard_scatter_maps

        match_scores = torch.matmul(
            query_cls_reps.float(), doc_cls_reps.transpose(0, 1)
        )  # D * b

        batched_qtok_offsets = defaultdict(list)
        q_batch_offsets = defaultdict(list)

        for batch_offset, q_offsets in enumerate(all_query_offsets):
            for q_tok_id, q_tok_offset in q_offsets:
                if q_tok_id not in tok_id_2_reps:
                    continue
                batched_qtok_offsets[q_tok_id].append(q_tok_offset)
                q_batch_offsets[q_tok_id].append(batch_offset)

        batch_qtok_ids = list(batched_qtok_offsets.keys())
        batched_tok_scores = []

        for q_tok_id in batch_qtok_ids:
            q_tok_reps = query_tok_reps[batched_qtok_offsets[q_tok_id]]
            tok_reps = tok_id_2_reps[q_tok_id]
            tok_scores = torch.matmul(
                q_tok_reps.float(), tok_reps.transpose(0, 1)
            ).relu_()  # Bt * Ds
            batched_tok_scores.append(tok_scores)

        for i, q_tok_id in enumerate(batch_qtok_ids):
            ivl_scatter_map = all_ivl_scatter_maps[q_tok_id]
            shard_scatter_map = all_shard_scatter_maps[q_tok_id]

            tok_scores = batched_tok_scores[i]
            ivl_maxed_scores = torch.empty(len(shard_scatter_map))

            for j in range(tok_scores.size(0)):
                ivl_maxed_scores.zero_()
                c_scatter.scatter_max(
                    tok_scores[j].numpy(),
                    ivl_scatter_map.numpy(),
                    ivl_maxed_scores.numpy(),
                )
                boff = q_batch_offsets[q_tok_id][j]
                match_scores[boff].scatter_add_(0, shard_scatter_map, ivl_maxed_scores)

        top_scores, top_iids = match_scores.topk(top_docs, dim=1)
        result = [(doc_id, score.item()) for doc_id, score in zip(top_iids, top_scores)]
        return result


class COILDenseRetriever(DenseRetriever):
    def __init__(
        self,
        question_encoder: COIL,
        batch_size: int,
        tensorizer: Tensorizer,
        index: COILIndex,
    ):
        super().__init__(question_encoder, batch_size, tensorizer, index)

    @staticmethod
    def rebuild_offsets(offset, query_ids):
        query_offsets = defaultdict(list)
        for tok_id in offset:
            start, n_tok = offset[tok_id]
            for off, qid in enumerate(query_ids[start : start + n_tok]):
                query_offsets[qid].append((tok_id, start + off))
        return dict(query_offsets)

    def _get_data_loader(self, questions) -> DataLoader:
        encode_dataset = EncodeDataset(questions, self.tensorizer)
        return DataLoader(
            encode_dataset,
            batch_size=self.batch_size,
            collate_fn=DataCollatorWithPadding(
                self.tensorizer.tokenizer,
                max_length=self.tensorizer.max_length,
                padding="max_length",
            ),
            shuffle=False,
            drop_last=False,
            num_workers=10,
        )

    def generate_question_vectors(
        self, questions: List[str], query_token: str = None
    ) -> COILQuestionRepresentations:
        questions_tokens = []
        encoded = []
        data_loader = self._get_data_loader(questions)
        for batch in data_loader:
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    questions_tokens.extend(batch["input_ids"])
                    for k, v in batch.items():
                        batch[k] = v.to(self.device)
                    cls, reps = self.question_encoder.encode(**batch)
                    encoded.append((cls.cpu(), reps.cpu()))

        all_cls = torch.cat([x[0] for x in encoded]).numpy()
        all_reps = torch.cat([x[1] for x in encoded]).numpy()

        all_qids = []
        tok_rep_dict = defaultdict(list)
        tok_qid_dict = defaultdict(list)

        pad_id = self.tensorizer.get_pad_id()
        sep_id = self.tensorizer.get_sep_token_id()
        for qid, q_tokens in enumerate(questions_tokens):
            all_qids.append(qid)
            rep_dict = defaultdict(list)
            for sent_pos, tok_id in enumerate(q_tokens[1:]):  # skip CLS token
                # token_id: [[rep_vector], [rep_vector]]
                if tok_id == pad_id or tok_id == sep_id:
                    break

                rep_dict[tok_id].append(all_reps[qid][sent_pos + 1])  # skip CLS token

            for tok_id, tok_rep in rep_dict.items():
                tok_rep_dict[tok_id].extend(tok_rep)
                tok_qid_dict[tok_id].extend([qid for _ in range(len(tok_rep))])

        offset_dict = {}
        tok_all_ids = []
        tok_all_reps = []
        _offset = 0
        for tok_id in tok_qid_dict:
            tok_rep = np.stack(tok_rep_dict[tok_id], axis=0)
            offset_dict[tok_id.item()] = (_offset, tok_rep.shape[0])
            _offset += tok_rep.shape[0]
            tok_all_ids.append(np.array(tok_qid_dict[tok_id]))
            tok_all_reps.append(tok_rep)

        # remap to the original coil file names
        cls_pids = all_qids
        cls_reps = all_cls
        tok_pids = np.concatenate(tok_all_ids, axis=0)
        tok_reps = np.concatenate(tok_all_reps, axis=0)
        offsets = offset_dict

        offset_by_query = COILDenseRetriever.rebuild_offsets(offsets, tok_pids.tolist())
        offsets = []
        curr = 0

        _index_order = []

        # reorder representations
        for qid in cls_pids:
            q_offset = []
            for tok_id, off in offset_by_query[qid]:
                q_offset.append((tok_id, curr))
                curr += 1
                _index_order.append(off)
            offsets.append(q_offset)

        assert len(_index_order) == len(tok_reps)
        reps_by_query = tok_reps[_index_order]

        # remap into original coil name
        tok_reps = torch.tensor(reps_by_query)
        cls_reps = torch.tensor(cls_reps)

        del reps_by_query

        return COILQuestionRepresentations(
            tok_reps=tok_reps, offsets=offsets, cls_reps=cls_reps
        )

    def load_encoded_index_data(
        self,
        vector_files: List[str],
        args,
        **kwargs,
    ):
        self.index.load_shards_into_index(vector_files)

    def get_top_docs(
        self, query_vectors: COILQuestionRepresentations, top_docs: int = 100
    ) -> List[Tuple[List[object], List[float]]]:
        ## TODO: optimize searching from all shards
        return self.index.search_top_k(query_vectors, top_docs)
