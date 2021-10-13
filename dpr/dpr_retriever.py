import glob
import logging
import pickle
import time
from typing import Iterator, List, Tuple

import numpy as np
import torch
from torch import Tensor as T
from torch import nn

from dpr.indexer.faiss_indexers import (
    DenseIndexer,
)
from dpr.models.biencoder import BiEncoder, _select_span_with_token
from dpr.options import setup_logger
from utils.data_utils import Tensorizer
from retrievers.abstract_dense_retriever import DenseRetriever

logger = logging.getLogger(__name__)
setup_logger(logger)


class DRPLocalFaissDenseRetriever(DenseRetriever):
    """
    Does passage retrieving over the provided index and question encoder
    """

    def __init__(
        self,
        question_encoder: nn.Module,
        batch_size: int,
        tensorizer: Tensorizer,
        index: DenseIndexer,
    ):
        super().__init__(question_encoder, batch_size, tensorizer, index)

    def _iterate_encoded_files(self, vector_files: list, path_id_prefixes: List = None) -> Iterator[Tuple]:
        for i, file in enumerate(vector_files):
            logger.info("Reading file %s", file)
            id_prefix = None
            if path_id_prefixes:
                id_prefix = path_id_prefixes[i]
            with open(file, "rb") as reader:
                doc_vectors = pickle.load(reader)
                for doc in doc_vectors:
                    doc = list(doc)
                    if id_prefix and not str(doc[0]).startswith(id_prefix):
                        doc[0] = "{0}-{1}".format(id_prefix, str(doc[0]))
                    yield doc

    def _get_encoded_ctx_files(self, encoded_ctx_files: List[str], id_prefixes: List[str]) -> (List[str], List[str]):
        """
        Collect all the encoded context files and corresponding document id prefix before loading them into the index
        """
        assert len(encoded_ctx_files) == len(id_prefixes), "ctx len={} pref leb={}".format(
            len(encoded_ctx_files), len(id_prefixes)
        )

        input_paths = []
        path_id_prefixes = []
        for i, pattern in enumerate(encoded_ctx_files):
            pattern_files = glob.glob(pattern)
            pattern_id_prefix = id_prefixes[i]
            input_paths.extend(pattern_files)
            path_id_prefixes.extend([pattern_id_prefix] * len(pattern_files))

        logger.info("Embeddings files id prefixes: %s", path_id_prefixes)
        logger.info("Reading all passages data from files: %s", input_paths)

        return input_paths, path_id_prefixes

    def load_encoded_index_data(
        self,
        vector_files: List[str],
        buffer_size: int,
        doc_prefixes: List = None,
        **kwargs,
    ):
        """
        Indexes encoded passages takes form a list of files
        :param vector_files: file names to get passages vectors from
        :param buffer_size: size of a buffer (amount of passages) to send for the indexing at once
        :return:
        """
        vector_files, doc_prefixes = self._get_encoded_ctx_files(vector_files, doc_prefixes)

        buffer = []
        for i, item in enumerate(self._iterate_encoded_files(vector_files, path_id_prefixes=doc_prefixes)):
            buffer.append(item)
            if 0 < buffer_size == len(buffer):
                self.index.index_data(buffer)
                buffer = []
        self.index.index_data(buffer)
        logger.info("Data indexing completed.")

    def get_top_docs(self, query_vectors, top_docs: int = 100) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        time0 = time.time()
        query_vectors = query_vectors.numpy()
        results = self.index.search_knn(query_vectors, top_docs)
        logger.info("index search time: %f sec.", time.time() - time0)
        self.index = None
        return results

    def generate_question_vectors(self, questions: List[str], query_token: str = None, ) -> T:
        n = len(questions)
        query_vectors = []

        with torch.no_grad():
            for j, batch_start in enumerate(range(0, n, self.batch_size)):
                batch_questions = questions[batch_start: batch_start + self.batch_size]

                if query_token:
                    # TODO: tmp workaround for EL, remove or revise
                    if query_token == "[START_ENT]":
                        batch_token_tensors = [
                            _select_span_with_token(q, self.tensorizer, token_str=query_token) for q in batch_questions
                        ]
                    else:
                        batch_token_tensors = [
                            self.tensorizer.text_to_tensor(" ".join([query_token, q])) for q in batch_questions
                        ]
                else:
                    batch_token_tensors = [self.tensorizer.text_to_tensor(q) for q in batch_questions]

                q_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
                q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
                q_attn_mask = self.tensorizer.get_attn_mask(q_ids_batch)

                if self.selector:
                    rep_positions = self.selector.get_positions(q_ids_batch, self.tensorizer)

                    _, out, _ = BiEncoder.get_representation(
                        self.question_encoder,
                        q_ids_batch,
                        q_seg_batch,
                        q_attn_mask,
                        representation_token_pos=rep_positions,
                    )
                else:
                    _, out, _ = self.question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

                query_vectors.extend(out.cpu().split(1, dim=0))

                if len(query_vectors) % 100 == 0:
                    logger.info("Encoded queries %d", len(query_vectors))

        query_tensor = torch.cat(query_vectors, dim=0)
        logger.info("Total encoded queries tensor %s", query_tensor.size())
        assert query_tensor.size(0) == len(questions)
        return query_tensor
