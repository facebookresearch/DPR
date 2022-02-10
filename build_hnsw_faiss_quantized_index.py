#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc. All rights reserved.

import argparse
import glob
import logging
import pickle
from typing import List, Tuple, Iterator
import faiss
import numpy as np
import random

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


class DenseIndexer(object):
    def __init__(self, buffer_size: int = 50000):
        self.buffer_size = buffer_size
        self.index_id_to_db_id = []
        self.index = None

    def _update_id_mapping(self, db_ids: List):
        self.index_id_to_db_id.extend(db_ids)

    def index_data(self, data: List[Tuple[object, np.array]]):
        raise NotImplementedError

    def search_knn(
        self,
        query_vectors: np.array,
        top_docs: int,
        with_vectors: bool = False,
        add_random=False,
        all_random=False,
    ) -> List[Tuple[List[object], List[float]]]:
        raise NotImplementedError

    def serialize(self, file: str):
        logger.info("Serializing index to %s", file)

        index_file = file + ".index.dpr"
        meta_file = file + ".index_meta.dpr"

        faiss.write_index(self.index, index_file)
        with open(meta_file, mode="wb") as f:
            pickle.dump(self.index_id_to_db_id, f)

    def deserialize_from(self, file: str):
        logger.info("Loading index from %s", file)

        index_file = file + ".index.dpr"
        meta_file = file + ".index_meta.dpr"

        self.index = faiss.read_index(index_file)
        logger.info(
            "Loaded index of type %s and size %d", type(self.index), self.index.ntotal
        )

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert (
            len(self.index_id_to_db_id) == self.index.ntotal
        ), "Deserialized index_id_to_db_id should match faiss index size"


class DenseFlatIndexer(DenseIndexer):
    def __init__(self, vector_sz: int, buffer_size: int = 50000):
        super(DenseFlatIndexer, self).__init__(buffer_size=buffer_size)
        self.index = faiss.IndexFlatIP(vector_sz)

    def index_data(self, data: List[Tuple[object, np.array]]):
        n = len(data)
        # indexing in batches is beneficial for many faiss index types
        for i in range(0, n, self.buffer_size):
            db_ids = [t[0] for t in data[i : i + self.buffer_size]]
            vectors = [
                np.reshape(t[1], (1, -1)) for t in data[i : i + self.buffer_size]
            ]
            vectors = np.concatenate(vectors, axis=0)
            self._update_id_mapping(db_ids)
            self.index.add(vectors)

        indexed_cnt = len(self.index_id_to_db_id)
        logger.info("Total data indexed %d", indexed_cnt)

    def search_knn(
        self,
        query_vectors: np.array,
        top_docs: int,
        with_vectors: bool = False,
        add_random=False,
        all_random=False,
    ) -> List[Tuple[List[object], List[float]]]:
        scores, indexes = self.index.search(query_vectors, top_docs)
        if with_vectors:
            vectors = np.array(
                [
                    [self.index.reconstruct(int(idx)) for idx in ex_indexes]
                    for ex_indexes in indexes
                ]
            )
        # convert to external ids
        db_ids = [
            [self.index_id_to_db_id[i] for i in query_top_idxs]
            for query_top_idxs in indexes
        ]
        result = [(db_ids[i], scores[i]) for i in range(len(db_ids))]
        return (result, vectors) if with_vectors else result


class DenseHNSWSQ8Indexer(DenseIndexer):
    def __init__(
        self,
        vector_sz: int,
        buffer_size: int = 50000,
        store_n: int = 512,
        ef_search: int = 128,
        ef_construction: int = 200,
    ):
        super(DenseHNSWSQ8Indexer, self).__init__(buffer_size=buffer_size)

        # IndexHNSWFlat supports L2 similarity only
        # so we have to apply DOT -> L2 similairy space conversion with the help of an extra dimension
        # index = faiss.IndexHNSWFlat(vector_sz + 1, store_n)
        index = faiss.IndexHNSWSQ(vector_sz + 1, faiss.ScalarQuantizer.QT_8bit, store_n)
        index.hnsw.efSearch = ef_search
        index.hnsw.efConstruction = ef_construction
        self.index = index
        self.phi = 0

    def index_data(self, data: List[Tuple[object, np.array]]):
        n = len(data)

        # max norm is required before putting all vectors in the index to convert inner product similarity to L2
        if self.phi > 0:
            raise RuntimeError(
                "DPR HNSWF index needs to index all data at once,"
                "results will be unpredictable otherwise."
            )
        phi = 0
        for i, item in enumerate(data):
            id, doc_vector = item
            norms = (doc_vector ** 2).sum()
            phi = max(phi, norms)
        logger.info("HNSWF phi={}".format(phi))
        self.phi = phi

        # indexing in batches is beneficial for many faiss index types
        for i in range(0, n, self.buffer_size):
            db_ids = [t[0] for t in data[i : i + self.buffer_size]]
            vectors = [
                np.reshape(t[1], (1, -1)) for t in data[i : i + self.buffer_size]
            ]

            norms = [(doc_vector ** 2).sum() for doc_vector in vectors]
            aux_dims = [np.sqrt(phi - norm) for norm in norms]
            hnsw_vectors = [
                np.hstack((doc_vector, aux_dims[i].reshape(-1, 1)))
                for i, doc_vector in enumerate(vectors)
            ]
            hnsw_vectors = np.concatenate(hnsw_vectors, axis=0)
            self.index.train(hnsw_vectors)

            self._update_id_mapping(db_ids)
            self.index.add(hnsw_vectors)

        indexed_cnt = len(self.index_id_to_db_id)
        logger.info("Total data indexed %d", indexed_cnt)

    def search_knn(
        self,
        query_vectors: np.array,
        top_docs: int,
        with_vectors: bool = False,
        add_random=False,
        all_random=False,
    ) -> List[Tuple[List[object], List[float]]]:

        aux_dim = np.zeros(len(query_vectors), dtype="float32")
        query_nhsw_vectors = np.hstack((query_vectors, aux_dim.reshape(-1, 1)))
        logger.info("query_hnsw_vectors %s", query_nhsw_vectors.shape)
        scores, indexes = self.index.search(query_nhsw_vectors, top_docs)
        if all_random:
            for rr in range(top_docs):
                random_row = np.array(
                    [
                        random.randint(0, self.index.ntotal)
                        for _ in range(query_vectors.shape[0])
                    ]
                )
                indexes[:, rr] = random_row

        if add_random:
            random_row = np.array(
                [
                    random.randint(0, self.index.ntotal)
                    for _ in range(query_vectors.shape[0])
                ]
            )
            indexes[:, -1] = random_row
        if with_vectors:
            vectors = np.array(
                [
                    [self.index.reconstruct(int(idx))[:-1] for idx in ex_indexes]
                    for ex_indexes in indexes
                ]
            )
        # convert to external ids
        db_ids = [
            [self.index_id_to_db_id[i] for i in query_top_idxs]
            for query_top_idxs in indexes
        ]
        result = [(db_ids[i], scores[i]) for i in range(len(db_ids))]
        return (result, vectors) if with_vectors else result

    def deserialize_from(self, file: str):
        super(DenseHNSWSQ8Indexer, self).deserialize_from(file)
        # to trigger warning on subsequent indexing
        self.phi = 1


def iterate_encoded_files(vector_files: list) -> Iterator[Tuple[object, np.array]]:
    for i, file in enumerate(vector_files):
        logger.info("Reading file %s", file)
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                # TODO: remove old format support
                if len(doc) == 3:  # old format
                    title, db_id, doc_vector = doc
                else:
                    db_id, doc_vector = doc
                yield db_id, doc_vector


def main(args):
    index_buffer_sz = 100000000000000
    index = DenseHNSWSQ8Indexer(768, buffer_size=index_buffer_sz, store_n=args.n)
    buffer = []
    input_paths = glob.glob(args.ctx_files_pattern)
    logger.info("Reading all passages data from files =%s", input_paths)

    for i, item in enumerate(iterate_encoded_files(input_paths)):
        db_id, doc_vector = item
        buffer.append((db_id, doc_vector))
    index.index_data(buffer)
    logger.info("Data indexing completed.")
    index.serialize(args.dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ctx_files_pattern",
        type=str,
        default="",
        help="glob-pattern for document vectors to build index for",
    )
    parser.add_argument(
        "--dump_path",
        type=str,
        default="",
        help="path to write faiss index to",
    )
    parser.add_argument("--n", default=512, type=int)
    args = parser.parse_args()
    main(args)
