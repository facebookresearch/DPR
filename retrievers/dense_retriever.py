#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from coil.coil_retriever import COILDenseRetriever
from dpr.dpr_retriever import DRPLocalFaissDenseRetriever
from dpr.models import init_question_encoder_components
from dpr.options import set_cfg_params_from_state, setup_cfg_gpu, setup_logger
from dpr.utils.model_utils import (get_model_obj, load_states_from_checkpoint, setup_for_distributed_mode)

logger = logging.getLogger()
setup_logger(logger)

SUPPORTED_RETRIEVERS = {
    'dpr': DRPLocalFaissDenseRetriever,
    'coil': COILDenseRetriever,
}


def load_state_dict_to_model(saved_state, model, encoder_prefix=None):
    logger.info("Loading saved model state ...")

    state_to_load = saved_state
    if encoder_prefix:
        logger.info("Selecting question_encoder: %s", encoder_prefix)

        prefix_len = len(encoder_prefix)
        state_to_load = {
            key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if key.startswith(encoder_prefix)
        }

    model.load_state_dict(state_to_load, strict=False)


@hydra.main(config_path="../conf", config_name="coil_retriever")
def main(cfg: DictConfig):
    cfg = setup_cfg_gpu(cfg)
    logger.info("CFG (after gpu  configuration):")
    logger.info("%s", OmegaConf.to_yaml(cfg))

    tensorizer, question_encoder, _ = init_question_encoder_components(
        cfg.encoder.encoder_model_type, cfg, inference_only=True)

    question_encoder, _ = setup_for_distributed_mode(
        question_encoder, None, cfg.device, cfg.n_gpu, cfg.local_rank, cfg.fp16)

    question_encoder.eval()

    vector_size = question_encoder.get_out_size()

    # if load from weights
    if cfg.model_file:
        saved_state = load_states_from_checkpoint(cfg.model_file)
        set_cfg_params_from_state(saved_state.encoder_params, cfg)
        # load weights from the model file
        model_to_load = get_model_obj(question_encoder)

        # TODO rename this into encoder prefix, dpr question model prefix is "question_model."
        load_state_dict_to_model(saved_state, model_to_load, cfg.encoder_path)
        vector_size = model_to_load.get_out_size()
        logger.info("Encoder vector_size=%d", vector_size)

    # get questions & answers
    questions = []

    if not cfg.qa_dataset:
        logger.warning("Please specify qa_dataset to use")
        # question = input("Please enter a questions: ")
        question = "term service agreement definition"
        questions.append(question)

    else:
        print("Use qa_dataset or uncomment below")
        ds_key = cfg.qa_dataset
        logger.info("qa_dataset: %s", ds_key)

        qa_src = hydra.utils.instantiate(cfg.datasets[ds_key])
        qa_src.load_data()

        for ds_item in qa_src.data[0:5]:
            question, answers = ds_item.query, ds_item.answers
            questions.append(question)

    index = hydra.utils.instantiate(cfg.indexers[cfg.indexer])
    logger.info("Index class %s ", type(index))
    index_buffer_sz = index.buffer_size
    index.init_index(vector_size)

    retriever_class = SUPPORTED_RETRIEVERS.get(cfg.retriever)
    if not retriever_class:
        raise ValueError(f"{cfg.retriever} is not supported")

    retriever = retriever_class(question_encoder, cfg.batch_size, tensorizer, index)

    logger.info("Using special token %s", cfg.special_query_token)
    questions_tensor = retriever.generate_question_vectors(questions, query_token=cfg.special_query_token)

    logger.info("Loading encoded document into index")
    retriever.load_encoded_index_data(cfg.encoded_ctx_files, index_buffer_sz, doc_prefixes=cfg.doc_prefixes)

    # get top k results
    top_ids_and_scores = retriever.get_top_docs(questions_tensor, cfg.n_docs)

    results = []
    for q, id_score in zip(questions, top_ids_and_scores):
        doc_id, score = id_score[0], id_score[1]
        results.append((q, doc_id, score))

    print(results)
    import pdb;
    pdb.set_trace()


if __name__ == "__main__":
    main()
