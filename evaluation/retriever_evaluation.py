import json
import json
import logging
from typing import Dict, List, Tuple

from dpr.data.qa_validation import calculate_chunked_matches, calculate_matches
from dpr.data.retriever_data import TableChunk
from dpr.options import setup_logger

logger = logging.getLogger()
setup_logger(logger)


def validate(
    passages: Dict[object, Tuple[str, str]],
    answers: List[List[str]],
    result_ctx_ids: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
) -> List[List[bool]]:
    match_stats = calculate_matches(passages, answers, result_ctx_ids, workers_num, match_type)
    top_k_hits = match_stats.top_k_hits

    logger.info("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    logger.info("Validation results: top k documents hits accuracy %s", top_k_hits)
    return match_stats.questions_doc_hits


def save_results(
    passages: Dict[object, Tuple[str, str]],
    questions: List[str],
    answers: List[List[str]],
    top_passages_and_scores: List[Tuple[List[object], List[float]]],
    per_question_hits: List[List[bool]],
    out_file: str,
):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    # assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)

        merged_data.append(
            {
                "question": q,
                "answers": q_answers,
                "ctxs": [
                    {
                        "id": results_and_scores[0][c],
                        "title": docs[c][1],
                        "text": docs[c][0],
                        "score": scores[c],
                        "has_answer": hits[c],
                    }
                    for c in range(ctxs_num)
                ],
            }
        )

    with open(out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")
    logger.info("Saved results * scores  to %s", out_file)


def validate_tables(
    passages: Dict[object, TableChunk],
    answers: List[List[str]],
    result_ctx_ids: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
) -> List[List[bool]]:
    match_stats = calculate_chunked_matches(passages, answers, result_ctx_ids, workers_num, match_type)
    top_k_chunk_hits = match_stats.top_k_chunk_hits
    top_k_table_hits = match_stats.top_k_table_hits

    logger.info("Validation results: top k documents hits %s", top_k_chunk_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_chunk_hits]
    logger.info("Validation results: top k table chunk hits accuracy %s", top_k_hits)

    logger.info("Validation results: top k tables hits %s", top_k_table_hits)
    top_k_table_hits = [v / len(result_ctx_ids) for v in top_k_table_hits]
    logger.info("Validation results: top k tables accuracy %s", top_k_table_hits)

    return match_stats.top_k_chunk_hits

"""
Get more usage information from dense_retriever.py
"""
# get evaluation results
# we no longer need the index
# retriever = None

# all_passages = {}
# for ctx_src in ctx_sources:
#     ctx_src.load_data_to(all_passages)
#
# if len(all_passages) == 0:
#     raise RuntimeError("No passages data found. Please specify ctx_file param properly.")
#
# if cfg.validate_as_tables:
#     questions_doc_hits = validate_tables(
#         all_passages,
#         question_answers,
#         top_ids_and_scores,
#         cfg.validation_workers,
#         cfg.match,
#     )
# else:
#     questions_doc_hits = validate(
#         all_passages,
#         question_answers,
#         top_ids_and_scores,
#         cfg.validation_workers,
#         cfg.match,
#     )
#
# if cfg.out_file:
#     save_results(
#         all_passages,
#         questions,
#         question_answers,
#         top_ids_and_scores,
#         questions_doc_hits,
#         cfg.out_file,
#     )
#
# if cfg.kilt_out_file:
#     kilt_ctx = next(iter([ctx for ctx in ctx_sources if isinstance(ctx, KiltCsvCtxSrc)]), None)
#     if not kilt_ctx:
#         raise RuntimeError("No Kilt compatible context file provided")
#     assert hasattr(cfg, "kilt_out_file")
#     kilt_ctx.convert_to_kilt(qa_src.kilt_gold_file, cfg.out_file, cfg.kilt_out_file)