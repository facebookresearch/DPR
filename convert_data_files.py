#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import csv
import glob
import json
import logging
import os
import pickle
import re
import string
from typing import List, Tuple, Optional

import jsonlines

from dpr.data.qa_validation import has_answer
from dpr.utils.data_utils import normalize_question
from dpr.utils.tokenizers import SimpleTokenizer

logger = logging.getLogger()

logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

# -------------------- KILT eval ---------------------------------


def has_answer_kilt(answers, text) -> bool:
    text = normalize_kilt(text)
    for single_answer in answers:
        single_answer = normalize_kilt(single_answer)
        if single_answer in text:
            return True
    return False


def normalize_kilt(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# -----------------------------------------------------


def norm_question_conv(q):
    q = normalize_question(q)
    q = q.strip()
    # if q[-1] == "?":
    #    q = q[:-1]
    q = q.replace("?", "")
    return q.strip().lower()


def eval_dpr_retriever_out_file(file: str):

    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    with open(file, "r", encoding="utf-8") as f:
        logger.info("Reading file %s" % file)
        data = json.load(f)
        logger.info("Aggregated data size: {}".format(len(data)))

    top_k_hits = [0] * 100
    n = len(data)
    for sample in data:
        answers = sample["answers"]
        candidates = sample["ctxs"]
        for i, ctx in enumerate(candidates):
            passage = ctx["text"]

            if has_answer(answers, passage, tokenizer, "string"):
                top_k_hits[i:] = [v + 1 for v in top_k_hits[i:]]
                break
    logger.info("top_k_hits=%s", top_k_hits)
    logger.info("accuracy ratio =%s", [float(v / n) for v in top_k_hits])


def convert_retriever_data_to_retriever_input(
    data,
    out,
    gold_file=None,
    max_positives=2,
    max_negatives=10,
):

    gold_data = []
    if gold_file:
        with open(gold_file, "r", encoding="utf-8") as f:
            logger.info("Reading file %s" % gold_file)
            gold_data = json.load(f)
            logger.info("Gold aggregated data size: {}".format(len(gold_data)))

    if gold_data:  # convert to dict
        gold_data = {norm_question_conv(s["question"]): s for s in gold_data}

    no_question_in_gold = 0
    n = len(data)
    top_k_hits = [0] * 100

    for sample in data:
        ctxs = sample["ctxs"]

        best_hit = next((i for i, x in enumerate(ctxs) if x["has_answer"] == True), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

        del sample["ctxs"]
        positive_ctxs = [ctx for ctx in ctxs if ctx["has_answer"] is True][0:max_positives]
        gold_passage = None
        if gold_data:  # insert @ first position
            q = sample["question"]
            q = norm_question_conv(q)

            if q in gold_data:
                gold_passage = gold_data[q]["positive_ctxs"][0]
            else:
                no_question_in_gold += 1
        negative_ctxs = [ctx for ctx in ctxs if ctx["has_answer"] is False][0:max_negatives]

        sample["negative_ctxs"] = []
        sample["hard_negative_ctxs"] = negative_ctxs
        if gold_passage:
            positive_ctxs = [gold_passage] + positive_ctxs
        sample["positive_ctxs"] = positive_ctxs

    logger.info("top_k_hits %s", top_k_hits)
    logger.info("accuracy ratio =%s", [float(v / n) for v in top_k_hits])

    # skip samples with no positives
    data = [sample for sample in data if len(sample["positive_ctxs"]) > 0]
    logger.info("result data size %d", len(data))
    logger.info("no_question_in_gold %d", no_question_in_gold)

    with open(out, "w", encoding="utf-8") as f:
        logger.info("Writing file %s" % out)
        f.write(json.dumps(data, indent=4) + "\n")


def convert_retriever_data_to_retriever_input_entity_restricted(
    data,
    out,
    kilt_gold_file,
    gold_file=None,
    max_positives=2,
    max_negatives=10,
):
    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    gold_data = []
    if gold_file:
        with open(gold_file, "r", encoding="utf-8") as f:
            logger.info("Reading file %s" % gold_file)
            gold_data = json.load(f)
            logger.info("Gold aggregated data size: {}".format(len(gold_data)))

    question_to_entities = {}
    with jsonlines.open(kilt_gold_file, mode="r") as jsonl_reader:
        for k, kilt_sample in enumerate(jsonl_reader):
            kilt_q = norm_question_conv(kilt_sample["input"])
            if (
                kilt_sample["input"]
                == "What WWII Fleet Admiral of the United States Navy was on the board of the American Cable and Radio Corporation? "
            ):
                logger.info("!!! sample = %s", kilt_q)
            ents = [s["provenance"][0]["title"].lower() for s in kilt_sample["output"] if "provenance" in s]
            expanded_ents = []
            for entity in ents:
                candidate = entity.replace("(", "").replace(")", "")
                if candidate != entity:
                    expanded_ents.append(candidate)
                    # logger.info("new ent: %s", candidate)
            question_to_entities[kilt_q] = ents + expanded_ents

    if gold_data:  # convert to dict
        gold_data = {norm_question_conv(s["question"]): s for s in gold_data}

    no_question_in_gold = 0
    n = len(data)
    top_k_hits = [0] * 100
    top_k_hits_with_ent = [0] * 100
    no_q_in_ents = 0

    for sample in data:
        ctxs = sample["ctxs"]
        q = sample["question"]
        q = norm_question_conv(q)

        best_hit = next((i for i, x in enumerate(ctxs) if x["has_answer"] == True), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

        entities = question_to_entities.get(q, [])
        if not entities:
            no_q_in_ents += 1
            logger.warning("No question found in question_to_entities: %s", q)

        for i, ctx in enumerate(ctxs):
            if ctx["has_answer"] == True:
                # check entity
                if any([e for e in entities if e in (ctx["text"] + " " + ctx["text"])]):
                    ctx["has_ent"] = True
                else:
                    ctx["has_ent"] = False

        best_hit_with_ent = next(
            (i for i, x in enumerate(ctxs) if x["has_answer"] == True and x["has_ent"] == True), None
        )

        if best_hit_with_ent is not None:
            top_k_hits_with_ent[best_hit_with_ent:] = [v + 1 for v in top_k_hits_with_ent[best_hit_with_ent:]]

        del sample["ctxs"]
        positive_ctxs = [ctx for ctx in ctxs if ctx["has_answer"] is True and ctx["has_ent"] is True][0:max_positives]
        if not positive_ctxs:
            positive_ctxs = [ctx for ctx in ctxs if ctx["has_answer"] is True][0:max_positives]
        gold_passage = None
        if gold_data:  # insert @ first position
            if q in gold_data:
                gold_passage = gold_data[q]["positive_ctxs"][0]
                answers = sample["answers"]
                gold_passage_text = gold_passage["title"] + " " + gold_passage["text"]
                if not has_answer(answers, gold_passage_text, tokenizer, match_type="string") and not has_answer_kilt(
                    answers, gold_passage_text
                ):
                    logger.warning("No answer in gold: answers=%s, gp=%s", answers, gold_passage_text)
                    gold_passage = None

            else:
                logger.warning("No question found in gold_data: %s", q)
                no_question_in_gold += 1
        negative_ctxs = [ctx for ctx in ctxs if ctx["has_answer"] is False][0:max_negatives]

        sample["negative_ctxs"] = []
        sample["hard_negative_ctxs"] = negative_ctxs
        if gold_passage:
            positive_ctxs = [gold_passage] + positive_ctxs
        sample["positive_ctxs"] = positive_ctxs

    logger.info("no_q_in_ents %s", no_q_in_ents)
    logger.info("top_k_hits %s", top_k_hits)
    logger.info("accuracy ratio =%s", [float(v / n) for v in top_k_hits])

    logger.info("top_k_hits_with_ent %s", top_k_hits_with_ent)
    logger.info("top_k_hits_with_ent ratio =%s", [float(v / n) for v in top_k_hits_with_ent])

    # skip samples with no positives
    data = [sample for sample in data if len(sample.get("positive_ctxs", [])) > 0]
    logger.info("result data size %d", len(data))
    logger.info("no_question_in_gold %d", no_question_in_gold)

    with open(out, "w", encoding="utf-8") as f:
        logger.info("Writing file %s" % out)
        f.write(json.dumps(data, indent=4) + "\n")


def convert_retriever_results_to_retriever_input_old(
    path, out, max_positives, max_negatives, gold_file=None, kilt_gold_file=None
):
    with open(path, "r", encoding="utf-8") as f:
        logger.info("Reading file %s" % path)
        data = json.load(f)
        logger.info("Aggregated data size: {}".format(len(data)))
    if kilt_gold_file:
        convert_retriever_data_to_retriever_input_entity_restricted(
            data, out, kilt_gold_file, gold_file=gold_file, max_positives=max_positives, max_negatives=max_negatives
        )
    else:
        convert_retriever_data_to_retriever_input(
            data, out, gold_file=gold_file, max_positives=max_positives, max_negatives=max_negatives
        )


def do_subject_check(q: str, ctx: dict, answers: List, tokenizer, is_slot_task: bool) -> Tuple[int, Optional[dict]]:
    text = ctx["text"]
    if "has_answer" in ctx:
        has_ans = ctx["has_answer"]
    else:
        has_ans = has_answer(answers, text, tokenizer, "string")

    has_ans = has_ans or has_answer_kilt(answers, text)
    ctx["has_answer"] = has_ans

    if is_slot_task:
        subject = q[: q.index("[SEP]")].strip()
        if has_ans:
            if has_answer([subject], text, tokenizer, match_type="string") or has_answer_kilt([subject], text):
                # positives_found += 1
                # new_candidates.append(ctx)
                return 1, ctx
            else:
                return 0, None  # hard case, better to discard
        else:
            return 0, ctx

    return 1 if has_ans else 0, ctx


def convert_retriever_results_to_retriever_input(
    dpr_file: str,
    gold_file: str,
    out_file: str,
    is_slot_task: bool = False,
    kilt_gold_file: str = None,
    max_positives=2,
    max_negatives=10,
):
    with open(dpr_file, "r", encoding="utf-8") as f:
        logger.info("Reading file %s" % dpr_file)
        data = json.load(f)
        logger.info("Aggregated data size: {}".format(len(data)))

    # data = data[0:1000]
    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    for k, sample in enumerate(data):
        answers = sample["answers"]
        candidates = sample["ctxs"]
        positives_found = 0
        new_candidates = []
        for i, ctx in enumerate(candidates):
            q = sample["question"]

            is_positive, candidate_ctx = do_subject_check(q, ctx, answers, tokenizer, is_slot_task)
            if is_positive:
                positives_found += 1
            if candidate_ctx:
                new_candidates.append(candidate_ctx)

            if len(new_candidates) >= max_negatives + max_positives and positives_found >= max_positives:
                break

        sample["ctxs"] = new_candidates
        if k % 1000 == 0:
            logger.info("Processed %d", k)

    if kilt_gold_file:
        convert_retriever_data_to_retriever_input_entity_restricted(
            data,
            out_file,
            kilt_gold_file,
            gold_file=gold_file,
            max_positives=max_positives,
            max_negatives=max_negatives,
        )
    else:
        convert_retriever_data_to_retriever_input(data, out_file, gold_file=gold_file)


def convert_bm25_kilt_results_to_dpr_input(
    input_file,
    gold_file,
    out_file,
    id_to_title: dict = {},
    is_slot_task: bool = False,
    kilt_gold_file: str = None,
    max_positives=2,
    max_negatives=10,
):
    data = []
    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    logger.info("Reading %s", input_file)
    with jsonlines.open(input_file, mode="r") as jsonl_reader:
        for k, ex in enumerate(jsonl_reader):
            d = {}
            q = ex["input"]
            d["question"] = q
            answers = set()
            for a in ex["output"]:
                if "answer" in a:
                    answers.add(a["answer"])
            answers = list(answers)
            d["answers"] = answers
            d["id"] = ex["id"]
            new_candidates = []
            positives_found = False
            for c in ex["output"][0]["provenance"]:
                text = c["text"]
                id = c["title"]
                title = id_to_title[id]
                ctx = {
                    "id": id,
                    "title": title,
                    "text": text,
                }
                is_positive, candidate_ctx = do_subject_check(q, ctx, answers, tokenizer, is_slot_task)
                if is_positive:
                    positives_found += 1
                if candidate_ctx:
                    new_candidates.append(candidate_ctx)

                if len(new_candidates) >= max_negatives + max_positives and positives_found >= max_positives:
                    break

            d["ctxs"] = new_candidates
            # assert all("has_answer" in p for p in all_passages)
            data.append(d)
            if k % 1000 == 0:
                logger.info("Processed %d", k)

    if kilt_gold_file:
        convert_retriever_data_to_retriever_input_entity_restricted(
            data,
            out_file,
            kilt_gold_file,
            gold_file=gold_file,
            max_positives=max_positives,
            max_negatives=max_negatives,
        )
    else:
        convert_retriever_data_to_retriever_input(data, out_file, gold_file=gold_file)


def create_cc_net_eval_passages(dense_root, dense_files: List, bm25_root, bm25_files: List, out_file):

    passages = []  # (id, text, title)
    for dpr_file in dense_files:
        dpr_file = os.path.join(dense_root, dpr_file)
        with open(dpr_file, "r", encoding="utf-8") as f:
            logger.info("Reading file %s" % dpr_file)
            data = json.load(f)
            logger.info("Aggregated data size: {}".format(len(data)))
        for s in data:
            ctxs = [(c["id"], c["text"], c["title"]) for c in s["ctxs"]]
            passages.extend(ctxs)
        logger.info("passages size: %s", len(passages))

    with open("/checkpoint/piktus/sphere/chunkid_title.pkl", "rb") as f:
        id_to_title = pickle.load(f)
    logger.info("id_to_title  len %d", len(id_to_title))

    for file in bm25_files:
        file = os.path.join(bm25_root, file)
        with jsonlines.open(file, mode="r") as jsonl_reader:
            logger.info("Reading file %s" % file)
            for k, ex in enumerate(jsonl_reader):
                for c in ex["output"][0]["provenance"]:
                    text = c["text"]
                    id = c["title"]
                    title = id_to_title[id]
                    passages.append((id, text, title))
        logger.info("passages size: %s", len(passages))
    logger.info("Saving to %s", out_file)
    with open(out_file, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        for r in passages:
            writer.writerow(r)


def find_phi():
    files = glob.glob("/checkpoint/vladk/biencoder/cc_net5_embs_sp_ccnet_finetune_paq_all_lr2e5_40e_32/wiki_passages_*")
    phi = 0
    for file in files:
        with open(file, "rb") as reader:
            logger.info("Reading {}".format(file))
            try:
                doc_vectors_with_id = pickle.load(reader)
            except Exception:
                logging.info("Couldn't unpickle {}".format(file))

            for i, item in enumerate(doc_vectors_with_id):
                id, doc_vector = item[0:2]
                norms = (doc_vector ** 2).sum()
                phi = max(phi, norms)
        logger.info("HNSWF DotProduct -> L2 space phi={}".format(phi))
    logger.info("HNSWF DotProduct -> L2 space phi={}".format(phi))


def update_is_wiki_in_meta():
    with open("/checkpoint/piktus/sphere/chunkid_is_wiki.pkl", "rb") as f:
        id_to_is_wiki = pickle.load(f)
    logger.info("id_to_is_wiki %d", len(id_to_is_wiki))
    root = "/checkpoint/vladk/faiss-indexes/hnswsq_cc-net2/"
    is_wiki_count = 0

    for i in range(31):
        meta_file = os.path.join(root, str(i), "meta.pkl")
        logger.info("Loading meta file from %s", meta_file)
        with open(meta_file, "rb") as reader:
            meta = pickle.load(reader)
        logger.info("metadata deserialized, size: %d", len(meta))
        new_meta = []
        for k, s in enumerate(meta):
            id = s[0]
            is_wiki = id_to_is_wiki[id] == "1"
            assert len(s) == 4
            if is_wiki:
                is_wiki_count += 1
            new_sample = (s[0], s[1], s[2], is_wiki)
            new_meta.append(new_sample)
            if k % 100000 == 0:
                logger.info("k=%d, is_wiki_count=%d", k, is_wiki_count)
        logger.info("new meta size %d", len(new_meta))
        meta_file = os.path.join(root, str(i), "meta_fx.pkl")
        with open(meta_file, mode="wb") as f:
            pickle.dump(new_meta, f)
        logger.info(f"Saved index & meta to: {meta_file}")


def main():
    """
    root = "/checkpoint/vladk/biencoder/val_ur_ccnet2_sp_ccnet_finetune_paq_all_lr2e5_40e/"
    convert_retriever_results_to_retriever_input(
        f"{root}/trivia_train.json",
        "/checkpoint/fabiopetroni/KILT/multitask_dpr_datasets/v0.20200817/triviaqa-train-multikilt.json",
        f"{root}/trivia_train_dpr_input.json",
        is_slot_task=False,
        kilt_gold_file="/checkpoint/fabiopetroni/KILT/datasets/triviaqa-train-kilt.jsonl",
    )

    convert_retriever_results_to_retriever_input(
        f"{root}/hotpot_train.json",
        "/checkpoint/fabiopetroni/KILT/multitask_dpr_datasets/v0.20200817/hotpotqa-train-multikilt.json",
        f"{root}/hotpot_train_dpr_input.json",
        is_slot_task=False,
        kilt_gold_file="/checkpoint/fabiopetroni/KILT/datasets/hotpotqa-train-kilt.jsonl",
    )
    convert_retriever_results_to_retriever_input(
        f"{root}/nq_train.json",
        "/checkpoint/fabiopetroni/KILT/multitask_dpr_datasets/v0.20200817/nq-train-multikilt.json",
        f"{root}/nq_train-dpr_input.json",
        is_slot_task=False,
        kilt_gold_file="/checkpoint/fabiopetroni/KILT/datasets/nq-train-kilt.jsonl",
    )

    convert_retriever_results_to_retriever_input(
        f"{root}/zeroshot_train.json",
        "/checkpoint/fabiopetroni/KILT/multitask_dpr_datasets/v0.20200817/zeroshot-train-multikilt.json",
        f"{root}/zeroshot_train_dpr_input.json",
        is_slot_task=True,
        kilt_gold_file="/checkpoint/fabiopetroni/KILT/datasets/structured_zeroshot-train-kilt.jsonl",
    )
    for i in range(1, 7):
        convert_retriever_results_to_retriever_input(
            f"{root}/trex_train{i}.json",
            "/checkpoint/fabiopetroni/KILT/multitask_dpr_datasets/v0.20200817/trex-train-multikilt.json",
            f"{root}/trex_train{i}_dpr_input.json",
            is_slot_task=True,
            kilt_gold_file="/checkpoint/fabiopetroni/KILT/datasets/trex-train-kilt.jsonl",
        )
    """

    with open("/checkpoint/piktus/sphere/chunkid_title.pkl", "rb") as f:
        id_to_title = pickle.load(f)
    logger.info("id_to_title  len %d", len(id_to_title))

    convert_bm25_kilt_results_to_dpr_input(
        "/checkpoint/piktus/sphere/fid_data/bm25/cc_net/nq-dev-kilt.jsonl",
        "/checkpoint/fabiopetroni/KILT/multitask_dpr_datasets/v0.20200817/nq-dev-multikilt.json",
        "/checkpoint/vladk/biencoder/cc-net2019-results-bm25-eh/nq-dev_dpr_input.json",
        id_to_title=id_to_title,
        kilt_gold_file="/checkpoint/fabiopetroni/KILT/datasets/nq-dev-kilt.jsonl",
    )

    convert_bm25_kilt_results_to_dpr_input(
        "/checkpoint/piktus/sphere/fid_data/bm25/cc_net/hotpotqa-train-kilt.jsonl",
        "/checkpoint/fabiopetroni/KILT/multitask_dpr_datasets/v0.20200817/hotpotqa-train-multikilt.json",
        "/checkpoint/vladk/biencoder/cc-net2019-results-bm25-eh/hotpot-train_dpr_input.json",
        id_to_title=id_to_title,
        kilt_gold_file="/checkpoint/fabiopetroni/KILT/datasets/hotpotqa-train-kilt.jsonl",
    )

    convert_bm25_kilt_results_to_dpr_input(
        "/checkpoint/piktus/sphere/fid_data/bm25/cc_net/nq-train-kilt.jsonl",
        "/checkpoint/fabiopetroni/KILT/multitask_dpr_datasets/v0.20200817/nq-train-multikilt.json",
        "/checkpoint/vladk/biencoder/cc-net2019-results-bm25-eh/nq-train_dpr_input.json",
        id_to_title=id_to_title,
        kilt_gold_file="/checkpoint/fabiopetroni/KILT/datasets/nq-train-kilt.jsonl",
    )

    convert_bm25_kilt_results_to_dpr_input(
        "/checkpoint/piktus/sphere/fid_data/bm25/cc_net/triviaqa-train-kilt.jsonl",
        "/checkpoint/fabiopetroni/KILT/multitask_dpr_datasets/v0.20200817/triviaqa-train-multikilt.json",
        "/checkpoint/vladk/biencoder/cc-net2019-results-bm25-eh/trivia-train_dpr_input.json",
        id_to_title=id_to_title,
        kilt_gold_file="/checkpoint/fabiopetroni/KILT/datasets/triviaqa-train-kilt.jsonl",
    )

    convert_bm25_kilt_results_to_dpr_input(
        "/checkpoint/piktus/sphere/fid_data/bm25/cc_net/structured_zeroshot-train-kilt.jsonl",
        "/checkpoint/fabiopetroni/KILT/multitask_dpr_datasets/v0.20200817/zeroshot-train-multikilt.json",
        "/checkpoint/vladk/biencoder/cc-net2019-results-bm25-eh/zeroshot-train_dpr_input.json",
        id_to_title=id_to_title,
        is_slot_task=True,
        kilt_gold_file="/checkpoint/fabiopetroni/KILT/datasets/structured_zeroshot-train-kilt.jsonl",
    )

    convert_bm25_kilt_results_to_dpr_input(
        "/checkpoint/piktus/sphere/fid_data/bm25/cc_net/trex-train-kilt.jsonl",
        "/checkpoint/fabiopetroni/KILT/multitask_dpr_datasets/v0.20200817/trex-train-multikilt.json",
        "/checkpoint/vladk/biencoder/cc-net2019-results-bm25-eh/trex-train_dpr_input.json",
        id_to_title=id_to_title,
        is_slot_task=True,
        kilt_gold_file="/checkpoint/fabiopetroni/KILT/datasets/trex-train-kilt.jsonl",
    )


if __name__ == "__main__":
    main()
