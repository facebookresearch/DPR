#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import csv
import glob
import gzip
import json
import logging
import os
import pickle
import random
from typing import List, Dict, Tuple

import jsonlines
import torch

from dpr.data.speech_data import _get_id_to_audio_file_map_paq
from dpr.data.qa_validation import has_answer, has_answer_kilt
from dpr.data.reader_data import (
    ReaderSample,
    ReaderPassage,
    _find_answer_positions,
    _concat_pair,
)
from dpr.models.hf_models import BertTensorizer
from dpr.utils.data_utils import normalize_question, read_data_from_jsonl_files
from dpr.utils.data_utils import read_data_from_json_files
from dpr.utils.tokenizers import SimpleTokenizer

logger = logging.getLogger()

logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


def text_to_tensor(text, tensorizer, add_special_tokens: bool, title=None):
    # simple_retokenized = ' '.join(stokenizer.tokenize(text).words(uncased=True))
    simple_retokenized = text
    return tensorizer.text_to_tensor(simple_retokenized, title=title, add_special_tokens=add_special_tokens)


def get_bert_tensorizer(pretrained_model_cfg, sequence_length, tokenizer=None):
    if not tokenizer:
        tokenizer = get_bert_tokenizer(pretrained_model_cfg, do_lower_case=True)
    return BertTensorizer(tokenizer, sequence_length, pad_to_max=False)


def get_bert_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    return BertTokenizer.from_pretrained(pretrained_cfg_name, do_lower_case=do_lower_case)


def find_answer_spans(ctx: ReaderPassage, answers: List, question, tensorizer, answers_token_ids):
    if ctx.has_answer:
        if ctx.passage_token_ids is None:
            ctx.passage_token_ids = text_to_tensor(ctx.passage_text, tensorizer, add_special_tokens=False)
        answer_spans = [
            _find_answer_positions(ctx.passage_token_ids, answers_token_ids[i]) for i in range(len(answers))
        ]
        # logger.info('!!! answer_spans = %s', answer_spans)

        # flatten spans list
        answer_spans = [item for sublist in answer_spans for item in sublist]
        answers_spans = list(filter(None, answer_spans))
        ctx.answers_spans = answers_spans

        if not answers_spans:
            logger.warning(
                "No answer found in passage id=%s text=%s, answers=%s, question=%s",
                ctx.id,
                ctx.passage_text,
                answers,
                question,
            )

            # logger.info('!!! ctx_tokens basic tok %s', btokenizer.tokenize(ctx.passage_text))
            # logger.info('!!! answers_tokens basic tok %s', btokenizer.tokenize(answers[0]))
        ctx.has_answer = bool(answers_spans)

    # logger.info('!!! find_answer_spans has answer = %s', ctx.has_answer)
    return ctx


def create_reader_sample_ids(
    sample: ReaderPassage,
    question: str,
    tensorizer,
    sep_tensor,
    is_train_set,
    use_tailing_sep=False,
):
    # question_and_title = tensorizer.text_to_tensor(sample.title, title=question, add_special_tokens=True)
    question_and_title = text_to_tensor(sample.title, tensorizer, add_special_tokens=True, title=question)
    if sample.passage_token_ids is None:
        # sample.passage_token_ids = tensorizer.text_to_tensor(sample.passage_text, add_special_tokens=False)
        sample.passage_token_ids = text_to_tensor(sample.passage_text, tensorizer, add_special_tokens=False)

    all_concatenated, shift = _concat_pair(
        question_and_title,
        sample.passage_token_ids,
        tailing_sep=sep_tensor if use_tailing_sep else None,
    )

    assert shift > 1
    if sample.has_answer and is_train_set:
        sample.answers_spans = [(span[0] + shift, span[1] + shift) for span in sample.answers_spans]
        # logger.info('!!! answers_spans %s', sample.answers_spans)
        # logger.info('!!! answers_spans decoded %s',[tensorizer.tokenizer.decode(all_concatenated[span[0]: span[1]].tolist()) for span in sample.answers_spans])

    sample.sequence_ids = all_concatenated
    sample.passage_offset = shift

    # logger.info('!!! sequence_ids %s', all_concatenated)
    # logger.info('!!! sequence_ids passage_offset %s', all_concatenated[shift:])
    # logger.info('!!! sequence_ids decoded %s', tensorizer.tokenizer.decode(all_concatenated.tolist()))

    return sample


def convert_sewon_json_file_to_reader2format(args):
    src = "/checkpoint/sewonmin/data/nq-dpr/nq-test-multi.json"
    is_train = False
    shard_id = args.shard_id
    shard_size = args.shard_size

    shard_start_idx = shard_id * shard_size
    shard_end_idx = shard_start_idx + shard_size
    # shard_end_idx = min(shard_start_idx + shard_size, len(data))
    logger.info("data range: %d-%d", shard_start_idx, shard_end_idx)

    data = []
    with open(src, "r") as f:
        logger.info("Reading file %s" % src)
        for i, line in enumerate(f):
            if i >= shard_start_idx and i < shard_end_idx:
                data.append(json.loads(line))
            # if len(data)==10:
            #    break

    logger.info("shard data size: %d", len(data))

    tensorizer = get_bert_tensorizer("bert-base-uncased", 350)
    results = []

    sep_tensor = tensorizer.get_pair_separator_ids()

    logger.info("!!! sep_tensor %s", sep_tensor)

    # data = data[shard_start_idx:shard_end_idx]
    # logger.info("data range: %d-%d", shard_start_idx, shard_end_idx)

    for id, sample in enumerate(data):
        q = sample["question"]

        # if q not in ['when was i do n \'t like mondays released', 'when was i don\'t like mondays released']:
        #    continue

        # logger.info('!!! q = %s', q)

        answers = sample["answers"]
        answers_token_ids = [text_to_tensor(a, tensorizer, add_special_tokens=False) for a in answers]
        reader_positive_passages = []

        for positive in sample["positives"]:
            title, passage_text_tokens, answers_detected = positive

            rp = ReaderPassage(None, text=" ".join(passage_text_tokens), title=title, has_answer=True)
            if is_train:
                ctx = find_answer_spans(rp, answers, q, tensorizer, answers_token_ids)
                if ctx.has_answer:
                    reader_positive_passages.append(rp)
            else:
                reader_positive_passages.append(rp)

        # logger.info('!!! rp = %s', reader_positive_passages[0])

        positive_passages = [
            create_reader_sample_ids(s, q, tensorizer, sep_tensor, is_train) for s in reader_positive_passages
        ]

        reader_negative_passages = []

        for negative in sample["negatives"]:
            title, passage_text_tokens, answers_detected = negative
            assert not answers_detected

            rp = ReaderPassage(None, text=" ".join(passage_text_tokens), title=title, has_answer=False)
            reader_negative_passages.append(rp)

        negative_passages = [
            create_reader_sample_ids(s, q, tensorizer, sep_tensor, True) for s in reader_negative_passages
        ]

        if is_train:
            result = ReaderSample(
                q,
                answers,
                positive_passages=positive_passages,
                negative_passages=negative_passages,
            )
        else:
            result = ReaderSample(q, answers, passages=positive_passages)

        results.append(result)

        if len(results) % 100 == 0:
            logger.info("Results %d", len(results))

    for i, r in enumerate(results):
        r.on_serialize()

    logger.info("Total Results %d", len(results))
    # exit(1)
    out_file = "/checkpoint/vladk/biencoder/nq_multi_data_sewon/nq-test-multi-reader2." + str(shard_id) + ".pkl"
    with open(out_file, mode="wb") as f:
        logger.info("Serialize %d results to %s", len(results), out_file)
        pickle.dump(results, f)


def convert_sewon_pkl_files_to_reader2format(args):
    dataset = "nq"
    type = "single-hybrid"
    type2 = "single-hybrid"

    # src_template ='/checkpoint/sewonmin/data/trivia/trivia-train*-passages-from-scott.pkl'
    src_template = "/checkpoint/sewonmin/data/new-new-dpr/{}-{}-{}.pkl"

    # src_template = '/checkpoint/sewonmin/data/new-new-dpr/{}-{}-{}.pkl'
    out_template = "/checkpoint/vladk/biencoder/{}_{}_data_sewon_from_pkl/{}-reader.pkl"

    set = "test"
    src = src_template.format(dataset, set, type)
    out_file = out_template.format(dataset, type2, set)
    convert_sewon_pkl_file_to_reader2format(src, out_file, is_train=False)

    set = "dev"
    src = src_template.format(dataset, set, type)
    out_file = out_template.format(dataset, type2, set)
    convert_sewon_pkl_file_to_reader2format(src, out_file, is_train=False)

    set = "train"
    # src = src_template.format(dataset, set, type)
    # out_file = out_template.format(dataset, type2, set)
    # convert_sewon_pkl_file_to_reader2format(src, out_file, is_train=True)

    src = "/checkpoint/sewonmin/data/new-new-dpr/{}-{}{}-{}.pkl"
    out_file = "/checkpoint/vladk/biencoder/{}_{}_data_sewon_from_pkl/{}-reader_{}.pkl"

    for i in range(5):
        convert_sewon_pkl_file_to_reader2format(
            src.format(dataset, set, i, type),
            out_file.format(dataset, type2, set, i),
            is_train=True,
        )


def convert_sewon_pkl_file_to_reader2format(src, out_file, is_train=True):
    with open(src, "rb") as f:
        features = pickle.load(f)

    data = features["features"]
    logger.info("data size: %d", len(data))
    tensorizer = get_bert_tensorizer("bert-base-uncased", 350)
    results = []

    sep_tensor = tensorizer.get_pair_separator_ids()
    logger.info("!!! sep_tensor %s", sep_tensor)

    for id, sample in enumerate(data):
        # logger.info('!!! sample %s', sample.keys())
        # logger.info('!!! sample id %s', sample['id'])
        # exit(1)

        question = sample["question"]
        answers = sample["answers"]
        positive_input_ids = sample["positive_input_ids"]
        positive_input_mask = sample["positive_input_mask"]
        positive_tokens = sample["positive_tokens"]
        positive_tok_to_orig_map = sample["positive_tok_to_orig_map"]

        positive_start_positions = sample["positive_start_positions"]
        positive_end_positions = sample["positive_end_positions"]

        # logger.info('!!! positive_input_ids %s', len(positive_input_ids))

        positive_reader_passages = []
        for i, positive_input_row in enumerate(positive_input_ids):
            mask_row = positive_input_mask[i]
            total_tokens = sum(mask_row)
            positive_seq_row = [id for k, id in enumerate(positive_input_row) if mask_row[k] == 1]
            assert total_tokens == len(positive_seq_row)

            positive_tokens_row = positive_tokens[i]
            positive_tok_to_orig_map_row = positive_tok_to_orig_map[i]
            seps = [k for k, t in enumerate(positive_tokens_row) if t == "[SEP]"]

            # logger.info('! positive_tokens_row %s', positive_tokens_row)
            # logger.info('! positive_tok_to_orig_map_row %s', positive_tok_to_orig_map_row)
            # logger.info('! seps %s', seps)

            # if positive_tok_to_orig_map_row[seps[-1] + 1] != 0:
            #    logger.info('!!! positive_tokens_row %s', positive_tokens_row)
            #    logger.info('!!! positive_tok_to_orig_map_row %s', positive_tok_to_orig_map_row)
            #    exit(1)
            # logger.info('!!! seps %d', len(seps))

            assert len(seps) == 2
            if positive_tokens_row[seps[-1]] != "[SEP]":
                logger.info("!!! positive_tokens_row ", positive_tokens_row)
                exit(1)

            rp = ReaderPassage(has_answer=True)
            rp.sequence_ids = torch.tensor(positive_seq_row, dtype=torch.long)
            rp.passage_offset = seps[-1] + 1

            if is_train:
                positive_start_positions_row = positive_start_positions[i]
                positive_end_positions_row = positive_end_positions[i]

                # logger.info('!!! positive_start_positions_row %s', positive_start_positions_row)

                answer_spans = [
                    (positive_start_positions_row[k], positive_end_positions_row[k])
                    for k in range(len(positive_start_positions_row))
                    if positive_start_positions_row[k] != 0
                ]
                assert all(span[0] != 0 and span[1] != 0 for span in answer_spans)
                rp.answers_spans = answer_spans

            positive_reader_passages.append(rp)

        negative_input_ids = sample["negative_input_ids"]
        negative_input_mask = sample["negative_input_mask"]

        # logger.info('!!! negative_input_ids %s', len(negative_input_ids))

        negative_reader_passages = []
        for i, negative_input_row in enumerate(negative_input_ids):
            mask_row = negative_input_mask[i]
            total_tokens = sum(mask_row)
            negative_seq_row = [id for k, id in enumerate(negative_input_row) if mask_row[k] == 1]
            assert total_tokens == len(negative_seq_row)

            rp = ReaderPassage(has_answer=False)
            rp.sequence_ids = torch.tensor(negative_seq_row, dtype=torch.long)
            negative_reader_passages.append(rp)

        if is_train:
            result = ReaderSample(
                question,
                answers,
                positive_passages=positive_reader_passages,
                negative_passages=negative_reader_passages,
            )
        else:
            result = ReaderSample(question, answers, passages=positive_reader_passages)

        """
        logger.info('!!! result q %s', question)
        logger.info('!!! result q %s', answers)
        logger.info('!!! result positive_passages %d', len(positive_reader_passages))
        logger.info('!!! result negative_passages %d', len(negative_reader_passages))

        for pp in positive_reader_passages:
            logger.info('!!! positive p: id=%s, title=%s, p_text=%s, aspans=%s, seq_ids=%s, score=%s, offset=%s', pp.id, pp.title, pp.passage_text, pp.answers_spans, pp.sequence_ids, pp.score, pp.passage_offset, )

        for np in negative_reader_passages:
            logger.info('!!! negative p: id=%s, title=%s, p_text=%s, aspans=%s, seq_ids=%s, score=%s, offset=%s', np.id, np.title, np.passage_text, np.answers_spans, np.sequence_ids, np.score, np.passage_offset, )
        exit(0)
        """
        results.append(result)

        if len(results) % 1000 == 0:
            logger.info("Results %d", len(results))

    for i, r in enumerate(results):
        r.on_serialize()

    logger.info("Total Results %d", len(results))

    with open(out_file, mode="wb") as f:
        logger.info("Serialize %d results to %s", len(results), out_file)
        pickle.dump(results, f)


def convert_biencoder_input_data(args):
    # path = '/private/home/scottyih/share/qa_encoder_training/nq-train.replace_with_search_results.json'
    # out = '/checkpoint/vladk/dpr_open_source/biencoder-nq-train.json'
    # path = '/checkpoint/vladk/dpr_open_source/triviaqa-train.json'
    # out = '/checkpoint/vladk/dpr_open_source/triviaqa-train_new.json'
    # path = "/private/home/scottyih/share/qa_encoder_training/squad1-dev.replace_with_search_results.json"
    # out = "/checkpoint/vladk/dpr_open_source/biencoder-squad1-dev.json"

    # path = "/private/home/scottyih/share/qa_encoder_training/webquestions-train.no_empty_positive.json"
    # path = "/private/home/scottyih/share/qa_encoder_training/webquestions-dev.no_empty_positive.json"
    # out = "/checkpoint/vladk/dpr_open_source/webquestions-train.json"
    # out = "/checkpoint/vladk/dpr_open_source/webquestions-dev.json"

    # path = "/private/home/scottyih/share/qa_encoder_training/curatedtrec-train.no_empty_positive.json"
    # out = "/checkpoint/vladk/dpr_open_source/curatedtrec-train.json"

    path = "/private/home/scottyih/share/qa_encoder_training/curatedtrec-dev.no_empty_positive.json"
    out = "/checkpoint/vladk/dpr_open_source/curatedtrec-dev.json"

    with open(path, "r", encoding="utf-8") as f:
        logger.info("Reading file %s" % path)
        data = json.load(f)
        logger.info("Aggregated data size: {}".format(len(data)))

    for i, sample in enumerate(data):
        # ctx+ & [ctx-] composition
        # logger.info('!! sampel keys %s', sample.keys())

        neg_ctxs = sample["distant_negatives"]
        hard_neg_ctxs = sample["negative_ctxs"]

        del sample["distant_gold_negatives"]
        del sample["negative_ctxs"]
        del sample["distant_negatives"]
        del sample["matched_ctxs"]

        sample["negative_ctxs"] = neg_ctxs
        sample["hard_negative_ctxs"] = hard_neg_ctxs

        if i % 1000 == 0:
            logger.info("!!! sample %d", i)

    with open(out, "w", encoding="utf-8") as f:
        logger.info("Writing file %s" % out)
        # data = json.dump(f)
        f.write(json.dumps(data, indent=4) + "\n")


def _get_gold_ctx_dict(
    file: str,
):  # -> Tuple[Dict[str, ReaderPassage], Dict[str, str]]:
    gold_passage_infos = {}  # question|question_tokens -> ReaderPassage (with title and gold ctx)

    # original NQ dataset has 2 forms of same question - original, and tokenized.
    # Tokenized form, even after concatenation is not fully consisted with the original form for some encoder tokenizers
    # Specifically, for the BERT tokenizer.
    # Depending of which form was used for retriever data training and results generation, it may be useful to convert
    # all questions to the canonical representation.
    original_questions = {}  # question from tokens -> original question (NQ only)

    with open(file, "r", encoding="utf-8") as f:
        logger.info("Reading file %s" % file)
        data = json.load(f)["data"]

    for sample in data:
        question = sample["question"].lower()
        question_from_tokens = sample["question_tokens"] if "question_tokens" in sample else question
        original_questions[question_from_tokens] = question
        title = sample["title"].lower()
        context = sample["context"]  # Note: This one is cased
        rp = ReaderPassage(sample["example_id"], text=context, title=title)
        if question in gold_passage_infos:
            logger.info("Duplicate question %s", question)
            rp_exist = gold_passage_infos[question]
            logger.info(
                "Duplicate question gold info: title new =%s | old title=%s",
                title,
                rp_exist.title,
            )
            logger.info("Duplicate question gold info: new ctx =%s ", context)
            logger.info("Duplicate question gold info: old ctx =%s ", rp_exist.passage_text)

        gold_passage_infos[question] = rp
        gold_passage_infos[question_from_tokens] = rp
    return gold_passage_infos, original_questions


def convert_sewon_json_file_to_prod_format(args):
    src = "/checkpoint/sewonmin/data/nq-dpr/nq-train-multi.json"
    out_file = "/checkpoint/vladk/dpr_for_prod/nq-train-multi.tsv"

    is_train = True
    shard_id = args.shard_id
    shard_size = args.shard_size

    shard_start_idx = shard_id * shard_size
    shard_end_idx = shard_start_idx + shard_size
    # shard_end_idx = min(shard_start_idx + shard_size, len(data))
    logger.info("data range: %d-%d", shard_start_idx, shard_end_idx)

    data = []

    with open(src, "r") as f:
        logger.info("Reading file %s" % src)
        for i, line in enumerate(f):
            if i >= shard_start_idx and i < shard_end_idx:
                data.append(json.loads(line))
            # if len(data)==10:
            #    break

    logger.info("shard data size: %d", len(data))

    gold_ctx_info, orig_questions = _get_gold_ctx_dict("/checkpoint/vladk/dpr_open_source/nq-train_gold_info.json")
    # out format:
    # title doc question 'list of answers'  list of answer start indices into doc Answerable Boolean(True/False) is_gold

    results = []

    for id, sample in enumerate(data):

        q = sample["question"]

        answers = sample["answers"]

        reader_positive_passages = []

        # add gold passage to results
        gold_rp = gold_ctx_info[q.lower()]
        gold_passage = gold_rp.passage_text
        answer_starts = []
        for answer in answers:
            if answer in gold_passage:
                answer_starts.append(gold_passage.index(answer))
            else:
                logger.info("! no answer in gold for answer=%s in ctx=%s", answer, gold_passage)

        if is_train and answer_starts:
            results.append([gold_rp.title, gold_passage, q, answers, answer_starts, True])

        for positive in sample["positives"][0:20]:
            title, passage_text_tokens, answers_detected = positive
            passage_text = " ".join(passage_text_tokens)
            if is_train and gold_passage.lower() == passage_text.lower():
                continue
            reader_positive_passages.append((title, passage_text))
            answer_starts = [a["answer_start"] for a in answers_detected]
            if answer_starts:
                results.append([title, passage_text, q, answers, answer_starts, True])
            else:
                logger.info("! no answer for answers=%s in ctx=%s", answers, passage_text)
                results.append([title, passage_text, q, answers, [], False])

        for negative in sample["negatives"][0:20]:
            title, passage_text_tokens, answers_detected = negative
            assert not answers_detected
            passage_text = " ".join(passage_text_tokens)
            reader_positive_passages.append((title, passage_text))
            results.append([title, passage_text, q, answers, [], False])

        if id % 100 == 0:
            logger.info("Results %d", id)

    with open(out_file, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        for r in results:
            writer.writerow(r)


def convert_music_to_qas_format():
    import jsonlines

    logger.info("! convert_music_to_qas_format")
    path = "/private/home/vladk/share/Results_Music_Question-Answer_Pair_-_Batch_*"

    train_out_file = "/private/home/vladk/data/music_QA/train_clean2_qas.csv"
    test_out_file = "/private/home/vladk/data/music_QA/test_clean2_qas.csv"

    train_out_jsonl_file = "/private/home/vladk/data/music_QA/train_clean2.jsonl"
    test_out_jsonl_file = "/private/home/vladk/data/music_QA/test_clean2.jsonl"

    data_files = glob.glob(path)

    logger.info("data_files %s", data_files)

    results = []

    long_ans = 0
    for file in data_files:
        with open(file, "rt", encoding="utf-8") as csvfile:  # , errors='surrogatepass'
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i == 0 or row[12] in [None, "", "\\u003C\\u003Cnot-applicable>>"] or row[13] in [None, ""]:
                    continue

                question = clean_music_data_string2(row[12])
                answer = clean_music_data_string2(row[13])

                """    
                question = row[12].strip('\"')
                answer = row[13].strip('\"')
                
                question = question.replace('"', '')
                answer = answer.replace('"', '')
                # remove unicode chars
                question = re.sub(r'\\u.{4}', '', question)
                answer = re.sub(r'\\u.{4}', '', answer)
                # remove tabs
                question = question.replace('\\t', '')
                answer = answer.replace('\\t', '')
                # remove backslashes
                question = question.replace('\\', '')
                answer = answer.replace('\\', '')
                # escape quote characters
                answer = answer.translate(str.maketrans({"'": r"\'", }))
                

                question = question.strip()
                answer = answer.strip()
                """

                if len(question) == 0:
                    continue
                if len(answer.split()) > 5:
                    long_ans += 1

                title = clean_music_data_string2(row[8])
                passage = clean_music_data_string2(row[10])

                results.append((question, answer, title, passage))

    logger.info("Combined results %d", len(results))
    logger.info("long_ans %d", long_ans)

    random.shuffle(results)

    train_ratio = int(len(results) * 0.95)
    train_results = results[0:train_ratio]
    test_results = results[train_ratio:]

    logger.info("train_results results %d", len(train_results))
    logger.info("test_results results %d", len(test_results))

    with open(train_out_file, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        for r in train_results:
            writer.writerow([r[0], [r[1]]])

    logger.info("Saved to %s", train_out_file)

    with open(test_out_file, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        for r in test_results:
            writer.writerow([r[0], [r[1]]])
    logger.info("Saved to %s", test_out_file)

    def _write_jsonl(file, data):
        with jsonlines.open(file, mode="w") as writer:  # encoding="utf-8", .encode('utf-8')
            for r in data:
                writer.write({"question": r[0], "context": r[3], "short_answers": [r[1]]})
        logger.info("Saved to %s", file)

    _write_jsonl(train_out_jsonl_file, train_results)
    _write_jsonl(test_out_jsonl_file, test_results)


import re


def clean_music_data_string(text: str):
    text = text.strip('"').replace('"', "")
    text = re.sub(r"\\u.{4}", "", text)
    # remove tabs
    return text.replace("\\t", "").replace("\\", "").strip()


def clean_music_data_string2(text: str):
    if not text:
        return ""
        # remove extra quotes
    text = text.replace('"', "")
    # render unicode chars
    text = text.rstrip("\\")
    text = re.sub(r"\\((?!u[a-f0-9A-F]{4}))", "", text)
    text = bytes(text, "utf-8").decode("unicode-escape")
    # remove tabs
    text = text.replace("\\t", "")
    return text.strip()


def load_passages(ctx_file: str, args) -> Dict[object, Tuple[str, str]]:
    docs = {}
    logger.info("Reading data from: %s", ctx_file)
    if args.new_chunks:
        with open(args.ctx_file, "rt", newline="") as fin:
            reader = csv.DictReader(fin, delimiter="\t")
            for row in reader:
                docs[row["id"]] = (row["text"], row["wikipedia_title"])

    if ctx_file.startswith(".gz"):
        with gzip.open(ctx_file) as tsvfile:
            reader = csv.reader(tsvfile, delimiter="\t")
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != "id":
                    docs[row[0]] = (row[1], row[2])
    else:
        with open(ctx_file) as tsvfile:
            reader = csv.reader(tsvfile, delimiter="\t")
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != "id":
                    docs[row[0]] = (row[1], row[2])
    return docs


def convert_music_csv_and_json_to_retriever_json():
    include_gold = True

    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)
    from dpr.data.qa_validation import has_answer

    gold_matched_file_pattern = "/private/home/vladk/data/music_QA/matched_train_shard_*.jsonl"
    # '/private/home/vladk/data/music_QA/matched_train_shard_*.jsonl'

    gold_matched_files = glob.glob(gold_matched_file_pattern)
    gold_matched_files.append("/private/home/vladk/data/music_QA/matched_test.jsonl")
    gold_passages = {}
    logger.info("gold_matched_files %s", gold_matched_files)

    for file in gold_matched_files:
        with jsonlines.open(file, mode="r") as jsonlReader:
            for jline in jsonlReader:
                q = jline["question"]
                ctx = jline["context"]
                id = jline["psg_id"]
                gold_passages[q] = (ctx, id)

    logger.info("Reading all passages...")
    all_passages = load_passages("/private/home/vladk/data/wikipedia/wiki_passages/psgs_w100.tsv")

    """
    # this snippet takes the original (i.e. biased) gold passage
    orig_csv_files = glob.glob('/private/home/vladk/share/Results_Music_Question-Answer_Pair_-_Batch_*')
    for file in orig_csv_files:
        with open(file) as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i == 0 or row[12] in [None, '', '\\u003C\\u003Cnot-applicable>>'] or row[13] in [None, '']:
                    continue
                title = clean_music_data_string(row[8])
                question = clean_music_data_string(row[12])
                passage = clean_music_data_string(row[10])
                if len(question) == 0:
                    continue
                gold_passages[question] = (passage, title)
    
    """

    logger.info("gold_passages results %d", len(gold_passages))

    def _process_results_file(results_file, out_file):
        gold_not_found = 0

        results_questions = {}

        with open(results_file, "r", encoding="utf-8") as f:
            samples = json.loads("".join(f.readlines()))
            retriever_format_samples = []
            for s in samples:
                passages = s["positive_ctxs"][0:100]
                question = s["question"]
                answers = s["answers"]

                results_questions[question] = 1

                positive_passages = []
                hard_negative_passages = []

                for passage in passages:
                    is_positive = has_answer(answers, passage["text"], tokenizer, "string")
                    if is_positive:
                        positive_passages.append(passage)
                    else:
                        hard_negative_passages.append(passage)

                if include_gold and question in gold_passages:
                    gold_passage, gold_id = gold_passages[question]
                    # get title
                    gold_info = all_passages[gold_id]
                    assert gold_passage == gold_info[0]
                    title = gold_info[1]

                    positive_passages = [
                        {
                            "title": title,
                            "text": gold_passage,
                        }
                    ] + positive_passages
                else:
                    gold_not_found += 1

                if len(positive_passages) == 0:
                    logger.info("positive_passages are empty q=%s", question)
                    continue
                retriever_format_samples.append(
                    {
                        "question": question,
                        "answers": answers,
                        "positive_ctxs": positive_passages[0:5],
                        "negative_ctxs": [],
                        "hard_negative_ctxs": hard_negative_passages[0:30],
                    }
                )

        logger.info("retriever_format_samples results %d", len(retriever_format_samples))
        logger.info("gold_not_found %d", gold_not_found)

        # save
        with open(out_file, "w") as writer:
            writer.write(json.dumps(retriever_format_samples, indent=4) + "\n")
        logger.info("saved to %s", out_file)
        return results_questions

    logger.info("Processing results file")

    # train_out_file = '/private/home/vladk/data/music_QA/train_qas.csv'
    # results_file = '/private/home/vladk/playground/bert-qa/code/lucene/out/music-test.top200.psg.b0.4.k0.9.psgs_w100.json'
    # out = '/private/home/vladk/data/music_QA/biencoder.music-test.gold_matched.json'

    # test_questions = _process_results_file(results_file, out)
    # logger.info('test_questions %s', len(test_questions))

    # check if all gold passage questions are in this set
    # gold_not_found = 0
    # for q in gold_passages.keys():
    #    if q not in test_questions:
    #        logger.info('no gold question found in results file: q={}'.format(q))
    #        gold_not_found += 1
    # logger.info('gold_not_found %d', gold_not_found)

    # check if all results questions are in the gold set
    """
    result_q_not_found = 0
    for q in test_questions.keys():
        if q not in gold_passages:
            logger.info('no result question found in gold set: q={}'.format(q))
            result_q_not_found += 1
    logger.info('result_q_not_found %d', result_q_not_found)
    """

    results_file = (
        "/private/home/vladk/playground/bert-qa/code/lucene/out/music-train.top200.psg.b0.4.k0.9.psgs_w100.json"
    )
    out = "/private/home/vladk/data/music_QA/biencoder.music-train.gold_matched.json"
    _process_results_file(results_file, out)


def convert_quora_to_dpr_json(args):
    file = "/checkpoint/vladk/datasets/quora/train.csv"

    out = "/checkpoint/vladk/datasets/quora/retriever_train.json"
    out2 = "/checkpoint/vladk/datasets/quora/retriever_dev.json"

    questions_results = {}
    all_questions = []
    with open(file) as ifile:
        reader = csv.reader(ifile, delimiter=",")
        for row in reader:
            if row[0] == "id":
                continue
            question1 = row[3]
            question2 = row[4]
            is_duplicate = row[5] == "1"

            if not question1 or not question2:
                logger.info("q1 %s", question1)
                logger.info("q2 %s", question2)
                logger.info("is_duplicate %s", is_duplicate)

                continue

            question1 = normalize_question(question1)
            question2 = normalize_question(question2)

            q1_res = questions_results.get(question1, ([], []))
            if is_duplicate:
                q1_res[0].append(question2)
            else:
                q1_res[1].append(question2)
            questions_results[question1] = q1_res

            q2_res = questions_results.get(question2, ([], []))
            if is_duplicate:
                q2_res[0].append(question1)
            else:
                q2_res[1].append(question1)
            questions_results[question2] = q2_res

            all_questions.append(question1)
            all_questions.append(question2)

    logger.info("Total questions %s", len(questions_results))
    logger.info("all_questions %s", len(all_questions))
    no_positive = 0
    no_negative = 0
    results = []
    for k, v in questions_results.items():

        if len(v[0]) == 0:
            no_positive += 1
            continue

        positive_ctxs = [{"text": q_text} for q_text in v[0]]
        hard_negative_ctxs = [{"text": q_text} for q_text in v[1]]

        negative_ctxs = set()

        if len(hard_negative_ctxs) == 0:
            no_negative += 1
            # sample 10 question from
            while len(negative_ctxs) < 10:
                i = random.randint(0, len(all_questions) - 1)
                if all_questions[i] != k and (all_questions[i] not in negative_ctxs):
                    negative_ctxs.add(all_questions[i])

            # logger.info('negative_ctxs %s', negative_ctxs)

        negative_ctxs = [{"text": t} for t in negative_ctxs]
        results.append(
            {
                "question": k,
                "positive_ctxs": positive_ctxs,
                "hard_negative_ctxs": hard_negative_ctxs,
                "negative_ctxs": negative_ctxs,
            }
        )

    logger.info("no_positive/no_negative %s / %s", no_positive, no_negative)
    logger.info("results %s", len(results))

    n = len(results)
    cut_off = n - 5000
    dev = results[cut_off:]
    logger.info("dev results %s", len(dev))
    train = results[0:cut_off]
    logger.info("train results %s", len(train))

    with open(out, "w") as writer:
        writer.write(json.dumps(train, indent=4) + "\n")
    logger.info("Saved to {}".format(out))

    with open(out2, "w") as writer:
        writer.write(json.dumps(dev, indent=4) + "\n")
    logger.info("Saved to {}".format(out2))


def get_iterator():
    for i in range(10):
        yield i


def get_iterator2():
    it = get_iterator()

    while True:
        r = next(it, None)
        if r is not None:
            yield r
        else:
            break


def test_apost():
    path = "/checkpoint/fabiopetroni/KILT/multitask_dpr_datasets/v0.20200817/nq-train-multikilt.json"
    with open(path, "r", encoding="utf-8") as f:
        logger.info("Reading file %s" % path)
        data = json.load(f)
        results = data
        logger.info("Aggregated data size: {}".format(len(results)))

    def calc(text, ctxs):
        return sum([ctx["text"].count(text) for ctx in ctxs])

    pos_ap1 = 0
    pos_ap2 = 0
    hn_ap1 = 0
    hn_ap2 = 0
    for r in results:
        pos = r["positive_ctxs"]
        neg = r["hard_negative_ctxs"]
        pos_ap1 += calc("’", pos)
        hn_ap1 += calc("’", neg)

        pos_ap2 += calc("'", pos)
        hn_ap2 += calc("'", neg)

    logger.info("Results: %s %s %s %s", pos_ap1, pos_ap2, hn_ap1, hn_ap2)


def augment_kilt_qa_ds_with_answers():
    cfg = [
        {
            "name": "nq_dev",
            "src": "/checkpoint/fabiopetroni/KILT/multitask_dpr_datasets/v0.20200817/nq-dev-multikilt.tsv",
            "qas": ["/private/home/scottyih/playground/bert-qa/data/nq-test.qa.csv"],
            "out": "/checkpoint/vladk/datasets/KILT/nq-dev-multikilt-answers.tsv",
        },
    ]

    replc_cfg = [
        ("ca n 't", "can't"),
        ("do n 't", "don't"),
        ("ai n 't", "ain't"),
        ("gon na", "gonna"),
        ("wan na", "wanna"),
        ("’", "'"),
        # ("'", "’"),
        ("i 'm", "i'm"),
        ("a ( n )", "a(n)"),
        ("he 's", "he's"),
        ("' robot '", "'robot'"),
        ("( but not a majority of )", "(but not a majority of)"),
        ("sunday . this is", "sunday. thisuis"),
        (
            "which foreign currency option is the ​ right but not the ​ obligation to buy foreign ​ currency",
            "which foreign currency option is the​ right bu not the​obligation to buy foreign​ currency",
        ),
        ("( fire safety )", "(fire safety)"),
        (
            "kitchen ​ brigade",
            "kitchen​ brigade",
        ),
        ("( or on painting )", "(or on painting)"),
        ("( both names )", "(both names)"),
        ("channels .", "channels."),
        ("wo n't", "won't"),
        ("gim me", "gimme"),
        ("sugar ( s )", "sugar(s)"),
        ("wo n 't", "won't"),
        ("campaign ' beti bachao-beti", "campaign 'beti bachao-beti"),
        ("$ 1", "$1"),
        ("babysitter 's", "babysitter's"),
        ("1 . ", "1."),
        ("ch3coo - ion", "ch3coo- ion"),
        ("' world teachers ' day '", "'world teachers' day'"),
        ("...", ". . ."),
    ]
    # trivia_test_kilt

    for ds in cfg:
        logger.info("Processing %s", ds["name"])
        qas = ds["qas"][0]
        answers_dict = {}

        with open(qas) as ifile:
            reader = csv.reader(ifile, delimiter="\t")
            for row in reader:
                question = row[0]
                answers = eval(row[1])
                answers_dict[question] = answers
                q_tokens = question.split()
                questions = set()
                questions.add(question)
                answers_dict[" ".join([t for t in q_tokens if t != " "])] = answers

                new_q = question
                for rep in replc_cfg:
                    new_q = new_q.replace(rep[0], rep[1])
                logger.warning("new question=%s", new_q)
                questions.add(new_q)
                answers_dict[new_q] = answers

                new_question = []
                for token in q_tokens:
                    if token.startswith("'"):
                        new_question[-1] = new_question[-1] + token
                    else:
                        new_question.append(token)
                new_question = " ".join(new_question)
                if new_question not in questions:
                    logger.warning("new question=%s", new_question)
                    answers_dict[new_question] = answers
                    answers_dict[new_question.replace("'", "’")] = answers

        answers_dict["he first line of http request message is called ____"] = [
            "A request line",
            "status line",
            "the status line",
            "A status line",
        ]
        answers_dict["in florida it is illegal to sell alcohol before 1 pm on any sunday. this is an example of"] = [
            "Blue laws"
        ]
        answers_dict["which foreign currency option is the​ right but not the​ obligation to buy foreign​ currency"] = [
            "foreign exchange option"
        ]
        answers_dict["the first line of http request message is called ____"] = [
            "A request line",
            "status line",
            "the status line",
            "A status line",
        ]
        answers_dict["who sang i'm gonna run away from you"] = ["Tami Lynn"]
        answers_dict["when was the r10+20 summit in rio de janeiro held"] = [
            "13 to 22 June 2012",
            "June 2012",
        ]
        answers_dict["which financial statement involves all aspects of the accounting​ equation"] = [
            "The balance sheet",
            "balance sheet",
        ]
        answers_dict["1. what was the precursor to the present day internet"] = ["the ARPANET project"]

        src = ds["src"]
        results = []
        with open(src) as ifile:
            reader = csv.reader(ifile, delimiter="\t")
            for row in reader:
                question = row[0]
                answers = []
                if question not in answers_dict:
                    q_tokens = question.split()
                    new_question = []

                    for token in q_tokens:
                        if token[0] == "(" and token[-1] == ")":
                            new_question.append(token[0])
                            new_question.append(token[1:-1])
                            new_question.append(token[-1])
                        else:
                            new_question.append(token)

                    new_question = " ".join(new_question)
                    if new_question not in answers_dict:
                        logger.warning(
                            "No answer for question=%s, new q=%s",
                            question,
                            new_question,
                        )
                    if new_question in answers_dict:
                        question = new_question
                    else:
                        new_question = " ".join(t for t in q_tokens if t != " ")
                    if new_question in answers_dict:
                        question = new_question
                if question in answers_dict:
                    answers = answers_dict[question]
                results.append((question, [a.lower() for a in answers]))
        outfile = ds["out"]
        with open(outfile, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter="\t")
            for item in results:
                writer.writerow(item)


def fix_music_results(file, out_file):

    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    with open(file, "r", encoding="utf-8") as f:
        logger.info("Reading file %s" % file)
        data = json.load(f)
        results = data
        logger.info("Aggregated data size: {}".format(len(results)))

    for result in data:
        answers = result["answers"]
        answers = [a if a != "" else " " for a in answers]
        answers = "".join(answers)
        answers = answers.split(",")
        # answers = [answers]
        answers = [a.strip() for a in answers]
        # logger.info("a=%s", answers)
        result["answers"] = answers
        for ctx in result["ctxs"]:
            text = ctx["text"]
            if text is None:  # cannot find the document for some reason
                logger.warning("no doc in db")
                continue

            answer_found = has_answer(answers, text, tokenizer, "string")
            ctx["has_answer"] = answer_found

    with open(out_file, "w") as writer:
        writer.write(json.dumps(data, indent=4) + "\n")
    logger.info("Saved results * scores  to %s", out_file)


def convert_nq_questions_into_csv(file, gold_file, outfile):
    with open(gold_file, "r", encoding="utf-8") as reader:
        json_data = reader.read()
    gold_data = json.loads(json_data)["data"]
    q_map = {}
    for item in gold_data:
        q_map[item["question_tokens"]] = item["question"]

    if file.endswith(".json"):
        with open(file, "r", encoding="utf-8") as reader:
            json_data = reader.read()
        data = json.loads(json_data)
        csv_data = []
        id = 0
        for item in data:
            q = item["question"]
            if q in q_map:
                q = q_map[q]
            id += 1
            csv_data.append((format(id, "06d"), q))
    elif file.endswith(".csv"):
        with open(file) as tsvfile:
            reader = csv.reader(tsvfile, delimiter="\t")
            # file format: question answers
            id = 0
            csv_data = []
            for row in reader:
                q = row[0]
                if q in q_map:
                    q = q_map[q]
                id += 1
                csv_data.append((format(id, "06d"), q))

    with open(outfile, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter="|")

        for x in csv_data:
            id, q = x[0], x[1]
            writer.writerow([id, q])


def create_asr_inference_files(audio_dir, tsv_file, out_tsv, out_ltr):

    tsv_lines = []
    tsv_lines.append(audio_dir)
    ltr_lines = []

    with open(tsv_file, "r") as csv_file:
        reader = csv.reader(csv_file, delimiter="|")
        for row in reader:
            a_file = "aud_dn_{}.wav".format(row[0])
            file_full = os.path.join(audio_dir, a_file)
            assert os.path.exists(file_full)
            tsv_lines.append(a_file)

            ltr = list(row[1].strip().upper().replace(" ", "|"))
            ltr = " ".join(ltr) + " |"
            ltr_lines.append(ltr)

    with open(out_tsv, "w") as out_tsv:
        out_tsv.write("%s\n" % tsv_lines[0])
        for line in tsv_lines[1:]:
            out_tsv.write("%s\t10000\n" % line)

    with open(out_ltr, "w") as out_ltr:
        for line in ltr_lines:
            out_ltr.write("%s\n" % line)


def remove_old_index_files(root_dir: str):
    from os import listdir
    from os.path import isfile
    from pathlib import Path

    shard_dirs = [os.path.join(root_dir, d) for d in listdir(root_dir) if not isfile(d)]

    def del_old(files):
        if len(files) < 2:
            return
        to_del = [str(f) for f in files[:-1]]
        logger.info(
            "Del files %s",
        )
        [os.remove(f) for f in to_del]

    for shard_root in shard_dirs:
        logger.info("shard_root  %s", shard_root)
        shard_files = sorted(Path(shard_root).iterdir(), key=os.path.getmtime)
        index_files = [p for p in shard_files if "index." in str(p)]
        meta_files = [p for p in shard_files if "meta." in str(p)]
        buffer_files = [p for p in shard_files if "buffer." in str(p)]

        del_old(index_files)
        del_old(meta_files)
        del_old(buffer_files)

        # logger.info('shard_files %s', shard_files)


def convert_stl_to_dpr_cp(cp_file, out_file):
    model = torch.load(cp_file)

    dpr_model_dict = {}
    dpr_dict = {
        "model_dict": dpr_model_dict,
        "optimizer_dict": None,
        "scheduler_dict": None,
        "offset": 0,
        "epoch": 0,
        "encoder_params": None,
    }

    for k, v in model.items():  # ["state_dict"]
        if "query_encoder.transformer" in k:
            new_k = "question_model" + k[len("query_encoder.transformer") :]
            dpr_model_dict[new_k] = v
        if "context_encoder.transformer" in k:
            new_k = "ctx_model" + k[len("context_encoder.transformer") :]
            dpr_model_dict[new_k] = v
    torch.save(dpr_dict, out_file)
    logger.info("Saved to %s", out_file)


def add_answers_to_kilt_retriever_files(out_file: str, kilt_gold_file: str):

    question_to_answers = []
    with jsonlines.open(kilt_gold_file, "r") as reader:
        kilt_data = list(reader)

        logger.info("!!! kilt_data %d", len(kilt_data))

        for ex in kilt_data:
            answers = set()
            q = ex["input"]
            for a in ex["output"]:
                if "answer" in a:
                    answers.add(a["answer"])
            question_to_answers.append((q, list(answers)))

    logger.info("!!! question_to_answers %d", len(question_to_answers))

    with open(out_file, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        for sample in question_to_answers:
            r = [sample[0], str(sample[1])]
            writer.writerow(r)

    logger.info("Saved to %s", out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_size", type=int, default=-1, help="Data size per shard")
    parser.add_argument("--shard_id", type=int, default=-1, help="shard id (0-based)")
    args = parser.parse_args()

    # find_phi()
    # update_is_wiki_in_meta()

    """
    create_cc_net_eval_passages(
        "/checkpoint/vladk/biencoder/cc-net2019-results-urbase/",
        [
            "nq_dev.json",
            "zeroshot_dev.json",
            "hotpot_dev.json",
            "trex_dev.json",
            "trivia_dev.json",
        ],
        "/checkpoint/piktus/sphere/fid_data/BM25/cc_net/",
        [
            "triviaqa-dev-kilt.jsonl",
            "trex-dev-kilt.jsonl",
            "structured_zeroshot-dev-kilt.jsonl",
            "nq-dev-kilt.jsonl",
            "hotpotqa-dev-kilt.jsonl",
        ],
        "/checkpoint/vladk/biencoder/cc-net2019-results-urbase/eval_passages.tsv",
    )
    """

    """

    dense_eval_results_to_dpr_train_files(
        "/checkpoint/vladk/biencoder/val_ur_ccnet2_sp_ccnet_finetune_paq_all_lr2e5_40e/trex_dev.json",
        "/checkpoint/fabiopetroni/KILT/multitask_dpr_datasets/v0.20200817/trex-dev-multikilt.json",
        None,
        # is_slot_task=True,
    )

    dense_eval_results_to_dpr_train_files(
        "/checkpoint/vladk/biencoder/val_ur_ccnet2_sp_ccnet_finetune_paq_all_lr2e5_40e/zeroshot_dev.json",
        "/checkpoint/fabiopetroni/KILT/multitask_dpr_datasets/v0.20200817/zeroshot-dev-multikilt.json",
        None,
        # is_slot_task=True,
    )

    dense_eval_results_to_dpr_train_files(
        "/checkpoint/vladk/biencoder/val_ur_ccnet2_sp_ccnet_finetune_paq_all_lr2e5_40e/nq_dev.json",
        "/checkpoint/fabiopetroni/KILT/multitask_dpr_datasets/v0.20200817/nq-dev-multikilt.json",
        None,
    )

    dense_eval_results_to_dpr_train_files(
        "/checkpoint/vladk/biencoder/val_ur_ccnet2_sp_ccnet_finetune_paq_all_lr2e5_40e/trivia_dev.json",
        "/checkpoint/fabiopetroni/KILT/multitask_dpr_datasets/v0.20200817/triviaqa-dev-multikilt.json",
        None,
    )

    dense_eval_results_to_dpr_train_files(
        "/checkpoint/vladk/biencoder/val_ur_ccnet2_sp_ccnet_finetune_paq_all_lr2e5_40e/hotpot_dev.json",
        "/checkpoint/fabiopetroni/KILT/multitask_dpr_datasets/v0.20200817/hotpotqa-dev-multikilt.json",
        None,
    )

    
    add_answers_to_kilt_retriever_files(
       "/checkpoint/vladk/dpr_open_source/ur/nq-dev-kilt.tsv",
       "/checkpoint/fabiopetroni/KILT/datasets/nq-dev-kilt.jsonl",
    )
    
    add_answers_to_kilt_retriever_files(
        "/checkpoint/vladk/dpr_open_source/ur/nq-train-kilt.tsv",
        "/checkpoint/fabiopetroni/KILT/datasets/nq-train-kilt.jsonl",
    )
    

    add_answers_to_kilt_retriever_files(
        "/checkpoint/vladk/dpr_open_source/ur/triviaqa-dev-kilt.tsv",
        "/checkpoint/fabiopetroni/KILT/datasets/triviaqa-dev-kilt.jsonl",
    )

    add_answers_to_kilt_retriever_files(
        "/checkpoint/vladk/dpr_open_source/ur/triviaqa-train-kilt.tsv",
        "/checkpoint/fabiopetroni/KILT/datasets/triviaqa-train-kilt.jsonl",
    )

    add_answers_to_kilt_retriever_files(
        "/checkpoint/vladk/dpr_open_source/ur/hotpotqa-dev-kilt.tsv",
        "/checkpoint/fabiopetroni/KILT/datasets/hotpotqa-dev-kilt.jsonl",
    )

    add_answers_to_kilt_retriever_files(
        "/checkpoint/vladk/dpr_open_source/ur/hotpotqa-train-kilt.tsv",
        "/checkpoint/fabiopetroni/KILT/datasets/hotpotqa-train-kilt.jsonl",
    )

    add_answers_to_kilt_retriever_files(
        "/checkpoint/vladk/dpr_open_source/ur/structured_zeroshot-train-kilt.tsv",
        "/checkpoint/fabiopetroni/KILT/datasets/structured_zeroshot-train-kilt.jsonl",
    )

    add_answers_to_kilt_retriever_files(
        "/checkpoint/vladk/dpr_open_source/ur/structured_zeroshot-dev-kilt.tsv",
        "/checkpoint/fabiopetroni/KILT/datasets/structured_zeroshot-dev-kilt.jsonl",
    )

    add_answers_to_kilt_retriever_files(
        "/checkpoint/vladk/dpr_open_source/ur/trex-dev-kilt.tsv",
        "/checkpoint/fabiopetroni/KILT/datasets/trex-dev-kilt.jsonl",
    )

    add_answers_to_kilt_retriever_files(
        "/checkpoint/vladk/dpr_open_source/ur/trex-train-kilt.tsv",
        "/checkpoint/fabiopetroni/KILT/datasets/trex-train-kilt.jsonl",
    )
    """

    # remove_old_index_files('/checkpoint/vladk/faiss-indexes/hnswsq_cc-news/')

    # convert_stl_to_dpr_cp(
    #    "/checkpoint/barlaso/paq-bert-base-neg1.ckpt",
    #    "/checkpoint/vladk/biencoder/paq/paq_base_hn.cp",
    # )

    """
    convert_stl_to_dpr_cp(
        "/checkpoint/barlaso/paq-bert-large-neg1.ckpt",
        "/checkpoint/vladk/biencoder/paq/paq_large_hn.cp",
    )
    
    convert_stl_to_dpr_cp(
        "/checkpoint/barlaso/hydra_outputs/PAQ_BERT_LARGE/2021-05-30-043136/0/lightning_logs/version_1801/checkpoints/checkpoint_best-v1.ckpt",
        "/checkpoint/vladk/biencoder/paq/paq_large_biencoder.cp",
    )
    
    convert_stl_to_dpr_cp(
        "/checkpoint/barlaso/hydra_outputs/PAQ/2021-04-27-152340/0/lightning_logs/version_40710591/checkpoints/checkpoint_best-v1.ckpt",
        "/checkpoint/vladk/biencoder/paq/paq_biencoder.cp",
    )
    """

    """
    split = "train"

    create_char_mlm_data_from_retriever_results(
        "/checkpoint/vladk/dpr_open_source/nq_single_{}_dense_results.json".format(split),
        176,
        "/checkpoint/vladk/speechqa/data/{}/base_retriever_results_chars_{}.json".format(split, split),
    )
        
    """

    """
    convert_nq_questions_into_csv(
        "/private/home/scottyih/playground/bert-qa/data/nq-train.qa.csv",
        "/checkpoint/vladk/dpr_open_source/nq-train_gold_info.json",
        "/checkpoint/vladk/speechqa/data/train/nq_all_train_questions.csv",
    )
    """
    """
    convert_nq_questions_into_csv(
        "/private/home/scottyih/playground/bert-qa/data/nq-dev.qa.csv",
        "/checkpoint/vladk/dpr_open_source/nq-dev_gold_info.json",
        "/checkpoint/vladk/speechqa/data    /dev/nq_all_dev_questions.csv",
    )
    
    
    split = "train"
    create_mlm_char_data(
        "/checkpoint/vladk/dpr_open_source/biencoder-nq-{}.json".format(split),
        176,
        "/checkpoint/vladk/speechqa/data/{split}/mlm-char_rep.txt".format(split=split),
    )
    
    split = "test"

    create_mlm_data(
        "/checkpoint/vladk/dpr_open_source/biencoder-nq-{}.json".format(split),
        "/checkpoint/kushall/data/speechqa/{split}.tsv".format(split=split),
        "/checkpoint/kushall/data/speechqa/hubert_clustering/layer6_km100/{split}_deduped_0_1.km".format(split=split),
        176,
        "/checkpoint/vladk/speechqa/data/{split}/mlm-text-only.txt".format(split=split),
        text_only=True
    )
    
    
    split = "train"
    create_asr_inference_files(
        "/checkpoint/vladk/speechqa/data/{}/all_wav/denoised_16k/".format(split),
        "/checkpoint/vladk/speechqa/data/{}/nq_all_{}_questions.csv".format(split, split),
        "/checkpoint/vladk/speechqa/data/{}/all_wav/{}.tsv".format(split, split),
        "/checkpoint/vladk/speechqa/data/{}/all_wav/{}.ltr".format(split, split),
    )

    
    create_asr_inference_files(
        "/checkpoint/vladk/speechqa/data/{}/denoised/".format(split),
        "/checkpoint/vladk/speechqa/data/{}/nq_{}_questions.csv".format(split, split),
        "/checkpoint/vladk/speechqa/data/{}/{}.tsv".format(split, split),
        "/checkpoint/vladk/speechqa/data/{}/{}.ltr".format(split, split),
    )
    """

    """
    create_asr_inference_files(
        "/checkpoint/vladk/speechqa/data/train/denoised/",
        "/checkpoint/vladk/speechqa/data/train/nq_train_questions.csv",
        "/checkpoint/vladk/speechqa/data/train/train.tsv",
        "/checkpoint/vladk/speechqa/data/train/train.ltr",
    )
    
    convert_nq_questions_into_csv(
        "/checkpoint/vladk/dpr_open_source/biencoder-nq-train.json",
        "/checkpoint/vladk/dpr_open_source/nq-train_gold_info.json",
        "/checkpoint/vladk/speechqa/data/nq_train_questions.csv",
    )
    
    convert_nq_questions_into_csv(
        "/checkpoint/vladk/dpr_open_source/biencoder-nq-dev.json",
        "/checkpoint/vladk/dpr_open_source/nq-dev_gold_info.json",
        "/checkpoint/vladk/speechqa/data/nq_dev_questions.csv",
    )    
    
    convert_nq_questions_into_csv(
        "/checkpoint/vladk/dpr_open_source/nq-test.qa.csv",
        "/checkpoint/vladk/dpr_open_source/nq-test_gold_info.json",
        "/checkpoint/vladk/speechqa/data/nq_test_questions.csv",
    )
    """

    # eval_dpr_retriever_out_file(
    #    "/private/home/xilun/hybridqa/nq/fid_inputs/hybrid_wikipedia_freebase_wikidata/wikipedia_textl_tables_90_plus_freebase_wikidata_elq_joint_directed_2hop_10/nq_dev.json"
    # )

    # convert_sewon_json_file_to_reader2format(args)
    # convert_sewon_pkl_files_to_reader2format(args)
    # convert_biencoder_input_data(args)
    """

    # convert_sewon_json_file_to_prod_format(args)
    # convert_music_to_qas_format()
    # convert_music_csv_and_json_to_retriever_json()

    # convert_quora_to_dpr_json(args)
    # augment_kilt_qa_ds_with_answers()
    # test_apost()

    fix_music_results(
        "/checkpoint/vladk/biencoder/validate_music_qa_multiset_async_funetune_4//music_test.json",
        "/checkpoint/vladk/biencoder/validate_music_qa_multiset_async_funetune_4//music_test_fixed.json",
    )

    fix_music_results(
        "/checkpoint/vladk/biencoder/validate_music_qa_multiset_async_funetune_4//music_train.json",
        "/checkpoint/vladk/biencoder/validate_music_qa_multiset_async_funetune_4//music_train_fixed.json",
    )

    fix_music_results(
        "/checkpoint/vladk/biencoder/validate_music_qa_multiset_async_funetune_4//music_dev.json",
        "/checkpoint/vladk/biencoder/validate_music_qa_multiset_async_funetune_4//music_dev_fixed.json",
    )
    """

    add_answers_to_kilt_retriever_files(
        "/checkpoint/vladk/dpr_open_source/ur/trex-dev-kilt.tsv",
        "/checkpoint/fabiopetroni/KILT/datasets/trex-dev-kilt.jsonl",
    )
