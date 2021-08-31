import argparse
import csv
import glob
import gzip
import json
import logging
import os

import jsonlines

from dpr.data.speech_data import _get_id_to_audio_file_map_paq
from dpr.utils.data_utils import read_data_from_json_files

logger = logging.getLogger()

logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


def create_mlm_data(
    json_file: str,
    wav_tsv_file: str,
    km_file: str,
    max_len: int,
    out_file: str,
    token_prefix: str = "w2v",
    max_positives: int = 5,
    q2q: bool = False,
    audio_only: bool = False,
):
    audio_file_prefix = "aud_dn_"
    orig_to_manifest_id_map = {}
    prefix_len = len(audio_file_prefix)
    suffix_len = len(".wav")
    with open(os.path.join(wav_tsv_file), "r") as fp:  # read tsv file
        lines = fp.read().split("\n")
        root = lines.pop(0).strip()
        manifest_id = 0
        for line in lines:
            if len(line) == 0:
                continue
            file = line.split("\t")[0]
            orig_id = int(file[prefix_len:-suffix_len])
            assert orig_id not in orig_to_manifest_id_map
            orig_to_manifest_id_map[orig_id] = manifest_id
            manifest_id += 1
        logging.info("last manifest_id %d", manifest_id)

    logging.info("orig_to_manifest_id_map %d", len(orig_to_manifest_id_map))

    data = read_data_from_json_files([json_file])
    # filter those without positive ctx
    data = [r for r in data if len(r["positive_ctxs"]) > 0]
    logger.info("Total cleaned data size: {}".format(len(data)))

    kms = []
    with open(km_file, "r") as ifile:
        reader = csv.reader(ifile, delimiter="\t")
        for i, row in enumerate(reader):
            assert len(row) == 1
            km_query_tokens = row[0].split()
            if len(km_query_tokens) > max_len:
                logging.info("!!! long km %s", len(km_query_tokens))
                km_query_tokens = km_query_tokens[0:max_len]
            kms.append(km_query_tokens)

    result = []  # just text
    assert len(data) == len(kms), "kms size={}".format(len(kms))
    for orig_id, sample in enumerate(data):
        positive_ctxs = sample["positive_ctxs"][0:max_positives]
        manifest_id = orig_to_manifest_id_map[orig_id + 1]
        # logging.info("orig_id=%d manifest_id=%d", orig_id, manifest_id)
        km = kms[manifest_id]
        km_with_prefixes = ["[" + token_prefix + str(t) + "]" for t in km]
        if q2q:
            if audio_only:
                sample_line = " ".join(km_with_prefixes) + "\n"
            else:
                sample_line = " ".join(km_with_prefixes) + "[SEP]" + sample["question"] + "\n"
            result.append(sample_line)
        else:
            for ctx in positive_ctxs:
                sample_line = " ".join(km_with_prefixes) + "[SEP]" + ctx["title"] + ". " + ctx["text"] + "\n"
                result.append(sample_line)

    # result=result[0:100]
    with open(out_file, "w") as ofile:
        ofile.writelines(result)
    logging.info("Results saved to %s", out_file)


def create_mlm_paq_data(
    jsonl_file: str,
    manifest_txt_file: str,
    km_file: str,
    wav_root_dir: str,
    km_manifest_tsv_file: str,
    sample_start: int,
    samples_end: int,
    max_len: int,
    out_file: str,
    token_prefix: str = "w2v",
    max_positives: int = 5,
    text_only: bool = False,
    question_to_questions: bool = False,
):
    audio_file_prefix = "aud_dn_"
    data = []
    logger.info("Reading file %s" % jsonl_file)
    with jsonlines.open(jsonl_file, mode="r") as jsonl_reader:
        for i, r in enumerate(jsonl_reader):
            if i < sample_start:
                continue
            if i >= samples_end:
                break
            data.append(r)
    logger.info("Aggregated data size: {}".format(len(data)))

    questions = set()
    for json_sample in data:
        q = json_sample["question"]
        questions.add(q)
    logger.info("questions size: {}".format(len(questions)))
    q_to_audio_file_map = _get_id_to_audio_file_map_paq(
        questions,
        wav_root_dir,
        audio_file_prefix,
        manifest_txt_file,
    )
    logger.info("q_to_audio_file_map size: {}".format(len(q_to_audio_file_map)))

    audio_file_to_q_map = {}

    for q, a_file in q_to_audio_file_map.items():
        audio_file_to_q_map[a_file] = q

    logging.info("audio_file_to_q_map %d", len(audio_file_to_q_map))

    audio_file_to_km_map = {}

    with open(km_manifest_tsv_file, "r") as fp:  # read tsv file
        lines = fp.read().split("\n")
        file_order = 0
        for line in lines:
            if len(line) == 0:
                continue
            file = line.split("\t")[0]
            abs_file_path = os.path.join(wav_root_dir, file)
            audio_file_to_km_map[abs_file_path] = file_order
            file_order += 1
    logging.info("audio_file_to_km_map %d", len(audio_file_to_km_map))

    kms = []
    long_km_files = 0
    with open(km_file, "r") as ifile:
        reader = csv.reader(ifile, delimiter="\t")
        for i, row in enumerate(reader):
            assert len(row) == 1
            km_query_tokens = row[0].split()
            if len(km_query_tokens) > max_len:
                long_km_files += 1
                km_query_tokens = km_query_tokens[0:max_len]
            kms.append(km_query_tokens)

    logging.info("kms %d", len(kms))
    logging.info("long_km_files %d", long_km_files)

    result = []  # just text

    for orig_id, sample in enumerate(data):
        q = sample["question"]
        q_audio_file = q_to_audio_file_map[q]
        if q_audio_file not in audio_file_to_km_map:
            continue
        km_id = audio_file_to_km_map[q_audio_file]
        km = kms[km_id]
        km_with_prefixes = ["[" + token_prefix + str(t) + "]" for t in km]

        if question_to_questions:
            sample_line = " ".join(km_with_prefixes) + "[SEP]" + q + "\n"
            result.append(sample_line)
        else:
            positive_ctxs = sample["positive_ctxs"][0:max_positives]
            # logging.info("orig_id=%d manifest_id=%d", orig_id, manifest_id)

            for ctx in positive_ctxs:
                if text_only:
                    sample_line = ctx["title"] + ". " + ctx["text"] + "\n"
                else:
                    sample_line = " ".join(km_with_prefixes) + "[SEP]" + ctx["title"] + ". " + ctx["text"] + "\n"
                result.append(sample_line)

    logging.info("Results size %d", len(result))

    with open(out_file, "w") as ofile:
        ofile.writelines(result)
    logging.info("Results saved to %s", out_file)


def create_mlm_char_data(
    json_file: str,
    max_len: int,
    out_file: str,
    token_prefix: str = "ct",
    max_positives: int = 5,
    text_only: bool = False,
):
    import string

    # tokens = list(string.ascii_lowercase) #+ ["'", '"', "-"] + list("0123456789")
    tokens = list(string.printable)
    # ["\u200b", "°", "£"]

    token_mapping = {t: str(id) for id, t in enumerate(tokens)}

    logger.info("Char token_mapping len %d", len(token_mapping))
    data = read_data_from_json_files([json_file])
    # filter those without positive ctx
    data = [r for r in data if len(r["positive_ctxs"]) > 0]
    logger.info("Total cleaned data size: {}".format(len(data)))
    result = []  # just text
    for orig_id, sample in enumerate(data):
        positive_ctxs = sample["positive_ctxs"][0:max_positives]
        q = sample["question"]
        q_chars = [c for c in list(q.lower()) if c != " "]
        if len(q_chars) > max_len:
            logging.info("long km %s", len(q_chars))
            q_chars = q_chars[0:max_len]

        q_chars_ids = [token_mapping[c] for c in q_chars if c in token_mapping]

        q_chars_tokens = ["[" + token_prefix + c + "]" for c in q_chars_ids]
        for ctx in positive_ctxs:
            if text_only:
                sample_line = ctx["title"] + ". " + ctx["text"] + "\n"
            else:
                sample_line = " ".join(q_chars_tokens) + "[SEP]" + ctx["title"] + ". " + ctx["text"] + "\n"
            result.append(sample_line)

    with open(out_file, "w") as ofile:
        ofile.writelines(result)
    logging.info("Results saved to %s", out_file)


def create_mlm_data_from_retriever_results(
    json_file: str,
    wav_tsv_file: str,
    km_file: str,
    max_len: int,
    out_file: str,
    token_prefix: str = "w2v",
):
    audio_file_prefix = "aud_dn_"
    orig_to_manifest_id_map = {}
    prefix_len = len(audio_file_prefix)
    suffix_len = len(".wav")
    with open(os.path.join(wav_tsv_file), "r") as fp:  # read tsv file
        lines = fp.read().split("\n")
        root = lines.pop(0).strip()
        manifest_id = 0
        for line in lines:
            if len(line) == 0:
                continue
            file = line.split("\t")[0]
            orig_id = int(file[prefix_len:-suffix_len])
            assert orig_id not in orig_to_manifest_id_map
            orig_to_manifest_id_map[orig_id] = manifest_id
            manifest_id += 1
        logging.info("last manifest_id %d", manifest_id)

    logging.info("orig_to_manifest_id_map %d", len(orig_to_manifest_id_map))

    data = read_data_from_json_files([json_file])
    # filter those without positive ctx

    kms = []
    with open(km_file, "r") as ifile:
        reader = csv.reader(ifile, delimiter="\t")
        for i, row in enumerate(reader):
            assert len(row) == 1
            km_query_tokens = row[0].split()
            if len(km_query_tokens) > max_len:
                logging.info("!!! long km %s", len(km_query_tokens))
                km_query_tokens = km_query_tokens[0:max_len]
            kms.append(km_query_tokens)

    assert len(data) == len(kms)

    for orig_id, sample in enumerate(data):
        orig_q = sample["question"]
        manifest_id = orig_to_manifest_id_map[orig_id + 1]
        # logging.info("orig_id=%d manifest_id=%d", orig_id, manifest_id)

        km = kms[manifest_id]
        km_with_prefixes = ["[" + token_prefix + str(t) + "]" for t in km]
        new_q = " ".join(km_with_prefixes)

        sample["query_text"] = orig_q
        sample["question"] = new_q

    # result=result[0:100]

    with open(out_file, "w") as writer:
        writer.write(json.dumps(data, indent=4) + "\n")
    logging.info("Results saved to %s", out_file)


def create_char_mlm_data_from_retriever_results(
    json_file: str,
    max_len: int,
    out_file: str,
    token_prefix: str = "ct",
):
    import string

    tokens = list(string.printable)
    token_mapping = {t: str(id) for id, t in enumerate(tokens)}
    logger.info("Char token_mapping len %d", len(token_mapping))

    data = read_data_from_json_files([json_file])
    logger.info("Total data size: {}".format(len(data)))

    for orig_id, sample in enumerate(data):
        q = sample["question"]
        q_chars = [c for c in list(q.lower()) if c != " "]
        if len(q_chars) > max_len:
            logging.info("!!! long km %s", len(q_chars))
            q_chars = q_chars[0:max_len]
        q_chars_ids = [token_mapping[c] for c in q_chars if c in token_mapping]
        q_chars_tokens = ["[" + token_prefix + c + "]" for c in q_chars_ids]
        new_q = " ".join(q_chars_tokens)
        sample["query_text"] = new_q
        sample["question"] = new_q

    with open(out_file, "w") as writer:
        writer.write(json.dumps(data, indent=4) + "\n")
    logging.info("Results saved to %s", out_file)


def main():

    split = "dev"

    """
    create_mlm_data(
        "/checkpoint/vladk/dpr_open_source/biencoder-nq-{}.json".format(split),
        "/checkpoint/kushall/data/speechqa/nq/hubert_quantization_retriever/{split}.tsv".format(split=split),
        "//checkpoint/kushall/data/speechqa/nq/hubert_clustering/layer6_km100/retriever/{split}_deduped_0_1.km".format(
            split=split
        ),
        256,
        "/checkpoint/vladk/speechqa/data/{split}/mlm-q2q-l256.txt".format(split=split),
        q2q=True,
    )

    create_mlm_data(
        "/checkpoint/vladk/dpr_open_source/biencoder-nq-{}.json".format(split),
        "/checkpoint/kushall/data/speechqa/nq/hubert_quantization_retriever/{split}.tsv".format(split=split),
        "//checkpoint/kushall/data/speechqa/nq/hubert_clustering/layer6_km100/retriever/{split}_deduped_0_1.km".format(
            split=split
        ),
        256,
        "/checkpoint/vladk/speechqa/data/{split}/mlm-qaudio-l256.txt".format(split=split),
        q2q=True,
        audio_only=True,
    )
    """

    create_mlm_data(
        "/checkpoint/vladk/dpr_open_source/biencoder-nq-{}.json".format(split),
        "/checkpoint/kushall/data/speechqa/nq/hubert_quantization_retriever/{split}.tsv".format(split=split),
        "//checkpoint/kushall/data/speechqa/nq/hubert_clustering/layer6_km100/retriever/{split}_deduped_0_1.km".format(
            split=split
        ),
        256,
        "/checkpoint/vladk/speechqa/data/{split}/mlm-q2ctx-l256.txt".format(split=split),
        q2q=False,
        max_positives=1,
    )

    """
    
    create_mlm_data_from_retriever_results(
        "/checkpoint/vladk/dpr_open_source/nq_single_{}_dense_results.json".format(split),
        "/checkpoint/vladk/speechqa/data/{}/all_wav/{}.tsv".format(split, split),
        "/checkpoint/kushall/data/speechqa/nq/hubert_clustering/layer6_km100/reader/{}_deduped_0_1.km".format(split),
        176,
        "/checkpoint/vladk/speechqa/data/{}/base_retriever_results_{}.json".format(split, split),
    )

    split = "test"

    create_mlm_data_from_retriever_results(
        "/checkpoint/vladk/biencoder/validate_nq_audio_nq_speech_hf_nftnd_bsz16_80e_lr1e4_ol8_maxgrad10//nq_audio_{}.json".format(
            split
        ),
        "/checkpoint/vladk/speechqa/data/{}/{}.tsv".format(split, split),
        "/checkpoint/kushall/data/speechqa/hubert_clustering/layer6_km100/retriever/{}_deduped_0_1.km".format(split),
        176,
        "/checkpoint/vladk/speechqa/data/{}/retriever_results_{}.json".format(split, split),
    )

    for split in ["dev", "train"]:
        create_mlm_data_from_retriever_results(
            "/checkpoint/vladk/biencoder/validate_nq_audio_nq_speech_hf_nftnd_bsz16_80e_lr1e4_ol8_maxgrad10//nq_audio_{}.json".format(
                split
            ),
            "/checkpoint/vladk/speechqa/data/{}/all_wav/{}.tsv".format(split, split),
            "/checkpoint/kushall/data/speechqa/hubert_clustering/layer6_km100/reader/{}_deduped_0_1.km".format(split),
            176,
            "/checkpoint/vladk/speechqa/data/{}/retriever_results_{}.json".format(split, split),
        )
        
    create_mlm_paq_data(
        "/checkpoint/vladk/speechqa/data/paq/train/PAQ.dpr.train.jsonl",
        "/checkpoint/kushall/data/speechqa/paq/PAQ.dpr.train_questions.txt",
        "/checkpoint/kushall/data/speechqa/paq/hubert_clustering/layer6_km100/reader/train.km",
        "/checkpoint/vladk/speechqa/data/paq/train/denoised_16k/",
        "/checkpoint/kushall/data/speechqa/paq/hubert_clustering/layer6_km100/reader/train.tsv",
        5000001,
        5003001,
        256,
        "/checkpoint/vladk/speechqa/data/dev/paq5m.txt",
        text_only=False,
    )

    create_mlm_paq_data(
        "/checkpoint/vladk/speechqa/data/paq/train/PAQ.dpr.train.jsonl",
        "/checkpoint/kushall/data/speechqa/paq/PAQ.dpr.train_questions.txt",
        "/checkpoint/kushall/data/speechqa/paq/hubert_clustering/layer6_km100/reader/train.km",
        "/checkpoint/vladk/speechqa/data/paq/train/denoised_16k/",
        "/checkpoint/kushall/data/speechqa/paq/hubert_clustering/layer6_km100/reader/train.tsv",
        0,
        5000000,
        300,
        "/checkpoint/vladk/speechqa/data/train/paq5m_q2q.txt",
        text_only=False,
        question_to_questions=True,
    )

    create_mlm_paq_data(
        "/checkpoint/vladk/speechqa/data/paq/train/PAQ.dpr.train.jsonl",
        "/checkpoint/kushall/data/speechqa/paq/PAQ.dpr.train_questions.txt",
        "/checkpoint/kushall/data/speechqa/paq/hubert_clustering/layer6_km100/reader/train.km",
        "/checkpoint/vladk/speechqa/data/paq/train/denoised_16k/",
        "/checkpoint/kushall/data/speechqa/paq/hubert_clustering/layer6_km100/reader/train.tsv",
        0,
        5000000,
        256,
        "/checkpoint/vladk/speechqa/data/train/paq5m.txt",
        text_only=False,
    )
    """


if __name__ == "__main__":
    main()
