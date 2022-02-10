import argparse
import csv
import json
import os
import pickle
from pathlib import Path

import jsonlines


def convert_to_kilt(dpr_output, kilt_gold_file, kilt_out_file):
    with open(dpr_output, "rt") as fin:
        dpr_output = json.load(fin)

    with jsonlines.open(kilt_gold_file, "r") as reader:
        kilt_gold_file = list(reader)
    assert len(kilt_gold_file) == len(dpr_output)

    map_path = "/checkpoint/fabiopetroni/KILT/multitask_dpr_datasets/v0.20200817/map_back_to_kilt.pkl"
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

    print("Saved KILT formatted results to: ", kilt_out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ctx_file",
        type=str,
        default="/checkpoint/fabiopetroni/KILT/multitask_dpr_datasets/v0.20200817/chunks.tsv",
    )
    parser.add_argument("--dpr_output", type=str)
    parser.add_argument("--kilt_gold", type=str)
    parser.add_argument("--kilt_out_file", type=str)

    args = parser.parse_args()
    # print('args', args)

    convert_to_kilt(args.dpr_output, args.kilt_gold, args.dpr_output)


if __name__ == "__main__":
    main()
