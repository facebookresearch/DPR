import collections
import csv
import glob
import logging
import os
import random
from typing import Dict, List, Tuple

import hydra
import jsonlines
import numpy as np
import soundfile as sf
import torch

# import torchaudio
import torch.nn.functional as F

from omegaconf import DictConfig
from torch import Tensor as T


from data.biencoder_data import (
    Dataset,
    BiEncoderPassage,
    get_dpr_files,
    normalize_passage,
)
from dpr.utils.data_utils import read_data_from_json_files, Tensorizer


logger = logging.getLogger(__name__)


class BiEncoderMixedSample(object):
    query: T
    positive_passages: List[BiEncoderPassage]
    negative_passages: List[BiEncoderPassage]
    hard_negative_passages: List[BiEncoderPassage]


class WavJsonTextDataset(Dataset):
    def __init__(
        self,
        json_file: str,
        wav_tsv_file: str,
        selector: DictConfig = None,
        encoder_type: str = None,
        shuffle_positives: bool = False,
        normalize_text: bool = False,
        normalize_audio: bool = False,
        audio_file_prefix: str = "aud_dn_",
    ):
        super().__init__(
            selector,
            encoder_type=encoder_type,
            shuffle_positives=shuffle_positives,
        )
        self.json_file = json_file
        self.wav_tsv_file = wav_tsv_file
        self.audio_file_prefix = audio_file_prefix

        self.data_files = []
        self.data = []
        self.normalize_text = normalize_text
        self.normalize_audio = normalize_audio
        self.id_to_audio_file_map = None
        logger.info("Data files: %s", self.data_files)

    def load_data(self):
        self.data_files = get_dpr_files(self.file)
        data = read_data_from_json_files(self.data_files)
        # filter those without positive ctx
        self.data = [r for r in data if len(r["positive_ctxs"]) > 0]
        logger.info("Total cleaned data size: {}".format(len(self.data)))
        self.id_to_audio_file_map = self._get_id_to_audio_file_map()
        logger.info("id_to_audio_file_map  size: %d", len(self.id_to_audio_file_map))

    def __getitem__(self, index) -> BiEncoderMixedSample:
        json_sample = self.data[index]
        r = BiEncoderMixedSample()
        # r.query = self._process_query(json_sample["question"])
        sample_id = index + 1
        audio_file = self.id_to_audio_file_map[sample_id]
        # r.query = torchaudio.load(audio_file)

        query_tensor = self.get_audio_feats(audio_file)
        logger.info("Audio query_tensor %s", query_tensor.size())

        positive_ctxs = json_sample["positive_ctxs"]
        negative_ctxs = (
            json_sample["negative_ctxs"] if "negative_ctxs" in json_sample else []
        )
        hard_negative_ctxs = (
            json_sample["hard_negative_ctxs"]
            if "hard_negative_ctxs" in json_sample
            else []
        )

        for ctx in positive_ctxs + negative_ctxs + hard_negative_ctxs:
            if "title" not in ctx:
                ctx["title"] = None

        def create_passage(ctx: dict):
            return BiEncoderPassage(
                normalize_passage(ctx["text"]) if self.normalize else ctx["text"],
                ctx["title"],
            )

        r.positive_passages = [create_passage(ctx) for ctx in positive_ctxs]
        r.negative_passages = [create_passage(ctx) for ctx in negative_ctxs]
        r.hard_negative_passages = [create_passage(ctx) for ctx in hard_negative_ctxs]
        return r

    def __len__(self):
        return len(self.data)

    #############################################

    def _get_id_to_audio_file_map(self):
        id_to_file_map = {}
        prefix_len = len(self.audio_file_prefix)
        suffix_len = len(".wav")
        with open(os.path.join(self.file), "r") as fp:  # read tsv file
            lines = fp.read().split("\n")
            root = lines.pop(0).strip()
            for line in lines:
                if len(line) == 0:
                    continue
                file = line.split("\t")[0]
                id = int(file[prefix_len:-suffix_len])
                file_path = os.path.join(root, file)
                id_to_file_map[id] = file_path
        return id_to_file_map

    def read_audio(self, fname):
        """ Load an audio file and return PCM along with the sample rate """
        wav, sr = sf.read(fname)
        assert sr == 16e3
        return wav

    def get_audio_feats(self, loc) -> T:
        x = self.read_audio(loc)
        with torch.no_grad():
            source = torch.from_numpy(x).float().cuda()
            if self.normalize_audio:
                assert source.dim() == 1, source.dim()
                with torch.no_grad():
                    source = F.layer_norm(source, source.shape)
            source = source.view(1, -1)
        return source

        # m_res = self.model(source=source, padding_mask=None)
        # res = m_res["encoder_out"].squeeze(1)
        # res = res.argmax(-1).cpu()
        # w2v_out = m_res["w2v_out"].squeeze(0).cpu()

        # return w2v_out, res
