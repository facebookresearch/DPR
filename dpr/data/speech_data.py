import csv
import logging
import os
from typing import List

import soundfile as sf
import torch

# import torchaudio
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor as T

from dpr.data.biencoder_data import (
    BiEncoderPassage,
    get_dpr_files,
    normalize_passage,
    JsonQADataset,
)
from dpr.data.retriever_data import QASrc, QASample
from dpr.utils.data_utils import read_data_from_json_files

logger = logging.getLogger(__name__)


class BiEncoderMixedSample(object):
    query: T
    positive_passages: List[BiEncoderPassage]
    negative_passages: List[BiEncoderPassage]
    hard_negative_passages: List[BiEncoderPassage]


class WavJsonTextDataset(JsonQADataset):
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
        max_features_sz: int = 100000,
    ):

        super().__init__(
            json_file,
            selector,
            encoder_type=encoder_type,
            shuffle_positives=shuffle_positives,
            normalize=normalize_text,
        )
        self.wav_tsv_file = wav_tsv_file
        self.audio_file_prefix = audio_file_prefix
        self.normalize_audio = normalize_audio
        self.id_to_audio_file_map = None
        self.max_features_sz = max_features_sz

        # tmp
        self.cut_samples = 0

    def load_data(self):
        self.data_files = get_dpr_files(self.file)
        logger.info("Data files: %s", self.data_files)
        data = read_data_from_json_files(self.data_files)
        # filter those without positive ctx
        self.data = [r for r in data if len(r["positive_ctxs"]) > 0]
        logger.info("Total cleaned data size: {}".format(len(self.data)))
        self.id_to_audio_file_map = _get_id_to_audio_file_map(
            self.audio_file_prefix, self.wav_tsv_file
        )
        logger.info("id_to_audio_file_map  size: %d", len(self.id_to_audio_file_map))

    def __getitem__(self, index) -> BiEncoderMixedSample:
        json_sample = self.data[index]
        r = BiEncoderMixedSample()
        sample_id = index + 1
        audio_file = self.id_to_audio_file_map[sample_id]

        query_tensor = _get_audio_feats(audio_file, self.normalize_audio)
        # logger.info("Audio query_tensor %s", query_tensor.size())

        if query_tensor.size(1) > self.max_features_sz:
            query_tensor = query_tensor[:, 0 : self.max_features_sz]
            self.cut_samples += 1
            if self.cut_samples % 100 == 0:
                logger.info("!!! cut_samples %d", self.cut_samples)

        # if query_tensor.size(1) == 371519:
        #    logger.info("!!! 371519 Audio size for file =%s", audio_file)

        # r.query = torchaudio.load(audio_file)
        r.query = query_tensor

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


class WavTextQADataset(QASrc):
    def __init__(
        self,
        file: str,
        wav_tsv_file: str,
        audio_file_prefix: str = "aud_dn_",
        max_features_sz: int = 100000,
        normalize_audio: bool = False,
    ):
        super().__init__(file)
        self.wav_tsv_file = wav_tsv_file
        self.id_to_audio_file_map = {}
        self.audio_file_prefix = audio_file_prefix
        self.max_features_sz = max_features_sz
        self.normalize_audio = normalize_audio

        # TODO: tmp
        self.length_buckets = {}

    def __getitem__(self, index) -> QASample:
        sample = self.data[index]
        sample_id = index + 1
        audio_file = self.id_to_audio_file_map[sample_id]
        query_tensor = _get_audio_feats(audio_file, self.normalize_audio)
        logger.info("Audio query_tensor %s", query_tensor.size())

        # TODO: tmp
        size = query_tensor.size(1)
        bucket = int(size / 10000)
        cnt = self.length_buckets.get(bucket, 0)
        self.length_buckets[bucket] = cnt + 1

        if query_tensor.size(1) > self.max_features_sz:
            query_tensor = query_tensor[:, 0 : self.max_features_sz]

        sample.query = query_tensor
        return sample

    def __len__(self):
        return len(self.data)

    def load_data(self):
        super().load_data()
        data = []

        with open(self.file) as ifile:
            reader = csv.reader(ifile, delimiter="\t")
            for row in reader:
                question = row[0]
                answers = eval(row[1])
                data.append(QASample(self._process_question(question), None, answers))

        self.data = data
        self.id_to_audio_file_map = _get_id_to_audio_file_map(
            self.audio_file_prefix, self.wav_tsv_file
        )
        logger.info("id_to_audio_file_map  size: %d", len(self.id_to_audio_file_map))


def _read_audio(fname):
    """ Load an audio file and return PCM along with the sample rate """
    wav, sr = sf.read(fname)
    assert sr == 16e3
    return wav


def _get_audio_feats(loc, normalize_audio: bool) -> T:
    x = _read_audio(loc)
    with torch.no_grad():
        source = torch.from_numpy(x).float()  # .cuda()
        if normalize_audio:
            assert source.dim() == 1, source.dim()
            with torch.no_grad():
                source = F.layer_norm(source, source.shape)
        source = source.view(1, -1)
    return source


def _get_id_to_audio_file_map(audio_file_prefix: str, wav_tsv_file: str):
    id_to_file_map = {}
    prefix_len = len(audio_file_prefix)
    suffix_len = len(".wav")
    with open(os.path.join(wav_tsv_file), "r") as fp:  # read tsv file
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
