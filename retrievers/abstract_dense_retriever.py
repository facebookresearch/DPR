from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from torch import nn
from transformers import TrainingArguments

from utils.data_utils import Tensorizer


class DenseRetriever(ABC):
    def __init__(
        self,
        question_encoder: nn.Module,
        batch_size: int,
        tensorizer: Tensorizer,
        index,
        device: torch.device = None,
    ):
        self.question_encoder = question_encoder
        self.question_encoder.eval()
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.index = index
        self.selector = None
        if not device:
            device = TrainingArguments("").device
        self.device = device

    @abstractmethod
    def generate_question_vectors(self, questions: List[str], query_token: str = None):
        raise NotImplementedError

    @abstractmethod
    def load_encoded_index_data(
        self,
        vector_files: List[str],
        buffer_size: int,
        doc_prefixes: List = None,
        **kwargs
    ):
        pass

    @abstractmethod
    def get_top_docs(
        self, query_vectors, top_docs: int = 100
    ) -> List[Tuple[List[object], List[float]]]:
        raise NotImplementedError
