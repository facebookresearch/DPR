from typing import List

from torch.utils.data import Dataset
from transformers import BatchEncoding

from utils.data_utils import Tensorizer


class EncodeDataset(Dataset):
    def __init__(self, questions: List[str], tensorizer: Tensorizer):
        self.questions = questions
        self.tensorizer = tensorizer

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, item) -> [BatchEncoding, BatchEncoding]:
        psg = self.questions[item]
        encoded_psg = self.tensorizer.tokenize(psg)
        return encoded_psg
