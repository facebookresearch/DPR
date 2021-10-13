from torch import Tensor as T
from transformers import AutoTokenizer, PreTrainedTokenizer


class Tensorizer(object):
    """
    Component for all text to model input data conversions and related utility methods
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int, pad_to_max: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max

    def tokenize(
        self,
        text: str,
        title: str = None,
        add_special_tokens: bool = False,
        apply_max_len: bool = False,
    ):
        """
        {'input_ids': [101, 19204, 17629, 102], 'token_type_ids': [0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1]}
        """
        raise NotImplementedError

    # Note: title, if present, is supposed to be put before text (i.e. optional title + document body)
    def text_to_tensor(
        self,
        text: str,
        title: str = None,
        add_special_tokens: bool = True,
        apply_max_len: bool = True,
    ):
        raise NotImplementedError

    def get_pair_separator_ids(self) -> T:
        raise NotImplementedError

    def get_pad_id(self) -> int:
        raise NotImplementedError

    def get_attn_mask(self, tokens_tensor: T):
        raise NotImplementedError

    def is_sub_word_id(self, token_id: int):
        raise NotImplementedError

    def to_string(self, token_ids, skip_special_tokens=True):
        raise NotImplementedError

    def set_pad_to_max(self, pad: bool):
        raise NotImplementedError

    def get_token_id(self, token: str) -> int:
        raise NotImplementedError

    def get_sep_token_id(self) -> int:
        raise NotImplementedError