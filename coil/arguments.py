import os
from dataclasses import dataclass, field
from typing import Optional, Union, List
from transformers import TrainingArguments


@dataclass
class COILModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    token_dim: int = field(default=768)
    cls_dim: int = field(default=768)
    token_rep_relu: bool = field(
        default=False,
    )
    token_norm_after: bool = field(default=False)
    cls_norm_after: bool = field(default=False)
    x_device_negatives: bool = field(default=False)
    pooling: str = field(default="max")
    no_sep: bool = field(
        default=False,
    )
    no_cls: bool = field(
        default=False,
    )
    cls_only: bool = field(
        default=False,
    )


@dataclass
class COILDataArguments:
    train_dir: str = field(default=None, metadata={"help": "Path to train directory"})
    train_path: Union[str] = field(
        default=None, metadata={"help": "Path to train data"}
    )
    train_group_size: int = field(default=8)

    pred_path: List[str] = field(
        default=None, metadata={"help": "Path to prediction data"}
    )
    pred_dir: str = field(
        default=None, metadata={"help": "Path to prediction directory"}
    )
    pred_id_file: str = field(default=None)
    rank_score_path: str = field(
        default=None, metadata={"help": "where to save the match score"}
    )

    encode_in_path: List[str] = field(
        default=None, metadata={"help": "Path to data to encode"}
    )
    encoded_save_path: str = field(
        default=None, metadata={"help": "where to save the encode"}
    )

    q_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    p_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    document: bool = field(default=False)

    def __post_init__(self):
        if self.train_dir is not None:
            files = os.listdir(self.train_dir)
            self.train_path = [
                os.path.join(self.train_dir, f)
                for f in files
                if f.endswith("tsv") or f.endswith("json")
            ]
        if self.pred_dir is not None:
            files = os.listdir(self.pred_dir)
            self.pred_path = [os.path.join(self.pred_dir, f) for f in files]


@dataclass
class COILTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0)
    do_encode: bool = field(
        default=False, metadata={"help": "Whether to run encoding on the test set."}
    )
