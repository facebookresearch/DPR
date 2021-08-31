import logging
import math
import os
import sys
from dataclasses import dataclass
from dataclasses import field
from typing import Dict, List, Optional, Tuple, Union

import torch
import transformers
from datasets import load_dataset
from transformers import BertTokenizer, BertForPreTraining
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

# from transformers.models.bert.tokenization_bert import BertTokenizer
# from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.0")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    seq_classification: bool = field(
        default=False,
        metadata={"help": "Use MLM+NSP model" "with private models)."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={"help": "The percentage of the train set used as validation set in case there's no validation split"},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )

    # Speech Q&A ----------------------------------------------------
    mask_audio: bool = field(
        default=False,
    )
    mask_text: bool = field(
        default=False,
    )
    audio_token_type_id: Optional[int] = field(
        default=0,
    )

    audio_mlm_prob: Optional[float] = field(
        default=0.03,
    )
    audio_mlm_t_sigma: Optional[float] = field(
        default=3.0,
    )
    audio_mlm_t_max: Optional[int] = field(
        default=20,
    )
    audio_mlm_t_min: Optional[int] = field(
        default=3,
    )

    # ----------------------------------------------------

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
        if "validation" not in datasets.keys():
            datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
            )
            datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        extension = None
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if extension and extension == "txt":
            extension = "text"
        datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    logger.info("Config=%s", config)
    logger.info("model_args=%s", model_args)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        # [vk] -----------------------------------------------------------------
        tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
        # tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    logger.info("!!! tokenizer=%s", type(tokenizer))

    # [vk]: insert new wav2vec tokens --------------------------------------------

    if not os.path.exists(model_args.model_name_or_path):
        new_tokens_prefix = "w2v"
        new_tokens = ["[" + new_tokens_prefix + str(i) + "]" for i in range(100)]
        from dpr.models.hf_models import _add_special_tokens

        _add_special_tokens(tokenizer, new_tokens)

    """
    logger.info("additional_special_tokens %s", tokenizer.additional_special_tokens)
    logger.info("all_special_tokens_extended: %s", tokenizer.all_special_tokens_extended)
    logger.info("additional_special_tokens_ids: %s", tokenizer.additional_special_tokens_ids)
    logger.info("all_special_tokens %s", tokenizer.all_special_tokens)

    logger.info("!!! test tokenize %s", tokenizer.tokenize("[CLS] [w2v60] [w2v19] [w2v46][SEP]does"))
    enc = tokenizer.encode("[CLS] [w2v60] [w2v19] [w2v46] [w2v24][SEP] does")
    logger.info("!!! test encode %s", enc)
    logger.info("!!! test decode %s", tokenizer.decode(enc))
    """

    """
    if not os.path.exists(model_args.model_name_or_path):
        new_tokens_prefix = "ct"
        new_tokens = ["[" + new_tokens_prefix + str(i) + "]" for i in range(100)]
        from dpr.models.hf_models import _add_special_tokens

        _add_special_tokens(tokenizer, new_tokens)
    """

    # ----------------------------------------------------------------------------

    if model_args.seq_classification and model_args.model_name_or_path:
        logger.info("Initializing mlm+nsp model")
        model = BertForPreTraining.from_pretrained(model_args.model_name_or_path)
    elif model_args.model_name_or_path:

        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))
    logger.info("!!! model %s", type(model))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    logger.info("!!! max_seq_length %d", max_seq_length)

    if data_args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        logger.info("!!! line_by_line")
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
            return tokenizer(
                examples["text"],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=[text_column_name],
            load_from_cache_file=not data_args.overwrite_cache,
        )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    # Data collator
    # This one will take care of randomly masking the tokens.
    logger.info("!!! Init data collator, mlm prob=%s", data_args.mlm_probability)
    eval_gen = torch.Generator()
    eval_gen.manual_seed(3421798903252422493)
    pad_to_multiple_of_8 = data_args.line_by_line and training_args.fp16 and not data_args.pad_to_max_length
    data_collator = SpeechDataCollatorForMLM(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        eval_gen=eval_gen,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        mask_audio=data_args.mask_audio,
        mask_text=data_args.mask_text,
        audio_token_type_id=data_args.audio_token_type_id,
        audio_mlm_prob=data_args.audio_mlm_prob,
        audio_mlm_t_sigma=data_args.audio_mlm_t_sigma,
        audio_mlm_t_max=data_args.audio_mlm_t_max,
        audio_mlm_t_min=data_args.audio_mlm_t_min,
        nsp_expand=model_args.seq_classification,
    )

    # DataCollatorForLM

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[SpeechMLMCallback(data_collator)],
    )

    logger.info("!!! Training")
    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
        perplexity = math.exp(metrics["eval_loss"])
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


@dataclass
class SpeechDataCollatorForMLM:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use masked language modeling. If set to :obj:`False`, the labels are the same as the
            inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
            non-masked tokens and the value to predict for the masked token.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
    .. note::
        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    """

    tokenizer: PreTrainedTokenizerBase
    eval_gen: torch.Generator
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    audio_mlm_prob: int = 0.03
    audio_mlm_t_sigma: float = 3.0
    audio_mlm_t_max: int = 40
    audio_mlm_t_min: int = 8
    use_eval_gen: bool = False

    only_text_mode: bool = False
    only_text_token_type_id: int = 0
    mask_text: bool = True
    mask_audio: bool = True
    audio_token_type_id: int = 1
    nsp_expand: bool = False

    def __post_init__(self):
        if self.only_text_mode:
            self.mask_audio = False
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )
        attrs = vars(self)
        logger.info("Collator initialized: %s", ", ".join("%s: %s" % item for item in attrs.items()))

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.

        if self.nsp_expand:
            examples, nsp_labels = expand_nsp(self.tokenizer, examples)

        # logger.info("!!! examples %s", len(examples))
        # logger.info("!!! examples0 input_ids %s", type(examples[0]["input_ids"]))

        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            logger.info("!!! not BatchEncoding ")
            batch = {"input_ids": _collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)}

        if self.nsp_expand:
            batch["next_sentence_label"] = torch.tensor(nsp_labels).to(batch["input_ids"].device)

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)

        # if not special_tokens_mask:
        #    logger.info('!!! no special mask')

        # logger.info("!!! input_ids 0 =%s %s", batch["input_ids"][0], batch["input_ids"].size())
        # logging.info("!!! attention_mask 0 =%s", batch["attention_mask"][0])
        # logging.info("!!! decoded input_ids 0 =%s", self.tokenizer.decode(batch["input_ids"][0]))

        token_type_ids = self._create_toke_type_ids(batch["input_ids"], batch["token_type_ids"], self.tokenizer)
        batch["token_type_ids"] = token_type_ids

        # logging.info("!!! token_type_ids 0 =%s", token_type_ids[0])

        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], token_type_ids, special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels

        # logger.info('!!! labels %s', batch["labels"][0])
        # logger.info('!!! inps %s', batch["input_ids"][0])
        return batch

    def mask_tokens(
        self, inputs: torch.Tensor, token_type_ids: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

            # [vk], this is only for BertTokenizer (non 'Fast')
            special_tokens_mask |= inputs == self.tokenizer.pad_token_id
        else:
            special_tokens_mask = special_tokens_mask.bool()

        # logger.info("!!! special_tokens_mask=%s", special_tokens_mask[0])

        if self.only_text_mode:
            text_mask = torch.full(labels.shape, dtype=torch.bool, fill_value=True)
            wav_mask = torch.full(labels.shape, dtype=torch.bool, fill_value=False)
        else:
            audio_ttid = self.audio_token_type_id
            text_ttid = 0 if audio_ttid == 1 else 1
            wav_mask = token_type_ids == audio_ttid
            text_mask = token_type_ids == text_ttid

        if self.mask_text:
            # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
            text_p_matrix = torch.full(labels.shape, self.mlm_probability)
            text_p_matrix.masked_fill_(special_tokens_mask, value=0.0)
            text_p_matrix.masked_fill_(wav_mask, value=0.0)
            text_masked_indices = self._bernoulli(text_p_matrix)

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = self._bernoulli(torch.full(labels.shape, 0.8)) & text_masked_indices
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            # 10% of the time, we replace masked input tokens with random word
            indices_random = self._bernoulli(torch.full(labels.shape, 0.5)) & text_masked_indices & ~indices_replaced

            random_words = torch.randint(low=1000, high=len(self.tokenizer), size=labels.shape, dtype=torch.long)
            inputs[indices_random] = random_words[indices_random]
        else:
            text_masked_indices = torch.full(labels.shape, False)

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        # -------------- audio part -----------------------
        if self.mask_audio:
            wav_p_matrix = torch.full(labels.shape, self.audio_mlm_prob)
            wav_p_matrix.masked_fill_(special_tokens_mask, value=0.0)
            wav_p_matrix.masked_fill_(text_mask, value=0.0)

            while True:
                wav_masked_indices = self._bernoulli(wav_p_matrix)
                segments_num = torch.count_nonzero(wav_masked_indices).item()
                if segments_num > 0:
                    break

            # logging.info('!!! segments_num %d', segments_num)
            wav_mask_budget = float(torch.count_nonzero(wav_mask & ~special_tokens_mask).item() * self.mlm_probability)
            # logging.info('!!! wav_mask_budget %d', wav_mask_budget)

            av_segments_len = wav_mask_budget / segments_num
            # logging.info('!!! av_segments_len %d', av_segments_len)

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            wav_indices_replaced = self._bernoulli(torch.full(labels.shape, 0.8)) & wav_masked_indices
            # extend masked segments for consecutive t tokens.
            wav_indices_replaced_segments = self._gen_segments_mask(wav_indices_replaced, av_segments_len)
            # apply wav_mask again to avoid overlap with the text part
            wav_indices_replaced_segments = wav_indices_replaced_segments & wav_mask
            inputs[wav_indices_replaced_segments] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            # logging.info("!!! inputs0 repl=%s", inputs[0])

            # replaced_tokens = torch.torch.count_nonzero(wav_indices_replaced_segments[0]).item()
            # logger.info('!!! wav_indices_replaced_segments %s', wav_indices_replaced_segments[0].byte())
            # logger.info('!!! replaced tokens %s', replaced_tokens)

            # ALTERNATIVE UNCHANGED MASKING: adjusting to remaining budget
            # remaining_mask_budget = wav_mask_budget - replaced_tokens
            # logger.info('!!! remaining_mask_budget %s', remaining_mask_budget)
            # if remaining_mask_budget > 0:
            # wav_masked_unchanged_segments_num = torch.count_nonzero(wav_masked_unchanged_indices).item()
            # av_unchanged_segments_len = remaining_mask_budget / wav_masked_unchanged_segments_num

            wav_masked_unchanged_indices = wav_masked_indices & ~wav_indices_replaced_segments
            wav_masked_unchanged_segments = self._gen_segments_mask(wav_masked_unchanged_indices, av_segments_len)

            wav_masked_unchanged_segments = wav_masked_unchanged_segments & wav_mask
            # wav_masked_unchanged_tokens = torch.count_nonzero(wav_masked_unchanged_segments[0]).item()
            # logger.info('!!! unchanged tokens %s', wav_masked_unchanged_tokens)
            # logger.info('!!! budget vs replaced+unchanged %s | %s', wav_mask_budget,           (replaced_tokens + wav_masked_unchanged_tokens))

            masked_indices = text_masked_indices | wav_indices_replaced_segments | wav_masked_unchanged_segments
        else:
            masked_indices = text_masked_indices

        # logger.info('!!! masked_indices %s, %s', masked_indices[0].byte(), masked_indices.dtype)
        # total_masked_tokens = torch.count_nonzero(masked_indices).item()
        # logging.info('!!! total_masked_tokens=%d', total_masked_tokens)

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        return inputs, labels

    def _create_toke_type_ids(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, tokenizer: BertTokenizer):

        sep_tokens_indexes = torch.nonzero(input_ids == tokenizer.sep_token_id)
        bsz = input_ids.size(0)

        if sep_tokens_indexes.size(0) == 2 * bsz:
            double_sep = True
        elif sep_tokens_indexes.size(0) == bsz:
            # audio or text only
            double_sep = False
        else:
            raise RuntimeError(
                "Incorrect SEP token presence bsz={} sep_tokens_indexes={}".format(bsz, sep_tokens_indexes)
            )

        token_type_ids = torch.full(input_ids.shape, dtype=token_type_ids.dtype, fill_value=0, device=input_ids.device)

        if self.only_text_mode and sep_tokens_indexes.size() != (bsz * 2, 2):
            # Text only
            # logging.warning(f"Unexpected sep_tokens_indexes size={sep_tokens_indexes.size()}")
            # logging.info('!!! token_type_ids %s', token_type_ids)
            token_type_ids.fill_(self.only_text_token_type_id)
            # pass
        else:
            audio_ttid = self.audio_token_type_id
            text_ttid = 0 if audio_ttid == 1 else 1
            for i in range(bsz):
                # audio part is at the left side of the first SEP token
                """
                if sep_tokens_indexes[2 * i, 1] + 1 >= token_type_ids[i].size(0):
                    logger.warning(
                        "!!! size mismatch tti=%s while sep_tokens_indexes[2 * i]=%s",
                        token_type_ids[i].size(),
                        sep_tokens_indexes[2 * i],
                    )
                """
                if double_sep:
                    token_type_ids[i, 0 : sep_tokens_indexes[2 * i, 1] + 1] = audio_ttid
                    token_type_ids[i, sep_tokens_indexes[2 * i, 1] + 1 :] = text_ttid
                else:  # audio only
                    token_type_ids[i, 0 : sep_tokens_indexes[i, 1] + 1] = audio_ttid
                    # token_type_ids[i, sep_tokens_indexes[2 * i, 1] + 1:] = text_ttid
        return token_type_ids

    def _gen_segment_len(self, average: float, shape: Tuple = (1,)) -> Union[int, torch.Tensor]:

        r = torch.normal(average, self.audio_mlm_t_sigma, shape, generator=self.eval_gen if self.use_eval_gen else None)
        r = r.round().int().clamp(self.audio_mlm_t_min, self.audio_mlm_t_max)
        if shape == (1,):
            return r.item()
        return r

    def _bernoulli(self, matrix: torch.Tensor):
        return torch.bernoulli(matrix, generator=self.eval_gen if self.use_eval_gen else None).bool()

    def _gen_segments_mask(self, indices: torch.Tensor, av_len: int):
        indices_segments = indices.clone()
        lengths = self._gen_segment_len(av_len, shape=indices.size())
        lengths[~indices] = 0
        # logging.info('!! lengths %s', lengths)
        max_len = lengths.max().item()
        # logging.info('!! max_len %s', max_len)

        # for k in range(indices.size(0)):
        #    logging.info('!!! initial indices %d : %s', k, indices_segments[k].byte())

        for i in range(1, max_len):
            shifted_indices = indices[:, 0:-i]
            # logging.info('!! shifted_indices %s', shifted_indices)
            stop_mask = (lengths > i)[:, 0:-i]
            # logging.info('!! i=%d stop_mask %s', i, stop_mask.byte())
            shifted_indices = shifted_indices & stop_mask
            # logging.info('!! adjusted shifted_indices %s', shifted_indices)
            indices_segments[:, i:] |= shifted_indices

        # for k in range(indices.size(0)):
        #    logging.info('!!! indices_segments %d : %s', k, indices_segments[k].byte())

        return indices_segments


def expand_nsp(
    tokenizer,
    examples: List[Dict[str, List[int]]],
    max_neg_per_question: int = 1,
) -> (List[Dict[str, List[int]]], List[int]):
    bsz = len(examples)
    questions_ids = []
    ctx_ids = []
    positive_pair_ids = []
    for e in examples:
        token_ids = e["input_ids"]
        sep_idx = token_ids.index(102)
        questions_ids.append(token_ids[:sep_idx])
        ctx_ids.append(token_ids[sep_idx:])
        positive_pair_ids.append(token_ids)

    expanded_input_ids = []
    labels = []

    def _trunc_max(ids_list: List[int], max_len: int = 512):
        ids_list = ids_list[0:max_len]
        ids_list[-1] = tokenizer.sep_token_id
        return ids_list

    for i, q in enumerate(questions_ids):
        expanded_input_ids.append(_trunc_max(positive_pair_ids[i]))
        labels.append(0)
        q_positive_ctx = ctx_ids[i]
        # create negative pairs
        q_neg_added = 0
        for j in range(bsz):
            neg_ctx = ctx_ids[j]
            if j == i or neg_ctx == q_positive_ctx:
                continue
            neg_pair = q + neg_ctx
            expanded_input_ids.append(_trunc_max(neg_pair))
            labels.append(1)
            q_neg_added += 1
            if q_neg_added >= max_neg_per_question:
                break

    expanded_examples = [
        {
            "input_ids": expanded_input_ids[i],
            "attention_mask": [1] * len(expanded_input_ids[i]),
            "token_type_ids": [0] * len(expanded_input_ids[i]),
        }
        for i in range(len(expanded_input_ids))
    ]
    return expanded_examples, labels


class SpeechMLMCallback(transformers.TrainerCallback):
    def __init__(self, collator: SpeechDataCollatorForMLM):
        self.collator = collator

    def on_epoch_begin(self, args, state, control, **kwargs):
        logging.info("!! begin epoch")

    def on_evaluate(self, args, state, control, **kwargs):
        logging.info("!! on_evaluate")

    def on_step_begin(self, args, state, control, **kwargs):
        self.collator.use_eval_gen = False

    def on_prediction_step(self, args, state, control, **kwargs):
        self.collator.use_eval_gen = True


#######################################################################################################


@dataclass
class DataCollatorForLM:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use masked language modeling. If set to :obj:`False`, the labels are the same as the
            inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
            non-masked tokens and the value to predict for the masked token.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
    .. note::
        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {"input_ids": _collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)}

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
