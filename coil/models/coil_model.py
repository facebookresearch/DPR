import logging
import os
from typing import Dict

import torch
import torch.distributed as dist
from torch import Tensor
from torch import nn
from torch.cuda.amp import autocast
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizer,
    Trainer,
)
from transformers import PreTrainedModel, TrainingArguments
from transformers.modeling_outputs import BaseModelOutputWithPooling

from dpr.options import setup_logger
from utils.data_utils import Tensorizer

logger = logging.getLogger(__name__)
setup_logger(logger)


class COIL(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        model_args,
        data_args,
        train_args: TrainingArguments,
    ):
        super().__init__()
        self.model: PreTrainedModel = model
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.data_args, self.model_args, self.train_args = (
            data_args,
            model_args,
            train_args,
        )
        self.tok_proj = nn.Linear(768, model_args.token_dim)
        self.cls_proj = nn.Linear(768, model_args.cls_dim)

        if model_args.token_norm_after:
            self.ln_tok = nn.LayerNorm(model_args.token_dim)
        if model_args.cls_norm_after:
            self.ln_cls = nn.LayerNorm(model_args.cls_dim)

    @classmethod
    def from_pretrained(
        cls, model_args, data_args, train_args: TrainingArguments, *args, **kwargs
    ):
        hf_model = AutoModel.from_pretrained(*args, **kwargs)
        model = COIL(hf_model, model_args, data_args, train_args)
        path = args[0]
        if os.path.exists(os.path.join(path, "model.pt")):
            logger.info("loading extra weights from local files")
            model_dict = torch.load(os.path.join(path, "model.pt"), map_location="cpu")
            model.load_state_dict(model_dict, strict=False)
        else:
            logger.info("using pre-trained model without loading local checkpoint")
        return model

    def save_pretrained(self, output_dir: str):
        self.model.save_pretrained(output_dir)
        model_dict = self.state_dict()
        hf_weight_keys = [k for k in model_dict.keys() if k.startswith("model")]
        for k in hf_weight_keys:
            model_dict.pop(k)
        torch.save(model_dict, os.path.join(output_dir, "model.pt"))
        torch.save(
            [self.data_args, self.model_args, self.train_args],
            os.path.join(output_dir, "args.pt"),
        )

    def encode(self, **features):
        assert all(
            [x in features for x in ["input_ids", "attention_mask", "token_type_ids"]]
        )
        model_out: BaseModelOutputWithPooling = self.model(**features, return_dict=True)
        cls = self.cls_proj(model_out.last_hidden_state[:, 0])
        reps = self.tok_proj(model_out.last_hidden_state)
        if self.model_args.cls_norm_after:
            cls = self.ln_cls(cls)
        if self.model_args.token_norm_after:
            reps = self.ln_tok(reps)

        if self.model_args.token_rep_relu:
            reps = torch.relu(reps)

        return cls, reps

    def get_out_size(self):
        return None

    def forward(self, qry_input: Dict, doc_input: Dict):
        qry_out: BaseModelOutputWithPooling = self.model(**qry_input, return_dict=True)
        doc_out: BaseModelOutputWithPooling = self.model(**doc_input, return_dict=True)

        qry_cls = self.cls_proj(qry_out.last_hidden_state[:, 0])
        doc_cls = self.cls_proj(doc_out.last_hidden_state[:, 0])

        qry_reps = self.tok_proj(qry_out.last_hidden_state)  # Q * LQ * d
        doc_reps = self.tok_proj(doc_out.last_hidden_state)  # D * LD * d

        if self.model_args.cls_norm_after:
            qry_cls, doc_cls = self.ln_cls(qry_cls), self.ln_cls(doc_cls)
        if self.model_args.token_norm_after:
            qry_reps, doc_reps = self.ln_tok(qry_reps), self.ln_tok(doc_reps)

        if self.model_args.token_rep_relu:
            qry_reps = torch.relu(qry_reps)
            doc_reps = torch.relu(doc_reps)

        # mask ingredients
        doc_input_ids: Tensor = doc_input["input_ids"]
        qry_input_ids: Tensor = qry_input["input_ids"]
        qry_attention_mask: Tensor = qry_input["attention_mask"]

        self.mask_sep(qry_attention_mask)

        if not self.training:
            # in testing phase, we have Q == D
            assert doc_input_ids.size(0) == qry_input_ids.size(
                0
            ), "we expect same number of query/doc"
            tok_scores = self.compute_tok_score_pair(
                doc_reps, doc_input_ids, qry_reps, qry_input_ids, qry_attention_mask
            )

            # compute cls score separately
            cls_scores = (qry_cls * doc_cls).sum(-1)

            # sum the scores
            if self.model_args.no_cls:
                scores = tok_scores
            elif self.model_args.cls_only:
                scores = cls_scores
            else:
                if self.train_args.fp16:
                    with autocast(False):
                        scores = tok_scores.float() + cls_scores.float()  # B
                else:
                    scores = tok_scores + cls_scores  # B

            # loss not defined during inference
            return scores.view(-1)

        else:
            # for training phase, we have D = Q * group_size
            if self.model_args.x_device_negatives:
                # the idea is simple
                # fake it as if everything is on current device
                # gradient is taken care of at reduction time
                doc_input_ids, doc_cls, doc_reps = self.gather_tensors(
                    doc_input_ids, doc_cls, doc_reps
                )
                (
                    qry_input_ids,
                    qry_attention_mask,
                    qry_cls,
                    qry_reps,
                ) = self.gather_tensors(
                    qry_input_ids, qry_attention_mask, qry_cls, qry_reps
                )

            # qry_reps: Q * LQ * d
            # doc_reps: D * LD * d
            tok_scores = self.compute_tok_score_cart(
                doc_reps, doc_input_ids, qry_reps, qry_input_ids, qry_attention_mask
            )

            # remove padding and cls token
            if self.model_args.no_cls:
                scores = tok_scores
            elif self.model_args.cls_only:
                scores = torch.matmul(qry_cls, doc_cls.transpose(1, 0))  # Q * D
            else:
                cls_scores = torch.matmul(qry_cls, doc_cls.transpose(1, 0))  # Q * D
                with autocast(False):
                    scores = tok_scores.float() + cls_scores.float()  # Q * D

            labels = torch.arange(
                scores.size(0), device=doc_input["input_ids"].device, dtype=torch.long
            )
            # offset the labels
            labels = labels * self.data_args.train_group_size
            loss = self.cross_entropy(scores, labels)

            return loss, scores.view(-1)

    def mask_sep(self, qry_attention_mask):
        if self.model_args.no_sep:
            sep_pos = (
                qry_attention_mask.sum(1).unsqueeze(1) - 1
            )  # the sep token position
            _zeros = torch.zeros_like(sep_pos)
            qry_attention_mask.scatter_(1, sep_pos.long(), _zeros)

        return qry_attention_mask

    def compute_tok_score_pair(
        self, doc_reps, doc_input_ids, qry_reps, qry_input_ids, qry_attention_mask
    ):
        exact_match = qry_input_ids.unsqueeze(2) == doc_input_ids.unsqueeze(
            1
        )  # B * LQ * LD
        exact_match = exact_match.float()
        # qry_reps: B * LQ * d
        # doc_reps: B * LD * d
        scores_no_masking = torch.bmm(
            qry_reps, doc_reps.permute(0, 2, 1)
        )  # B * LQ * LD
        if self.model_args.pooling == "max":
            tok_scores, _ = (scores_no_masking * exact_match).max(dim=2)  # B * LQ
        else:
            raise NotImplementedError(
                f"{self.model_args.pooling} pooling is not defined"
            )
        # remove padding and cls token
        tok_scores = (tok_scores * qry_attention_mask)[:, 1:].sum(-1)
        return tok_scores

    def compute_tok_score_cart(
        self, doc_reps, doc_input_ids, qry_reps, qry_input_ids, qry_attention_mask
    ):
        qry_input_ids = qry_input_ids.unsqueeze(2).unsqueeze(3)  # Q * LQ * 1 * 1
        doc_input_ids = doc_input_ids.unsqueeze(0).unsqueeze(1)  # 1 * 1 * D * LD
        exact_match = doc_input_ids == qry_input_ids  # Q * LQ * D * LD
        exact_match = exact_match.float()
        scores_no_masking = torch.matmul(
            qry_reps.view(-1, self.model_args.token_dim),  # (Q * LQ) * d
            doc_reps.view(-1, self.model_args.token_dim).transpose(
                0, 1
            ),  # d * (D * LD)
        )
        scores_no_masking = scores_no_masking.view(
            *qry_reps.shape[:2], *doc_reps.shape[:2]
        )  # Q * LQ * D * LD
        # scores_no_masking = scores_no_masking.permute(0, 2, 1, 3)  # Q * D * LQ * LD
        if self.model_args.pooling == "max":
            scores, _ = (scores_no_masking * exact_match).max(dim=3)  # Q * LQ * D
        else:
            raise NotImplementedError(
                f"{self.model_args.pooling} pooling is not defined"
            )

        tok_scores = (scores * qry_attention_mask.unsqueeze(2))[:, 1:].sum(1)
        return tok_scores

    def _gather_tensor(self, t: Tensor):
        all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(all_tensors, t)
        all_tensors[self.train_args.local_rank] = t
        return all_tensors

    def gather_tensors(self, *tt: Tensor):
        tt = [torch.cat(self._gather_tensor(t)) for t in tt]
        return tt


class COILTensorizer(Tensorizer):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, max_length: int, pad_to_max: bool = False
    ):
        super().__init__(tokenizer, max_length, pad_to_max)

    def tokenize(
        self,
        text: str,
        title: str = None,
        add_special_tokens: bool = True,
        apply_max_len: bool = True,
    ):
        """
        {'input_ids': [101, 19204, 17629, 102], 'token_type_ids': [0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1]}
        """
        return self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            truncation="only_first",
            pad_to_max_length=self.pad_to_max,
        )

    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    def get_sep_token_id(self) -> int:
        return self.tokenizer.sep_token_id


def get_coil_question_encoder_components(args, inference_only: bool = False, **kwargs):
    """
    required values in args
        model_args, data_args, training_args
    """
    # coil specific args
    model_args = args.model_args
    data_args = args.data_args
    training_args = args.training_args

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, use_fast=True)
    tensorizer = COILTensorizer(tokenizer=tokenizer, max_length=data_args.q_max_len)

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=1,
        cache_dir=model_args.cache_dir,
    )

    model = COIL.from_pretrained(
        model_args,
        data_args,
        training_args,
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    optimizer = None
    if not inference_only:
        trainer = Trainer(
            model=model,
            args=training_args,
        )
        trainer.create_optimizer()
        optimizer = trainer.optimizer

    return tensorizer, model, optimizer
