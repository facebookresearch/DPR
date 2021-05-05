#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Command line arguments utils
"""


import logging
import subprocess

import numpy as np
import os
import random
import socket
import torch

from omegaconf import DictConfig

logger = logging.getLogger()

# TODO: to be merged with conf_utils.py


def set_cfg_params_from_state(state: dict, cfg: DictConfig):
    """
    Overrides some of the encoder config parameters from a give state object
    """
    if not state:
        return

    cfg.do_lower_case = state["do_lower_case"]

    if "encoder" in state:
        saved_encoder_params = state["encoder"]
        # TODO: try to understand why cfg.encoder = state["encoder"] doesn't work

        for k, v in saved_encoder_params.items():

            # TODO: tmp fix
            if k=='q_wav2vec_model_cfg':
                k='q_encoder_model_cfg'
            if k=='q_wav2vec_cp_file':
                k='q_encoder_cp_file'
            if k=='q_wav2vec_cp_file':
                k='q_encoder_cp_file'



            setattr(cfg.encoder, k, v)
    else:  # 'old' checkpoints backward compatibility support
        pass
        # TODO: tmp
        # cfg.encoder.pretrained_model_cfg = state["pretrained_model_cfg"]
        # cfg.encoder.encoder_model_type = state["encoder_model_type"]
        # cfg.encoder.pretrained_file = state["pretrained_file"]
        # cfg.encoder.projection_dim = state["projection_dim"]
        # cfg.encoder.sequence_length = state["sequence_length"]


def get_encoder_params_state_from_cfg(cfg: DictConfig):
    """
    Selects the param values to be saved in a checkpoint, so that a trained model can be used for downstream
    tasks without the need to specify these parameter again
    :return: Dict of params to memorize in a checkpoint
    """
    return {
        "do_lower_case": cfg.do_lower_case,
        "encoder": cfg.encoder,
    }


def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def setup_cfg_gpu(cfg):
    """
    Setup params for CUDA, GPU & distributed training
    """
    logger.info("args.local_rank %s", cfg.local_rank)
    ws = os.environ.get("WORLD_SIZE")
    cfg.distributed_world_size = int(ws) if ws else 1
    logger.info("WORLD_SIZE %s", ws)
    if cfg.local_rank == -1 or cfg.no_cuda:  # single-node multi-gpu (or cpu) mode
        device = str(torch.device("cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu"))
        cfg.n_gpu = torch.cuda.device_count()
    else:  # distributed mode
        torch.cuda.set_device(cfg.local_rank)
        device = str(torch.device("cuda", cfg.local_rank))
        torch.distributed.init_process_group(backend="nccl")
        cfg.n_gpu = 1

    cfg.device = device

    logger.info(
        "Initialized host %s as d.rank %d on device=%s, n_gpu=%d, world size=%d",
        socket.gethostname(),
        cfg.local_rank,
        cfg.device,
        cfg.n_gpu,
        cfg.distributed_world_size,
    )
    logger.info("16-bits training: %s ", cfg.fp16)
    return cfg


def _infer_slurm_init(cfg):

    # if cfg.distributed_port

    node_list = os.environ.get("SLURM_STEP_NODELIST")
    if node_list is None:
        node_list = os.environ.get("SLURM_JOB_NODELIST")
    logger.info("SLURM_JOB_NODELIST: %s", node_list)

    # cfg.n_gpu
    # cfg.local_rank
    # cfg.distributed_world_size

    if node_list is not None:
        try:
            hostnames = subprocess.check_output(["scontrol", "show", "hostnames", node_list])
            distributed_init_method = "tcp://{host}:{port}".format(
                host=hostnames.split()[0].decode("utf-8"),
                port=cfg.distributed_port,
            )
            nnodes = int(os.environ.get("SLURM_NNODES"))

            logger.info("SLURM_NNODES: %s", nnodes)

            ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")
            if ntasks_per_node is not None:
                ntasks_per_node = int(ntasks_per_node)
            else:
                ntasks = int(os.environ.get("SLURM_NTASKS"))
                nnodes = int(os.environ.get("SLURM_NNODES"))
                assert ntasks % nnodes == 0
                ntasks_per_node = int(ntasks / nnodes)
            if ntasks_per_node == 1:
                gpus_per_node = torch.cuda.device_count()
                node_id = int(os.environ.get("SLURM_NODEID"))
                cfg.distributed_rank = node_id * gpus_per_node
                cfg.distributed_world_size = nnodes * gpus_per_node
            # cfg.distributed_rank = int(os.environ.get("SLURM_PROCID"))
            # cfg.device_id = int(os.environ.get("SLURM_LOCALID"))

        except subprocess.CalledProcessError as e:  # scontrol failed
            raise e
        except FileNotFoundError:  # Slurm is not installed
            pass


def setup_logger(logger):
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    log_formatter = logging.Formatter("[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(log_formatter)
    logger.addHandler(console)
