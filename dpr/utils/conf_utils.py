import glob
import logging
import os

import hydra
from omegaconf import DictConfig

from dpr.data.biencoder_data import JsonQADataset

logger = logging.getLogger(__name__)


class BiencoderDatasetsCfg(object):
    def __init__(self, cfg: DictConfig):
        ds_cfg = cfg.datasets
        self.train_datasets_names = cfg.train_datasets
        logger.info("train_datasets: %s", self.train_datasets_names)
        self.train_datasets = _init_datasets(self.train_datasets_names, ds_cfg)
        self.dev_datasets_names = cfg.dev_datasets
        logger.info("dev_datasets: %s", self.dev_datasets_names)
        self.dev_datasets = _init_datasets(self.dev_datasets_names, ds_cfg)
        self.sampling_rates = cfg.train_sampling_rates


def _init_datasets(datasets_names, ds_cfg: DictConfig):
    if isinstance(datasets_names, str):
        return [_init_dataset(datasets_names, ds_cfg)]
    elif datasets_names:
        return [_init_dataset(ds_name, ds_cfg) for ds_name in datasets_names]
    else:
        return []


def _init_dataset(name: str, ds_cfg: DictConfig):
    if os.path.exists(name):
        # use default biencoder json class
        return JsonQADataset(name)
    elif glob.glob(name):
        files = glob.glob(name)
        return [_init_dataset(f, ds_cfg) for f in files]
    # try to find in cfg
    if name not in ds_cfg:
        raise RuntimeError("Can't find dataset location/config for: {}".format(name))
    return hydra.utils.instantiate(ds_cfg[name])
