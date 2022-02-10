import logging
import logging
import os
import sys

import hydra
from omegaconf import DictConfig

from dpr.options import (
    setup_cfg_gpu,
    set_seed,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

log_formatter = logging.Formatter(
    "[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
console = logging.StreamHandler()
console.setFormatter(log_formatter)
logger.addHandler(console)


@hydra.main(config_path="conf", config_name="gen_ctx_only_data")
def main(cfg: DictConfig):
    if cfg.output_dir is not None:
        os.makedirs(cfg.output_dir, exist_ok=True)

    cfg = setup_cfg_gpu(cfg)
    set_seed(cfg)

    logger.info("CFG (after gpu  configuration):")
    logger.info("%s", cfg.pretty())

    # iterate over list of datasets, extract passages and mine positives from ctx source

    # compose ctx database

    ctx_src = hydra.utils.instantiate(cfg.ctx_sources[cfg.ctx_src])

    all_passages = {}
    ctx_src.load_data_to(all_passages)

    # gen title->[ctx] dict
    title_to_ctx = {}
    for id, biencoder_passage in all_passages.items():
        title_passages = title_to_ctx.get(biencoder_passage.title, [])
        title_passages.append(biencoder_passage.text)
        title_to_ctx[biencoder_passage.title] = title_passages

    # load biencoder  dataset
    ds_key = cfg.dataset
    logger.info("dataset: %s", ds_key)
    samples_src = hydra.utils.instantiate(cfg.datasets[ds_key])
    samples_src.load_data()

    # create dataset passages -> their title relevant data
    take_positive_num = 5
    take_negative_num = 5
    result = []
    for sample in samples_src:  # BiEncoderSample
        for biencoder_passage in sample.positive_passages:
            biencoder_passage.title


if __name__ == "__main__":
    logger.info("Sys.argv: %s", sys.argv)
    hydra_formatted_args = []

    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--") :])
        else:
            hydra_formatted_args.append(arg)
    logger.info("Hydra formatted Sys.argv: %s", hydra_formatted_args)
    sys.argv = hydra_formatted_args
    main()
