import os
import json
import random
from pathlib import Path
from typing import Any, Dict
from datetime import datetime

import torch
import hydra
import numpy as np
from loguru import logger
from omegaconf import OmegaConf

from src.logging.get_logger import get_logger
from src.training.train import train
from src.dataloader.data import get_dataloader
from src.utils import (
    move_data_to_temp,
    plot_and_save_num_tile_distribution,
    log_df_info,
    get_cohort_df,
    verify_data_exists,
)

from src.conf.train_schema import AppCfg


@logger.catch(reraise=True)
@hydra.main(version_base=None, config_path="../../conf", config_name="default_config")
def run_experiment(config: AppCfg) -> None:
    # torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    if config.other.seed is not None:
        # In order to make torch deterministic work
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

        torch.manual_seed(config.other.seed)
        random.seed(config.other.seed)
        np.random.seed(config.other.seed)

    # Save the entire folder, for ease with inference (being able to choose dataloader etc)
    get_logger(config, log_name="train.log")
    Path(config.paths.output_dir).mkdir(exist_ok=True)

    if (
        (Path(config.paths.output_dir) / "models" / "checkpoints").exists()
        and (
            len(
                list(
                    (Path(config.paths.output_dir) / "models" / "checkpoints").glob(
                        f"*{config.training.n_steps}.pth"
                    )
                )
            )
            > 0
        )
        and not config.other.force
    ):
        logger.warning(
            f"Found a model with {config.training.n_steps} steps. Output already exists."
        )
        exit()
    with open(Path(config.paths.output_dir) / "config.yaml", "w") as f:
        OmegaConf.save(config, f)

    logger.info(str(config) + "\n\n\n")
    Path(config.paths.output_dir).mkdir(exist_ok=True, parents=True)

    # Get dataframes with relevant information
    train_df, category_to_index = get_cohort_df(
        slide_table=config.paths.data_table, target_label=config.data.target_label
    )
    with open(Path(config.paths.output_dir) / "label_str_to_idx.json", "w") as f:
        json.dump(category_to_index, f)
    # Save information about units in the train set
    train_df.to_csv(Path(config.paths.output_dir) / "train.csv", index=False)
    log_df_info(train_df, config)

    verify_data_exists(train_df, config.other.disable_TQDM)

    if config.paths.temp_dir is not None and config.paths.temp_dir != "":
        train_df = move_data_to_temp(
            train_df,
            config.paths.temp_dir,
            config.other.disable_TQDM,
            verify=config.paths.verify_temp,
        )

    info: Dict[str, Any] = {
        "description": "MIL training",
        "train_slide": str(Path(config.paths.data_table).absolute()),
        "target_label": str(config.data.target_label),
        "output_dir": str(Path(config.paths.output_dir).absolute()),
        "datetime": datetime.now().astimezone().isoformat(),
        "category_to_index": category_to_index,
    }

    # Save json info
    with open(Path(config.paths.output_dir) / "info.json", "w") as f:
        json.dump(info, f)

    train_dl = get_dataloader(
        train_df,
        config,
        shuffle=True,
        is_train=True,
    )

    plot_and_save_num_tile_distribution(train_dl, Path(config.paths.output_dir))

    # Train a model on the data and return it alongside the predictions made on the
    train(train_dl, config, Path(config.paths.output_dir))


if __name__ == "__main__":

    run_experiment()
