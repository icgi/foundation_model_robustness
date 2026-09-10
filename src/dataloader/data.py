from typing import Dict, List, Any, Sequence, Mapping

import torch
import pandas as pd
from loguru import logger
from torch.utils.data import DataLoader

from src.conf.train_schema import AppCfg
from src.dataloader.datasets import BagDataset
from src.dataloader.oversampler import BalancedOversampler


def create_data_loader(
    train_ds: BagDataset, config: AppCfg, shuffle: bool, is_train: bool
):
    if not is_train and config.data.oversample:
        raise ValueError(
            "Testing + oversampling is a weird choice. Are you sure you want to do this?."
        )
    if not is_train and shuffle:
        logger.warning(
            "Testing and shuffling certainly a choice. Are you sure you want to do this?"
        )

    data_loader_params = {
        "dataset": train_ds,
        "batch_size": config.training.batch_size,
        "shuffle": shuffle if not config.data.oversample and is_train else False,
        "drop_last": is_train,
        "num_workers": config.dataloader.num_workers,
        "persistent_workers": config.dataloader.persistent_workers,
        "prefetch_factor": (
            config.dataloader.prefetch_factor
            if config.dataloader.prefetch_factor != "none"
            else None
        ),
        "sampler": (
            BalancedOversampler(train_ds)
            if config.data.oversample and is_train
            else None
        ),
    }

    # Conditionally add pin_memory and pin_memory_device
    if config.dataloader.pin_memory:
        data_loader_params["pin_memory"] = True
        #data_loader_params["pin_memory_device"] = str(config.training.device)

    logger.debug(f"Making dataloader with params: {data_loader_params}")

    # Create DataLoader with the specified parameters
    dl = DataLoader(**data_loader_params)

    return dl


def get_dataloader(
    data_df: pd.DataFrame,
    config: AppCfg,
    shuffle: bool,
    is_train: bool,
) -> DataLoader:
    """Returns a train dataloader and a test dataloader"""
    # Create the datasets
    train_ds = BagDataset(
        dataframe=data_df,
        config=config,
        device=(
            torch.device(config.training.device)
            if config.dataloader.store_on_gpu
            else torch.device("cpu")
        ),
        max_bag_size=config.training.train_max_bag_size if is_train else None,
    )

    dl = create_data_loader(train_ds, config, shuffle, is_train)

    return dl
