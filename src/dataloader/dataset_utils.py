from pathlib import Path

import h5py
import torch
import pandas as pd
from loguru import logger
from .sample_bag_utils import read_h5
from .dataclasses import Scan


def get_out_of_bounds_tensor(path: str, num_tiles: int) -> torch.Tensor:
    try:
        out_of_bounds_tensor: torch.Tensor = read_h5(path, field_name="out_of_bounds")  # type: ignore
    except KeyError:
        logger.warning(
            f"No out_of_bounds field found in {path}, assuming no out of bounds tiles"
        )
        out_of_bounds_tensor = torch.zeros(num_tiles, dtype=torch.bool)
    return out_of_bounds_tensor


def get_scan_from_row(
    row: pd.Series, stream_data: bool, target_label: str, device: str
) -> Scan:
    scanner = row["scanner"]
    path = row["path"]
    label = row[target_label + "_index"]
    slide = row["slide"]

    with h5py.File(path, "r") as f:
        num_tiles = f["features"].shape[0]

    tile_ids = read_h5(path, field_name="tile_ids")
    coords = read_h5(path, field_name="coords")
    tile_names = None  # read_h5(path, field_name="tile_names")

    out_of_bounds_tensor = get_out_of_bounds_tensor(path, num_tiles)

    return Scan(
        slide,
        Path(path),
        scanner,
        None if stream_data else read_h5(path, field_name="features").to(device),
        num_tiles,
        label,
        out_of_bounds_tensor,
        tile_ids,  # type: ignore
        coords,  # type: ignore
        tile_names,  # type: ignore
    )
