from pathlib import Path
from typing import Tuple, Union, Sequence, Dict, List, Optional

import h5py
import torch
import pandas as pd
from loguru import logger
from dataclasses import dataclass


@dataclass
class Scan:
    name: str
    path: Path
    scanner: str
    features: Optional[torch.Tensor]
    num_tiles: int
    label: int
    out_out_bounds_tensor: torch.Tensor
    tile_ids: torch.Tensor
    coords: torch.Tensor
    tile_names: Optional[List[str]]

    def __post_init__(self):
        # Sort all list based on tile_ids
        sorted_indices = torch.argsort(self.tile_ids)
        if self.features is not None:
            self.features = self.features[sorted_indices]

        self.out_out_bounds_tensor = self.out_out_bounds_tensor[sorted_indices]
        self.tile_ids = self.tile_ids[sorted_indices]
        self.coords = self.coords[sorted_indices]
        if self.tile_names is not None:
            self.tile_names = [self.tile_names[i] for i in sorted_indices.tolist()]
        else:
            self.tile_names = None
        self.possible_indicies = torch.arange(self.num_tiles)

    def remove_out_of_bounds(self, out_of_bounds_tensor: torch.Tensor) -> None:
        in_bounds_mask = ~out_of_bounds_tensor.bool()
        if self.features is not None:
            self.features = self.features[in_bounds_mask]
        self.possible_indicies = self.possible_indicies[in_bounds_mask]
        self.out_out_bounds_tensor = self.out_out_bounds_tensor[in_bounds_mask]
        self.tile_ids = self.tile_ids[in_bounds_mask]
        self.coords = self.coords[in_bounds_mask]
        if self.tile_names is not None:
            self.tile_names = [
                name
                for i, name in enumerate(self.tile_names)
                if in_bounds_mask[i].item()
            ]
        self.num_tiles = len(self.possible_indicies)


@dataclass
class Slide:
    name: str
    scans: Dict[str, Scan]

    def remove_out_of_bounds(self) -> None:
        scans = list(self.scans.values())

        out_of_bounds_total_mask = torch.zeros(
            scans[0].out_out_bounds_tensor.shape, dtype=torch.bool
        )
        for scan in self.scans.values():
            out_of_bounds_scan_mask = scan.out_out_bounds_tensor
            if not out_of_bounds_scan_mask.shape == out_of_bounds_total_mask.shape:
                raise ValueError(
                    f"Shapes are not matching, expected {out_of_bounds_total_mask.shape} but found {out_of_bounds_scan_mask.shape} for slide {self.name} and scanner {scan.scanner}\nThis is most likely due to the scans not actually being paired"
                )

            out_of_bounds_total_mask = torch.logical_or(
                out_of_bounds_total_mask, out_of_bounds_scan_mask
            )

        if out_of_bounds_total_mask.sum().item() > 0:
            logger.info(
                f"Out of bounds tiles for scan {self.name}: {out_of_bounds_total_mask.sum().item()}",
                "Percentage:",
                f"{round(100*(out_of_bounds_total_mask.sum().item() / out_of_bounds_total_mask.shape[0]), 2)}%",
            )

        for scan in self.scans.values():
            scan.remove_out_of_bounds(out_of_bounds_total_mask)

    def verify(self) -> None:
        for scan in self.scans.values():
            if not len(scan.tile_ids.tolist()) == len(set(scan.tile_ids.tolist())):  # type: ignore
                raise RuntimeError(
                    f"NB! TILE IDS ARE NOT UNIQUE!!! for {scan.path}"
                    + f" Total tile ids: {len(scan.tile_ids)} vs. unique tile ids: {len(set(scan.tile_ids.tolist()))} "
                    + "In order to use paired scans, tiles must be sorted by tile ids."
                    + " In order to do this deterministically, all tile ids must be unique."
                    + " If you see this error, it's likely that you have merged .h5's from a single patient."
                    + " If so, you should assure they're merged in the same order"
                    + " and then add a constant offset to the tile ids in each .h5 to make them unique."
                )

        # Check that tileids match
        tile_ids_list = [scan.tile_ids for scan in self.scans.values()]
        first_tile_ids = tile_ids_list[0]
        for other_tile_ids in tile_ids_list[1:]:
            if not torch.equal(first_tile_ids, other_tile_ids):
                raise ValueError(
                    f"Tile ids do not match for {self.name} with scans {[scan.scanner for scan in self.scans.values()]}. Sample: {first_tile_ids[:10]} vs. {other_tile_ids[:10]}"
                )

        # Check that possible indicies match
        possible_indicies_list = [
            scan.possible_indicies for scan in self.scans.values()
        ]
        first_possible_indicies = possible_indicies_list[0]
        for other_possible_indicies in possible_indicies_list[1:]:
            if not torch.equal(first_possible_indicies, other_possible_indicies):
                raise ValueError(
                    f"Possible indicies do not match for {self.name} with scans {[scan.scanner for scan in self.scans.values()]}. Sample: {first_possible_indicies[:10]} vs. {other_possible_indicies[:10]}"
                )

        # Check that labels match
        labels = [scan.label for scan in self.scans.values()]
        first_label = labels[0]
        for other_label in labels[1:]:
            if not first_label == other_label:
                raise ValueError(
                    f"Labels do not match for {self.name} with scans {[scan.scanner for scan in self.scans.values()]}"
                )
