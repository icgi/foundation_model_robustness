import random
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple, List, Dict, Any

import h5py
import torch
import pandas as pd
from tqdm import tqdm
from loguru import logger
from torch.utils.data import Dataset

from src.conf.train_schema import AppCfg
from .dataset_utils import get_scan_from_row
from .sample_bag_utils import read_h5, sample_bag_streamed, sample_bag
from .dataclasses import Scan, Slide


def validate_config_temporary(config: AppCfg):
    if (
        config.loss.embedding_loss_weight > 0.0
        and not config.data.pair_tiles
        and config.data.pair_scans
    ):
        raise ValueError(
            "Embedding loss requires paired tiles. Please set pair_tiles to True in the config."
        )
    if config.data.pair_tiles and not config.data.pair_scans:
        raise ValueError(
            "pair_tiles is set to True, but pair_scans is False. Please set pair_scans to True in the config, as this configuration is doesn't make sense."
        )
    if config.data.remove_out_of_bounds and not config.data.pair_tiles:
        raise ValueError(
            "remove_out_of_bounds is set to True, but pair_tiles is False. Please set pair_tiles to True in the config, as this configuration is doesn't make sense."
        )


class BagDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        config: AppCfg,
        device: str,
        max_bag_size: Optional[int] = None,
        return_tile_sizes: bool = False,
    ) -> None:
        super().__init__()
        self.df = dataframe
        self.config = config
        self._validate_df(self.df)
        # TODO: Move these checks to config validation
        validate_config_temporary(self.config)
        if config.data.pair_scans:
            if not self._check_all_paired(self.df):
                raise ValueError(
                    f"Not all slides have paired scans, which is required for paired training. Update your input list to only have paired scans"
                )
            logger.info("Using paired scans for training")
        else:
            logger.info("Using single scans for training")

        self.max_bag_size = max_bag_size
        self.device = device
        self.return_tile_names = return_tile_sizes  # Set this to True when testing
        if self.return_tile_names:
            logger.warning(
                "Returning tile names from dataset. This does not work nicely with batching. Only use for testing"
            )

        self.load_data()
        self._update_max_num_tiles()
        logger.info(f"Max number of tiles is {self.max_num_tiles}")

        # TODO: This shouldn't really be here, but instead with the other thing in the src/utils.py...
        self.log_scanner_per_slide_distribution(self.df)

        self.return_fields = [
            "features",
            "tile_ids",
        ]  # ["features", "coords", "tile_ids"]

        if self.return_tile_names:
            self.return_fields.append("tile_names")

    @property
    def bag_size(self) -> int:
        if self.max_bag_size is not None:
            return self.max_bag_size
        else:
            return self.max_num_tiles

    @property
    def labels(self):
        labels = []
        for slide in self.slides:
            for scan in slide.scans.values():
                labels.append(scan.label)
                break  # Only need one label per slide
        return labels

    def get_feature_vector_size(self):
        with h5py.File(self.df["path"].values[0], "r") as f:
            return f["features"].shape[-1]

    def get_num_labels(self):
        return self.df[self.config.data.target_label].nunique()

    def _validate_df(self, df: pd.DataFrame) -> None:
        """Validates that the dataframe has the correct format."""
        # TODO: Add more checks here
        required_columns = {
            "slide",
            "scanner",
            "path",
            self.config.data.target_label,
            self.config.data.target_label + "_index",
        }
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        if len(df) != df["path"].nunique():
            raise ValueError(
                (
                    "DataFrame has duplicate paths (i.e. datapoints). If this is intended to use oversampling, there is no support."
                    + " Please set the oversample flag in the config to achieve this."
                )
            )

    def log_num_scans_per_scanner(self, df: pd.DataFrame):
        """Get a mapping from scanner to number of scans."""
        num_scanners_to_count = defaultdict(int)
        for _, row in df.iterrows():
            scanner = row["scanner"]
            num_scanners_to_count[scanner] += 1
        for value, count in num_scanners_to_count.items():
            logger.info(f"There are {count} scans scanned on {value} scanner(s)")

    def log_scanner_per_slide_distribution(self, df: pd.DataFrame):
        scanner_distribution = defaultdict(int)
        for slide in df["slide"].unique():
            n_scanners = df[df["slide"] == slide].shape[0]
            scanner_distribution[n_scanners] += 1

        logger.info("Scanners per slide distribution:")
        for n_scanners, num in scanner_distribution.items():
            logger.info(f"Slides withs {n_scanners} scans: {num}")

    def _check_all_paired(self, df: pd.DataFrame) -> bool:
        """Check if all slides have paired scans."""
        for slide in df["slide"].unique():
            slide_df = df[df["slide"] == slide]
            if len(slide_df) == 1:
                return False
            elif not slide_df["scanner"].nunique() > 1:
                raise ValueError(
                    f"Found duplicate scanner: {slide_df}. This is not supposed to happen"
                )

        return True

    def _get_slide_to_scans_map(self, df: pd.DataFrame) -> Dict[str, List[pd.Series]]:
        """Get a mapping from slide_id to list of scanners."""
        slide_to_scanners: Dict[str, List[pd.Series]] = defaultdict(list)

        for i, row in df.iterrows():
            slide_id = row["slide"]
            slide_to_scanners[slide_id].append(row)

        return slide_to_scanners

    def load_data(self) -> None:
        slide_to_row_idxs = self._get_slide_to_scans_map(self.df)
        self.slides: List[Slide] = []

        for slide, scans in tqdm(
            slide_to_row_idxs.items(),
            desc="Reading file information",
            leave=False,
            dynamic_ncols=True,
            disable=self.config.other.disable_TQDM,
        ):
            scanner_to_scan = {}
            for row in scans:
                scan = get_scan_from_row(
                    row,
                    self.config.data.stream_data,
                    self.config.data.target_label,
                    self.device,
                )
                scanner_to_scan[row["scanner"]] = scan
            slide = Slide(slide, scanner_to_scan)
            self.slides.append(slide)

            if self.config.data.remove_out_of_bounds:
                slide.remove_out_of_bounds()
            if self.config.data.pair_tiles:
                slide.verify()

    def _update_max_num_tiles(self):
        self.max_num_tiles = 0
        for slide in self.slides:
            for scan in slide.scans.values():
                self.max_num_tiles = max(self.max_num_tiles, scan.num_tiles)

    def _read_scan_from_scan(self, scan: Scan, permutation: torch.Tensor):
        inp: Dict[str, Any] = {
            "coords": scan.coords,
            "tile_ids": scan.tile_ids,
        }
        if self.return_tile_names:
            inp["tile_names"] = scan.tile_names
        if self.config.data.stream_data:
            # TODO: THIS REALLY NEEDS TO JUST FIX THE SCAN DUE TO RETURNING COORDS AND TILE IDS AND SUCH
            # NOTE: I don't know what I meant with the above comment...

            output = sample_bag_streamed(
                inp,
                self.bag_size,
                scan,
                permutation,
            )

        else:
            inp["features"] = scan.features
            output = sample_bag(inp, self.bag_size, permutation, scan.possible_indicies)

        return output

    def read_one_scan(self, scan: Scan, permutation: torch.Tensor) -> Dict[str, Any]:
        output = self._read_scan_from_scan(scan, permutation)

        info = {
            "coords": output["coords"],
            "tile_ids": output["tile_ids"],
            "path": str(scan.path),
            "scanner": scan.scanner,
        }
        if self.return_tile_names:
            info["tile_names"] = output["tile_names"]

        return {
            "features": output["features"].float(),
            "size": output["size"],
            "label": scan.label,
            "info": info,
        }

    def get_unpaired_data(self, slide: Slide):
        scanners = list(slide.scans.keys())
        scanner = random.sample(scanners, 1)[0]
        scan = slide.scans[scanner]
        permutation = torch.randperm(scan.num_tiles)
        return self.read_one_scan(scan, permutation)

    def get_paired_data(self, slide: Slide):
        scanners = list(slide.scans.keys())
        scanner_1, scanner_2 = random.sample(scanners, 2)
        # We make a random permutation in order to sample later
        permutation_scanner_1 = torch.randperm(slide.scans[scanner_1].num_tiles)
        if self.config.data.pair_tiles:
            permutation_scanner_2 = permutation_scanner_1
        else:
            permutation_scanner_2 = torch.randperm(slide.scans[scanner_2].num_tiles)

        scanner_1_data = self.read_one_scan(
            slide.scans[scanner_1], permutation_scanner_1
        )
        scanner_2_data = self.read_one_scan(
            slide.scans[scanner_2], permutation_scanner_2
        )

        if not scanner_1_data["label"] == scanner_2_data["label"]:
            raise ValueError(
                f"Label doesn't match for paired scanners for slide {slide}... {scanner_1_data['label']} vs. {scanner_2_data['label']}"
            )
        if not slide.scans[scanner_1].name == slide.scans[scanner_2].name:
            raise ValueError(
                f"Scan names doesn't match for paired scanners for slide {slide}... {slide.scans[scanner_1].name} vs. {slide.scans[scanner_2].name}"
            )

        if self.config.data.pair_tiles:
            if not (
                scanner_1_data["info"]["tile_ids"] == scanner_2_data["info"]["tile_ids"]
            ).all():
                raise ValueError(
                    f"Tile ids doesn't match for paired scanners for slide {slide.name}... {scanner_1_data['info']['tile_ids']} vs. {scanner_2_data['info']['tile_ids']}"
                )

            num_inequal = sum(
                [
                    tile_id_1 != tile_id_2
                    for tile_id_1, tile_id_2 in zip(
                        scanner_1_data["info"]["tile_ids"],
                        scanner_2_data["info"]["tile_ids"],
                    )
                ]
            )
            if not num_inequal == 0:
                raise ValueError(
                    f"Tile ids are not matching for paired scanners for slide {slide}... Found {num_inequal} non matching tile ids"
                )
        return {"scanner_1": scanner_1_data, "scanner_2": scanner_2_data}

    def __getitem__(
        self, index: int
    ) -> (
        Tuple[torch.Tensor, torch.Tensor, int, Dict[str, Any]]
        | Dict[str, Dict[str, Any]]
    ):
        # Update the max number of tiles
        slide = self.slides[index]

        if self.config.data.pair_scans:
            data = self.get_paired_data(slide)
        else:
            data = self.get_unpaired_data(slide)

        return data

    def __len__(self):
        return len(self.slides)
