from collections import defaultdict

import torch
import numpy as np
from loguru import logger
from torch.utils.data import Sampler

from src.dataloader.datasets import BagDataset


class BalancedOversampler(Sampler):
    def __init__(self, train_ds: BagDataset):
        value_to_oversample_by = []
        slide_order = (slide.name for slide in train_ds.slides)
        for slide in slide_order:
            if slide not in train_ds.df["slide"].values:
                raise ValueError(
                    f"Slide {slide} not found in dataframe slides {train_ds.df['slide'].values}"
                )
            values = train_ds.df.loc[train_ds.df["slide"] == slide][
                train_ds.config.data.oversample_by_column
            ].values.tolist()
            if len(np.unique(values)) > 1:
                raise ValueError(
                    f"Slide {slide} has multiple values ({np.unique(values)}) for oversample_by_column {train_ds.config.data.oversample_by_column}, cannot oversample. Choose a consistent column."
                )
            value_to_oversample_by += [values[0]]

        self.value_to_oversample_by = value_to_oversample_by
        self.class_indices = self._get_class_indices()
        self.max_class_size = max(
            len(indices) for indices in self.class_indices.values()
        )
        logger.info(
            "Balance before upsampling (Note that these are for paired samples, i.e. usually it's 2x this)"
        )
        for label, indices in self.class_indices.items():
            logger.info(f"\tClass {label}: {len(indices)} samples")

        logger.info(
            f"Will upsample from {len(value_to_oversample_by)} to {self.max_class_size * len(self.class_indices)} samples"
        )

    def _get_class_indices(self):
        class_indices = defaultdict(list)
        for idx in range(len(self.value_to_oversample_by)):
            value = self.value_to_oversample_by[idx]
            if isinstance(value, torch.Tensor):
                value = value.long().item()
            class_indices[value].append(idx)
        return class_indices

    def __iter__(self):
        balanced_indices = []
        for cls, indices in self.class_indices.items():
            # Include all samples
            balanced_class_indices = indices.copy()
            num_samples = len(indices)
            num_extra_samples = self.max_class_size - num_samples
            if num_extra_samples > 0:
                # Randomly oversample to match the max class size
                extra_indices = np.random.choice(
                    indices, size=num_extra_samples, replace=True
                )
                balanced_class_indices.extend(extra_indices)
            balanced_indices.extend(balanced_class_indices)
        # Shuffle the combined indices
        np.random.shuffle(balanced_indices)
        if not len(balanced_indices) == len(self):
            raise ValueError(
                f"Balanced oversampler expected {len(self)} samples, got {len(balanced_indices)}"
            )
        return iter(balanced_indices)

    def __len__(self):
        # plus 3 due to oversampling
        return self.max_class_size * len(self.class_indices)
