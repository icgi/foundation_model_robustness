from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from hydra.core.config_store import ConfigStore


# ────────────────────────── leaf sections ────────────────────────────
@dataclass
class TrainingCfg:
    n_steps: int
    batch_size: int
    weight_decay: float
    momentum_sgd: float
    optimizer: str
    max_lr: float
    grad_norm_clip: int
    use_amp: bool
    checkpoint_interval: int
    train_max_bag_size: int
    device: str
    scheduler: Optional[str] = None


@dataclass
class ModelCfg:
    embedding_size: int
    encoder_layers: int
    num_classes: int
    dropout_head: float
    use_batchnorm: bool
    mil_model: str
    model_load_checkpoint: Optional[str]


@dataclass
class LossCfg:
    score_loss_function: str
    score_loss_weight: int
    use_softmax_in_mse_score: bool
    embedding_loss_weight: int
    embedding_loss_function: str
    contrastive_temperature: float
    contrastive_num_negative_samples: int
    contrastive_diff_patient_negative: bool
    contrastive_max_tiles_per_scan: Optional[int] = None  # None = use all tiles


@dataclass
class DataCfg:
    use_label_weight: bool
    oversample: bool
    oversample_by_column: str
    pair_scans: bool
    pair_tiles: bool
    stream_data: bool
    target_label: str
    remove_out_of_bounds: bool


@dataclass
class LoggingCfg:
    level: str
    sysout_level: str
    format: str
    colorize: bool
    backtrace: bool
    diagnose: bool


@dataclass
class OtherCfg:
    disable_TQDM: bool
    force: bool
    seed: Optional[int] = None


@dataclass
class DataloaderCfg:
    persistent_workers: bool
    pin_memory: bool
    num_workers: int
    prefetch_factor: int
    store_on_gpu: bool
    store_type: str


@dataclass
class PathCfg:
    data_table: str
    output_dir: str
    temp_dir: Optional[str]
    verify_temp: bool = False


# ────────────────────────── root section ─────────────────────────────
@dataclass
class AppCfg:
    training: TrainingCfg
    model: ModelCfg
    data: DataCfg
    dataloader: DataloaderCfg
    logging: LoggingCfg
    other: OtherCfg
    loss: LossCfg
    paths: PathCfg


# ─────────────────── register with Hydra (once) ─────────────────────
cs = ConfigStore.instance()
cs.store(name="app_cfg", node=AppCfg)
