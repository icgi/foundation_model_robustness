from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from hydra.core.config_store import ConfigStore


# ────────────────────────── leaf sections ────────────────────────────
@dataclass
class InferenceCfg:
    use_amp: bool
    device: str


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


@dataclass
class PathCfg:
    data_table: str
    output_dir: str
    model_path: str
    train_conf: Optional[str]
    tuning_json: Optional[str]
    temp_dir: Optional[str]
    verify_temp: bool = False


# ────────────────────────── root section ─────────────────────────────
@dataclass
class AppCfg:
    inference: InferenceCfg
    logging: LoggingCfg
    other: OtherCfg
    paths: PathCfg


# ─────────────────── register with Hydra (once) ─────────────────────
cs = ConfigStore.instance()
cs.store(name="app_cfg", node=AppCfg)
