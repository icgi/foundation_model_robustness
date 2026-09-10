import sys
from pathlib import Path

from tqdm import tqdm
from loguru import logger

from src.conf.train_schema import AppCfg as TrainCfg
from src.conf.inference_schema import AppCfg as InferenceCfg

_configured = False


def get_logger(
    config: TrainCfg | InferenceCfg,
    local_rank: int = 0,
    use_ddp: bool = False,
    log_name="log.log",
):
    global _configured
    if _configured:
        return logger

    logger.remove()
    if local_rank > 0 and use_ddp:
        _configured = True
        return logger
    logger.add(
        Path(config.paths.output_dir) / log_name,
        format=config.logging.format,
        colorize=config.logging.colorize,
        diagnose=config.logging.diagnose,
        level=config.logging.level.upper(),
    )

    if config.other.disable_TQDM:
        logger.add(
            sys.stdout,
            format=config.logging.format,
            colorize=config.logging.colorize,
            diagnose=config.logging.diagnose,
            level=config.logging.sysout_level.upper(),
        )
    else:
        logger.add(
            lambda msg: tqdm.write(msg, end=""),
            format=config.logging.format,
            colorize=config.logging.colorize,
            diagnose=config.logging.diagnose,
            level=config.logging.sysout_level.upper(),
        )

    _configured = True
    return logger