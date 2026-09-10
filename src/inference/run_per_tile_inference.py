from pathlib import Path
from typing import Dict
import json

from omegaconf import OmegaConf
import h5py
import time
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.amp import autocast
import torch.nn.functional as F
from tqdm import tqdm
import hydra
from loguru import logger

from src.utils import move_data_to_temp, get_cohort_df
from src.logging.get_logger import get_logger
from src.inference.utils import (
    get_model_weight_paths,
    load_models_from_weight_paths,
    load_data,
    get_output_for_df,
)
from src.conf.inference_schema import AppCfg as InferenceCfg
from src.conf.train_schema import AppCfg as TrainCfg

torch.backends.cudnn.conv.fp32_precision = "tf32"

def get_per_tile_scores_prediction(
    target_label: str,
    use_amp: bool,
    index_to_category: Dict[int, str],
    features: torch.Tensor,
    model: nn.Module,
) -> np.ndarray:
    """Helper method for getting per-tile scores for attention-based MIL models."""
    with torch.inference_mode():
        with autocast(
            dtype=torch.float16, device_type=str(features.device), enabled=use_amp
        ):
            scores, attention_scores = model.get_per_tile_score_attn(features)
    
    return scores.cpu().numpy().squeeze(), attention_scores.cpu().numpy().squeeze()


def run_inference_for_df(
    row: pd.Series,
    config: InferenceCfg,
    models: Dict[Path, torch.nn.Module],
 ):
    model_path_to_scores_attn_tile_name_path = {}

    features, size, tile_names, path = load_data(row, config.inference.device)

    for model_path, model in tqdm(
        models.items(),
        desc="Running inference for all models",
        leave=False,
        position=1,
        dynamic_ncols=True,
        disable=config.other.disable_TQDM,
    ):

        scores, attention = get_per_tile_scores_prediction(
            config.inference.use_amp,
            features,
            model,
        )

        model_path_to_scores_attn_tile_name_path[model_path] = {
            "scores": scores,
            "attention": attention,
            "tile_names": tile_names,
            "path": path,}

    # Release the memory from the gpu
    try:
        del features
        del size
        torch.cuda.empty_cache()
    except:
        pass
    return model_path_to_scores_attn_tile_name_path


@logger.catch(reraise=True)
@hydra.main(
    version_base=None, config_path="../../conf", config_name="default_inference_config"
)
def main(inference_config: InferenceCfg):

    logger = get_logger(inference_config, log_name="inference.log")
    model_weight_paths = get_model_weight_paths(inference_config)

    if not inference_config.other.force:
        # Rewrite this to use list comprehension
        num_model_weight_paths_before = len(model_weight_paths)
        model_weight_paths = [
            model_weight_path
            for model_weight_path in model_weight_paths
            if not get_output_for_df(inference_config, model_weight_path).exists()
        ]
        if len(model_weight_paths) == 0:
            logger.info(
                "All model outputs already exist, nothing to do. Use 'force: true' to override."
            )
            return
        logger.info(
            f"Force is disabled. After filtering out already existing outputs, {len(model_weight_paths)}/{num_model_weight_paths_before} models remain for inference"
        )

    with open(
        Path(inference_config.paths.output_dir) / "inference_config.yaml", "w"
    ) as f:
        OmegaConf.save(inference_config, f)

    if inference_config.paths.train_conf is not None:
        train_config: TrainCfg = OmegaConf.load(inference_config.paths.train_conf)  # type: ignore
    else:
        train_conf = model_weight_paths[0].parent.parent.parent / "config.yaml"
        if not train_conf.exists():
            raise FileNotFoundError(
                f"Can't inference location of training config, please specify paths.train_conf in the config."
            )
        train_config: TrainCfg = OmegaConf.load(train_conf)  # type: ignore

    df, category_to_index = get_cohort_df(
        slide_table=inference_config.paths.data_table,
        target_label=train_config.data.target_label,
    )

    if inference_config.paths.temp_dir is not None:
        df = move_data_to_temp(
            df,
            inference_config.paths.temp_dir,
            verify=inference_config.paths.verify_temp,
        )

    # Load the model
    with h5py.File(df["path"].values[0], "r") as f:
        num_features = f["features"].shape[-1]

    s = time.time()
    # We just run each case one by one...
    models = load_models_from_weight_paths(
        model_weight_paths, num_features, inference_config, train_config
    )

    model_path_to_scores_attn_tile_name_paths = {model_path: [] for model_path in model_weight_paths}


    for split_idx, row in tqdm(
        df.iterrows(),
        desc="Running inference for cases",
        position=0,
        dynamic_ncols=True,
        leave=False,
        disable=inference_config.other.disable_TQDM,
    ):
        s_case = time.time()

        model_path_to_scores_attn_tile_name_path = run_inference_for_df(
            row,
            inference_config,
            models,
            index_to_category,
            train_config.data.target_label,
        )

        for model_path, row in model_path_to_scores_attn_tile_name_path.items():
            model_path_to_scores_attn_tile_name_paths[model_path].append(row)

        logger.info(f"({(split_idx) + 1:>5}/{len(df)}) Ran inference for {row['slide']:>10} in {time.time() - s_case:.2f} seconds")  # type: ignore

    # Now save all predictions
    Path(inference_config.paths.output_dir).mkdir(exist_ok=True, parents=True)
    for model_path, vals in model_path_to_scores_attn_tile_name_paths.items():
        scores = np.array([v["scores"] for v in vals])
        attention = np.array([v["attention"] for v in vals])
        tile_names = np.array([v["tile_names"] for v in vals])
        paths = np.array([v["path"] for v in vals])
        import h5py
        with h5py.File(get_output_for_df(inference_config, model_path), "w") as f:
            f.create_dataset("scores", data=scores)
            f.create_dataset("attention", data=attention)
            f.create_dataset("tile_names", data=tile_names.astype("S"))
            f.create_dataset("paths", data=paths.astype("S"))
        
    case_per_seconds = len(df) / (time.time() - s)
    logger.info(
        f"Finished running inference for cases in {time.time() - s:.2f} seconds ({case_per_seconds:.2f} per second)"
    )


if __name__ == "__main__":
    main()
