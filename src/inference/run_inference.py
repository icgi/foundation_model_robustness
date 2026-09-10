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

# torch.set_float32_matmul_precision("high")
torch.backends.cudnn.conv.fp32_precision = "tf32"


def get_model_prediction(
    row: pd.Series,
    target_label: str,
    use_amp: bool,
    index_to_label: Dict[int, str],
    model: nn.Module,
    features: torch.Tensor,
    size: torch.Tensor,
    model_path: Path,
) -> pd.Series:
    """Helper method for predicting, and merging predictions with original data."""

    row_out = row.copy(deep=True)
    with torch.inference_mode():
        with autocast(
            dtype=torch.float16, device_type=str(features.device), enabled=use_amp
        ):
            logits = model(features, size).squeeze()
    # Softmax in order to get the probabilities from the logits
    unit_probs = F.softmax(logits, dim=0)

    # Merge row with preds
    for idx, label in index_to_label.items():
        row_out[f"{target_label}_{label}"] = unit_probs[idx].item()
        row_out[f"{target_label}_{label}_logits"] = logits[idx].item()

    argmax_labels = logits.argmax(0).item()
    row_out["predicted_label"] = index_to_label[argmax_labels]
    row_out["predicted_idx"] = argmax_labels
    row_out["model_used"] = model_path

    prob_cols = [
        f"{target_label}_{index_to_label[i]}" for i in range(len(index_to_label))
    ]
    logit_cols = [col + "_logits" for col in prob_cols]
    meta_cols = ["predicted_label", "predicted_idx", "model_used"]
    # put these first, keep all other original fields after
    front = ["case_id"] + prob_cols + logit_cols + meta_cols
    rest = [c for c in row_out.index if c not in set(front)]
    row_out = row_out.reindex(front + rest)
    return row_out


def run_inference_for_df(
    row: pd.Series,
    config: InferenceCfg,
    models: Dict[Path, torch.nn.Module],
    index_to_category: Dict[int, str],
    target_label: str
):
    model_path_to_row = {}

    features, size, _, _ = load_data(row, config.inference.device)

    for model_path, model in tqdm(
        models.items(),
        desc="Running inference for all models",
        leave=False,
        position=1,
        dynamic_ncols=True,
        disable=config.other.disable_TQDM,
    ):

        row_with_pred = get_model_prediction(
            row,
            target_label,
            config.inference.use_amp,
            index_to_category,
            model,
            features,
            size,
            model_path,
        )

        model_path_to_row[model_path] = row_with_pred

    # Release the memory from the gpu
    try:
        del features
        del size
        torch.cuda.empty_cache()
    except:
        pass
    return model_path_to_row


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

    # case_id, project, patient, slide, scanner, scan_name, outcome_lnm (aka label), path, augmentation
    df, category_to_index = get_cohort_df(
        slide_table=inference_config.paths.data_table,
        target_label=train_config.data.target_label,
    )
    try:
        with open(
            model_weight_paths[0].parent.parent.parent / "label_str_to_idx.json", "r"
        ) as f:
            category_to_index = json.load(f)
    except FileNotFoundError:
        with open(model_weight_paths[0].parent.parent.parent / "info.json", "r") as f:
            category_to_index = json.load(f)["category_to_index"]
    index_to_category = {v: k for k, v in category_to_index.items()}

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

    model_path_to_rows = {model_path: [] for model_path in model_weight_paths}
    for split_idx, row in tqdm(
        df.iterrows(),
        desc="Running inference for cases",
        position=0,
        dynamic_ncols=True,
        leave=False,
        disable=inference_config.other.disable_TQDM,
    ):
        s_case = time.time()

        model_path_to_row = run_inference_for_df(
            row,
            inference_config,
            models,
            index_to_category,
            train_config.data.target_label,
        )

        for model_path, row in model_path_to_row.items():
            model_path_to_rows[model_path].append(row.copy(deep=True))

        logger.info(f"({(split_idx) + 1:>5}/{len(df)}) Ran inference for {row['slide']:>10} in {time.time() - s_case:.2f} seconds")  # type: ignore

    # Now save all predictions
    Path(inference_config.paths.output_dir).mkdir(exist_ok=True, parents=True)
    for model_path, rows in model_path_to_rows.items():
        row = pd.DataFrame(rows)
        row.to_csv(
            get_output_for_df(inference_config, model_path),
            index=False,
        )
    case_per_seconds = len(df) / (time.time() - s)
    logger.info(
        f"Finished running inference for cases in {time.time() - s:.2f} seconds ({case_per_seconds:.2f} per second)"
    )


if __name__ == "__main__":
    main()
