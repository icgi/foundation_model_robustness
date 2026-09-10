from pathlib import Path
import time
from typing import Tuple, List, Dict, Set
import h5py
import torch
import pandas as pd
from loguru import logger
from src.conf.inference_schema import AppCfg as InferenceCfg
import json
from src.network.get_model import get_model

def get_model_weight_path_from_best_checkpoint(
    best_checkpoint: Dict[str, str],):
    if "model_path" in best_checkpoint:
       return [Path(best_checkpoint["model_path"])]
    else:
        print(best_checkpoint["best_checkpoint"])
        inference_path = Path(best_checkpoint["best_checkpoint"])
        base_model_path = inference_path.parent.parent / "models" / "checkpoints"
        print(f"Looking for model weights in {base_model_path}")
        return [base_model_path / (inference_path.name.split('_')[-2] + ".pth")]



def get_model_weight_paths(config: InferenceCfg) -> List[Path]:
    if Path(config.paths.model_path).is_file():
        model_weight_paths = [Path(config.paths.model_path)]
    else:
        model_weight_paths = list(Path(config.paths.model_path).rglob("*.pth"))
        if len(model_weight_paths) == 0:
            raise ValueError(
                f"Didn't find any '.pth' models in folder {config.paths.model_path}"
            )

        if config.paths.tuning_json:
            if not Path(config.paths.tuning_json).exists():
                raise ValueError(f"{config.paths.tuning_json} doesn't exist")
            if not Path(config.paths.tuning_json).suffix == ".json":
                raise ValueError(f"Expected {config.paths.tuning_json} to be a json.")
            with open(config.paths.tuning_json, "r") as f:
                best_checkpoint = json.load(f)
            model_weight_paths = get_model_weight_path_from_best_checkpoint(best_checkpoint)
            if not model_weight_paths[0].exists():
                raise ValueError(
                    f"Tuning points to model {model_weight_paths[0]} but it does not exist."
                )
            logger.info(
                f"Only running inference for the best model from tuning: {model_weight_paths[0].name}"
            )
    if len(model_weight_paths) == 0:
        raise ValueError(f"No model weights found in {config.paths.model_path}")

    return model_weight_paths


def load_models_from_weight_paths(
    model_weight_paths: List[Path],
    num_features: int,
    config: InferenceCfg,
    train_config,
) -> Dict[Path, torch.nn.Module]:
    s = time.time()
    models = {}
    for model_path in model_weight_paths:
        model = get_model(
            num_features, train_config, config.inference.device, compile=False
        )
        model.load_state_dict(
            torch.load(model_path, weights_only=True, map_location="cpu")
        )
        model.to(config.inference.device)
        model.eval()
        models[model_path] = model
    logger.info(f"Loaded {len(models)} models in {time.time()-s:.2f} seconds")
    return models


def load_data(row: pd.Series, device):
    path = row["path"]
    if not Path(path).exists():
        raise ValueError(f"Path {path} does not exist")
    with h5py.File(path, "r") as f:
        features = torch.from_numpy(f["features"][:]).to(device)
        tile_names = f["tile_names"][:]
    size = torch.tensor(features.shape[0]).to(device)
    return features.unsqueeze(0), size.unsqueeze(0), tile_names, path


def get_output_for_df(inference_config: InferenceCfg, model_weight_path: Path):
    return (
        Path(inference_config.paths.output_dir)
        / f"{model_weight_path.stem}_predictions.csv"
    )
