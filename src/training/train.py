import time
from typing import Dict, Any, Tuple, List
from pathlib import Path
from functools import partial

import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from loguru import logger
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from sklearn.metrics import balanced_accuracy_score
from torch.nn.utils.clip_grad import clip_grad_norm_

from src.network.get_model import get_model
from src.network.MIL_models.base_model import MILModel
from src.utils import plot_and_save_losses
from src.conf.train_schema import AppCfg
from src.training.train_utils import get_score_loss_function, get_embedding_loss_function, get_optimizer, get_lr_scheduler, get_label_weighting

# Import _Loss class
from torch.nn.modules.loss import _Loss

rng = np.random.default_rng()


def get_model_prediction_unpaired(
    model: MILModel,
    data: Dict[str, Any],
    device: torch.device,
    pred_loss_func: _Loss,
    use_amp: bool,
):
    features = data["features"].to(device)
    sizes = data["size"].to(device)
    labels = data["label"].to(device)

    with autocast(enabled=use_amp, device_type=str(features.device)):
        predictions = model(
            features,
            sizes,
            return_embeddings=False,
        )
        pred_loss = pred_loss_func(predictions, labels)

    return predictions, pred_loss, labels


def get_model_prediction_paired(
    model: MILModel,
    data: Dict[str, Any],
    device: torch.device,
    scanners: List[str],  # Take this in to ensure order is preserved
    pred_loss_func: _Loss,
    use_amp: bool,
) -> Tuple[
    torch.Tensor,
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    features = torch.cat([data[scanner]["features"] for scanner in scanners], dim=0).to(
        device
    )
    sizes = torch.cat([data[scanner]["size"] for scanner in scanners], dim=0).to(device)
    labels = torch.cat([data[scanner]["label"] for scanner in scanners], dim=0).to(
        device
    )

    with autocast(enabled=use_amp, device_type=str(features.device)):
        predictions, embeddings = model(
            features,
            sizes,
            return_embeddings=True,
        )
        pred_loss = pred_loss_func(predictions, labels)

    scanner_to_embeddings = {}
    scanner_to_predictions = {}
    for i, scanner in enumerate(scanners):  # type: ignore
        # First half belongs to scanner 1, second half to scanner 2
        slice_start = i * data[scanner]["features"].shape[0]
        slice_end = (i + 1) * data[scanner]["features"].shape[0]
        scanner_to_embeddings[scanner] = embeddings[slice_start:slice_end]
        scanner_to_predictions[scanner] = predictions[slice_start:slice_end]

    return predictions, scanner_to_predictions, scanner_to_embeddings, pred_loss, labels


def calculate_losses_paired(
    model: MILModel,
    data: Dict[str, Any],
    device: torch.device,
    pred_loss_func: _Loss,
    embedding_loss_func: _Loss,
    embedding_loss_weight: float,
    score_loss_func: _Loss,
    score_loss_weight: float,
    use_amp: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List, List]:

    scanners = sorted(list(data.keys()))

    predictions, scanner_to_predictions, scanner_to_embeddings, pred_loss, labels = (
        get_model_prediction_paired(
            model, data, device, scanners, pred_loss_func, use_amp
        )
    )

    if embedding_loss_weight > 0:
        embedding_loss = (
            embedding_loss_func(
                scanner_to_embeddings["scanner_1"], scanner_to_embeddings["scanner_2"]
            )
            * embedding_loss_weight
        )
    else:
        embedding_loss = torch.zeros(1, device=device, dtype=torch.float32)
    if score_loss_weight > 0:
        score_loss = (
            score_loss_func(
                scanner_to_predictions["scanner_1"], scanner_to_predictions["scanner_2"]
            )
            * score_loss_weight
        )
    else:
        score_loss = torch.zeros(1, device=device, dtype=torch.float32)

    return (
        pred_loss,
        embedding_loss,
        score_loss,
        predictions.argmax(-1).tolist(),
        labels.tolist(),
    )


def calculate_losses_unpaired(
    model: MILModel,
    data: Dict[str, Any],
    device: torch.device,
    pred_loss_func: _Loss,
    use_amp: bool,
) -> Tuple[torch.Tensor, List, List]:
    preds, pred_loss, labels = get_model_prediction_unpaired(
        model, data, device, pred_loss_func, use_amp
    )

    return pred_loss, preds.argmax(-1).tolist(), labels.tolist()


def train_one_epoch(
    model: MILModel,
    train_dl: DataLoader,
    pred_loss_func: _Loss,
    score_loss_func: _Loss,
    embedding_loss_func: _Loss,
    score_loss_weight: float,
    embedding_loss_weight: float,
    opt: torch.optim.Optimizer,
    scheduler: lr_scheduler._LRScheduler,
    use_amp: bool,
    is_paired: bool,
    grad_norm_clip: float,
    disable_TQDM: bool,
    steps_left: int,
):
    assert steps_left > 0, f"Steps_left must be positive? Got {steps_left}"
    device: torch.device = model.device  # type: ignore
    temp_pred_losses = []
    temp_score_losses = []
    temp_emb_losses = []
    steps = 0

    scaler = None
    if use_amp:
        scaler = GradScaler()

    model.train()
    total_predctions = []
    total_labels = []
    for data in tqdm(
        train_dl,
        desc="Training",
        leave=False,
        position=0,
        dynamic_ncols=True,
        disable=disable_TQDM,
    ):
        
        if steps >= steps_left:
            break

        if is_paired:
            pred_loss, embedding_loss, score_loss, predictions, labels = (
                calculate_losses_paired(
                    model,
                    data,
                    device,
                    pred_loss_func,
                    embedding_loss_func,
                    embedding_loss_weight,
                    score_loss_func,
                    score_loss_weight,
                    use_amp,
                )
            )
            loss = pred_loss + embedding_loss + score_loss
            temp_emb_losses.append(embedding_loss.item())
            temp_score_losses.append(score_loss.item())
        else:
            pred_loss, predictions, labels = calculate_losses_unpaired(
                model,
                data,
                device,
                pred_loss_func,
                use_amp,
            )
            loss = pred_loss

        total_predctions.append(predictions)
        total_labels.append(labels)
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            # Unscale in order to clip
            scaler.unscale_(opt)
            clip_grad_norm_(model.parameters(), grad_norm_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_norm_clip)
            opt.step()

        opt.zero_grad()
        steps += 1
        # Update the learning rate scheduler
        scheduler.step()

        temp_pred_losses.append(pred_loss.item())


    preds = np.concatenate(total_predctions)
    total_labels = np.concatenate(total_labels)
    bac = balanced_accuracy_score(total_labels, preds)
    accuracy = (preds == total_labels).mean()

    return (
        np.mean(temp_pred_losses).item(),
        np.mean(temp_emb_losses).item() if is_paired else 0.0,
        np.mean(temp_score_losses).item() if is_paired else 0.0,
        bac,
        accuracy,
        steps
    )


def train(
    train_dl: DataLoader,
    config: AppCfg,
    path: Path,
):
    device = config.training.device

    # Get/make paths
    checkpoint_folder = path / "models" / "checkpoints"
    checkpoint_folder.mkdir(exist_ok=True, parents=True)

    # Load model
    feature_size = train_dl.dataset.get_feature_vector_size()  # type: ignore
    model = get_model(feature_size, config, device)

    classification_loss_fn = nn.CrossEntropyLoss(
        weight=get_label_weighting(config, train_dl)
    )

    score_loss_fn = get_score_loss_function(config)
    embedding_loss_fn = get_embedding_loss_function(config)
    opt = get_optimizer(config, model.parameters())
    scheduler = get_lr_scheduler(config, opt)


    last_checkpoint = 0
    train_pred_losses = []
    train_score_losses = []
    train_emb_losses = []

    logger.info(f"Datapoints in training set: {len(train_dl.dataset)}")  # type: ignore

    pbar = tqdm(
        total = config.training.n_steps,
        desc="Training model",
        leave=True,
        position=1,
        dynamic_ncols=True,
        disable=config.other.disable_TQDM,
    )

    completed_steps = 0
    while completed_steps < config.training.n_steps:

        model.train()
        start = time.time()

        temp_pred_loss, temp_emb_loss, temp_score_loss, bac, accuracy, update_steps = train_one_epoch(
            model,
            train_dl,
            classification_loss_fn,
            score_loss_fn,  # type: ignore
            embedding_loss_fn,  # type: ignore
            config.loss.score_loss_weight,
            config.loss.embedding_loss_weight,
            opt,
            scheduler,  # type: ignore
            use_amp=config.training.use_amp,
            is_paired=config.data.pair_scans,
            grad_norm_clip=config.training.grad_norm_clip,
            disable_TQDM=config.other.disable_TQDM,
            steps_left = config.training.n_steps - completed_steps,
        )

        if update_steps == 0:
            raise ValueError(f"Updated steps is 0, which shouldn't be possible. Completed steps: {completed_steps} n_steps: {config.training.n_steps}")

        pbar.update(update_steps)
        completed_steps += update_steps

        train_pred_losses.append(temp_pred_loss)
        train_score_losses.append(temp_score_loss)
        train_emb_losses.append(temp_emb_loss)

        # Save checkpoint
        if (
            completed_steps - last_checkpoint
        ) >= config.training.checkpoint_interval or completed_steps >= config.training.n_steps:
            torch.save(
                model.state_dict(), checkpoint_folder / f"model-step-{completed_steps}.pth"
            )
            last_checkpoint = completed_steps

        score_loss_string = (
            f"Score loss: {temp_score_loss:>6.3f} "
            if temp_score_loss is not None and config.loss.score_loss_weight > 0
            else ""
        )
        pair_tile_loss_string = (
            f"Pair tile loss: {temp_emb_loss:>6.3f} "
            if temp_emb_loss is not None and config.loss.embedding_loss_weight > 0
            else ""
        )
        training_metric_string = (
            f"Steps: {completed_steps:>6}/{config.training.n_steps} "
            f"Classification loss: {temp_pred_loss:>6.3f} "
            f"{score_loss_string}"
            f"{pair_tile_loss_string}"
            f"Accuracy: {accuracy:>6.3f} "
            f"BAC: {bac:>6.3f} "
            f"lr: {scheduler.get_last_lr()[0]:3.5f} "
            f"Time: {time.time()-start:>3.2f} seconds"
        )

        logger.info(training_metric_string)


    plot_and_save_losses(
        train_pred_losses, train_score_losses, train_emb_losses, f"{path}/loss_plot.png"
    )
