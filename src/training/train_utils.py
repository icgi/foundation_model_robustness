from loguru import logger
from src.conf.train_schema import AppCfg
from functools import partial
from src.training.losses import KL_loss, contrastive_loss, cosine_similarity_loss, MSE_loss_score
import torch
from torch import nn
from torch.optim import lr_scheduler
import pandas as pd

def get_score_loss_function(config: AppCfg):
    if config.loss.score_loss_function.lower() == "mse":
        score_loss_fn = partial(
            MSE_loss_score, use_softmax=config.loss.use_softmax_in_mse_score
        )
    elif config.loss.score_loss_function.lower() == "kl_div":
        score_loss_fn = KL_loss
    else:
        raise ValueError(f"{config.loss.score_loss_function} not recognized")
    
    return score_loss_fn

def get_embedding_loss_function(config: AppCfg):
    if config.loss.embedding_loss_function.lower() == "mse":
        embedding_loss_fn = nn.MSELoss()
    elif config.loss.embedding_loss_function.lower() == "contrastive":
        embedding_loss_fn = partial(
            contrastive_loss,
            temperature=config.loss.contrastive_temperature,
            num_negative_samples=config.loss.contrastive_num_negative_samples,
            contrastive_diff_patient_negative=config.loss.contrastive_diff_patient_negative,
            max_tiles_per_scan=config.loss.contrastive_max_tiles_per_scan,
        )
    elif config.loss.embedding_loss_function.lower() == "kl_div":
        embedding_loss_fn = KL_loss
    elif config.loss.embedding_loss_function.lower() == "cosine_similarity":
        embedding_loss_fn = cosine_similarity_loss
    else:
        raise ValueError(f"{config.loss.embedding_loss_function} not recognized")
    return embedding_loss_fn

def get_optimizer(config: AppCfg, model_parameters):
    if config.training.optimizer.lower() == "adamw":
        opt = torch.optim.AdamW(
            model_parameters,
            lr=config.training.max_lr,
        )
    elif config.training.optimizer.lower() == "sgd":
        opt = torch.optim.SGD(
            model_parameters,
            lr=config.training.max_lr,
            weight_decay=config.training.weight_decay,
            momentum=config.training.momentum_sgd,
        )
    else:
        raise ValueError(f"{config.training.optimizer} not recognized or implemented")
    
    return opt


def get_lr_scheduler(config: AppCfg, opt: torch.optim.Optimizer):
    if config.training.scheduler == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=config.training.n_steps)
    elif config.training.scheduler == "one_cycle_lr":
        scheduler = lr_scheduler.OneCycleLR(
            opt,
            config.training.max_lr,
            total_steps=config.training.n_steps,
            final_div_factor=5e2,
        )
    else:
        raise ValueError(f"{config.training.scheduler} isn't implemented")
    
    return scheduler


def get_label_weighting(config, train_dl):
    if config.data.use_label_weight and config.data.oversample:
        assert (
            config.data.oversample_by_column != config.data.target_label
        ), "Oversampling and label weighting should not be used if you oversample based on the label you train towards."
    # Get the weighting for the loss
    if config.data.use_label_weight:
        # We loop over all datasets, get their labels as a list, and list comprehension to flatten it
        labels = train_dl.dataset.labels

        counts = pd.Series(labels).value_counts()
        weight = counts.sum() / counts
        weight /= weight.sum()

        unique_labels = sorted(set(labels))
        assert unique_labels == list(
            range(len(unique_labels))
        ), f"Labels must be sorted, contiguous integers starting from 0. {unique_labels}"

        # Now the weight tensor is directly based on labels
        weight = torch.tensor(
            [weight.get(t) for t in unique_labels],
            dtype=torch.float32,
            device=config.training.device,
        )
    else:
        assert (
            config.data.oversample
        ), "No weighting or oversampling, are you sure this is what you want?"
        weight = None

    return weight