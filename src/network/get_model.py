import typing as typ

import torch
from torch import nn

from src.conf.train_schema import AppCfg
from src.network.MIL_models.ABMIL import ABMILModel
from src.network.MIL_models.base_model import MILModel
from loguru import logger


def verify_model_checkpoint(
    model_params: typ.Dict, checkpoint_params: typ.Dict
) -> bool:
    """
    Verify that the model parameters match the checkpoint parameters.
    This is a simple check to ensure that the model architecture matches the checkpoint.
    """
    for key in model_params:
        if key not in checkpoint_params:
            logger.warning(f"Key {key} not found in checkpoint parameters.")
            return False
        if model_params[key] != checkpoint_params[key]:
            logger.warning(
                f"Value for key {key} does not match. Model: {model_params[key]}, Checkpoint: {checkpoint_params[key]}"
            )
            return False
    return True


def get_model(num_features: int, config: AppCfg, device, compile=False) -> MILModel:
    if config.model.mil_model == "ABMIL":
        model = ABMILModel(
            n_feats=num_features,
            n_out=config.model.num_classes,
            dropout_head=config.model.dropout_head,
            embedding_size=config.model.embedding_size,
            encoder_layers=config.model.encoder_layers,
            use_batchnorm=config.model.use_batchnorm,
        ).to(device)
    else:
        raise ValueError(f"Model {config.model.mil_model} not implemented")

    if config.model.model_load_checkpoint is not None:
        logger.info(f"Loading model from {config.model.model_load_checkpoint}")
        checkpoint = torch.load(config.model.model_load_checkpoint, map_location="cpu")
        # Verify that shapes match up
        if not verify_model_checkpoint(
            model.state_dict(), checkpoint["model_state_dict"]
        ):
            logger.warning(
                "Model parameters do not match checkpoint parameters. Loading may fail."
            )
        # Do everything on cpu just in scase
        model.to("cpu")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device)
        logger.info("Model loaded")

    if compile:
        model: MILModel = torch.compile(model)  # type: ignore
    return model
