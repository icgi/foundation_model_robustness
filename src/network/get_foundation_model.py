import os
from pathlib import Path
from typing import Tuple, List, Any

import timm
import torch
from torch import nn
from huggingface_hub import login
from loguru import logger

from src.conf.train_schema import AppCfg


def login_hf(token: str) -> None:
    """
    Logs into Hugging Face using the provided token.

    Parameters:
        token (str): The authentication token string or the path to a file containing the token.
    """

    @logger.catch(reraise=True)
    def try_login(token) -> None:
        # Check if the token is a path to a file
        if Path(token).is_file():
            with open(token, "r") as f:
                token = f.read().strip()

        # Ensure the token is not empty
        if not token:
            raise ValueError("The authentication token is empty or invalid.")

        # Log in to Hugging Face
        login(token=token, add_to_git_credential=False)

    logger.info("Logging into Huggingface")
    try_login(token)
    logger.info("Successfully logged into huggingface")


def get_normalisation_for_model(model_name: str) -> Tuple[List[float], List[float]]:
    if model_name == "h_optimus_1" or model_name == "h0_mini":
        return ([0.707223, 0.578729, 0.703617], [0.211883, 0.230117, 0.177517])
    elif model_name == "phikon":
        return ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        raise ValueError(f"Couldn't find normalization for model {model_name}")


def get_num_feats_model(config: AppCfg) -> int:
    model_name = config.model.backbone.model_name
    cls_or_patch = config.model.backbone.cls_or_patch
    if model_name == "h_optimus_1" or (
        model_name == "h0_mini" and cls_or_patch == "both"
    ):
        return 1536
    elif model_name == "phikon" or (model_name == "h0_mini" and cls_or_patch != "both"):
        return 768
    else:
        raise ValueError(f"Couldn't find number of features for model {model_name}")


def get_h_optimus_1_model(token: str, model_dir_str: str, **kwargs) -> nn.Module:

    login_hf(token)

    filename = "pytorch_model.bin"
    model_dir = Path(model_dir_str) / "h_optimus_1"
    if not (model_dir / filename).is_file():
        raise FileNotFoundError(f"Couldn't find model at {model_dir / filename}...")

    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-1",
        checkpoint_path=os.path.join(model_dir, filename),
        init_values=1e-5,
        dynamic_img_size=False,
        num_classes=0,
    )

    return model


def get_h0_mini_model(token: str, model_dir_str: str, cls_or_patch: str) -> nn.Module:

    login_hf(token)

    filename = "pytorch_model.bin"
    model_dir = Path(model_dir_str) / "h0_mini"
    file_path = model_dir / filename
    if not file_path.is_file():
        raise FileNotFoundError(f"Couldn't find model at {file_path}...")

    model = timm.create_model(
        "hf-hub:bioptimus/H0-mini",
        checkpoint_path=str(file_path),
        init_values=1e-5,
        dynamic_img_size=False,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        num_classes=0,
    )

    class H0_mini(nn.Module):
        def __init__(self, model, cls_or_patch) -> None:
            super().__init__()
            self.model = model
            # TODO: Make it so that it is possible to chose between cls and cls + patch
            self.cls_or_patch = cls_or_patch
            if not self.cls_or_patch in ["patch", "cls", "both"]:
                raise ValueError(
                    f"Expected cls_or_path to be either 'patch', 'cls', or 'both', but found {self.cls_or_patch}"
                )

        def __call__(self, x, *args: Any, **kwds: Any) -> torch.Tensor:
            output = self.model(x)  # size: 1 x 257 x 1280

            class_token = output[:, 0]  # size: 1 x 1280
            patch_tokens = output[
                :, self.model.num_prefix_tokens :
            ]  # size: 1 x 256 x 768

            if self.cls_or_patch == "patch":
                embedding = patch_tokens.mean(1)
            elif self.cls_or_patch == "cls":
                embedding = class_token
            else:
                embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
            return embedding

        def eval(self):
            self.model.eval()
            return self

        def to(self, device_or_type):
            self.model.to(device_or_type)
            return self

    model = H0_mini(model, cls_or_patch)

    return model


def get_phikon_model(model_dir_str):
    # Feel free to look up the documentation, I'd be happy to be wrong!
    from transformers import ViTModel

    phikon_path = Path(model_dir_str) / "phikon"
    if not phikon_path.is_dir():
        raise FileNotFoundError(f"Couldn't find Phikon folder at {phikon_path}...")

    class Phikon(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model: nn.Module = ViTModel.from_pretrained(
                phikon_path,
                add_pooling_layer=False,
                device_map="cpu",
                local_files_only=True,
            )  # type: ignore

        def __call__(self, x, *args: Any, **kwds: Any) -> torch.Tensor:
            if not 224 in x.shape:
                raise ValueError("Phikon model expects 224x224 images")
            return self.model(pixel_values=x).last_hidden_state[:, 0, :]

        def eval(self):
            self.model.eval()
            return self

    model = Phikon()

    return model


def get_backbone(config: AppCfg) -> nn.Module:
    model_name = config.model.backbone.model_name
    model_dir_str = config.model.backbone.model_dir
    token = config.model.backbone.token
    cls_or_patch = config.model.backbone.cls_or_patch
    logger.info(f"Loading {model_name} from Huggingface")
    if model_name == "h_optimus_1":
        return get_h_optimus_1_model(token, model_dir_str)
    elif model_name == "h0_mini":
        return get_h0_mini_model(token, model_dir_str, cls_or_patch)
    elif model_name == "phikon":
        return get_phikon_model(model_dir_str)
    else:
        raise ValueError(f"{model_name} is not a valid model")
