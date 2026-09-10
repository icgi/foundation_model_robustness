import torch
from torch import nn

from src.network.MILABMIL import MILModel
from src.network.get_foundation_model import (
    get_backbone,
    get_num_feats_model,
)
from src.conf.train_schema import AppCfg
from loguru import logger


class Model(nn.Module):
    def __init__(self, config: AppCfg, num_classes: int) -> None:
        super().__init__()
        self.backbone = get_backbone(
            config,
        )
        self.ABMIL = MILModel(
            get_num_feats_model(config),
            config.model.head.dropout_rate,
            config.model.head.num_attention_heads,
            num_classes,
        )

        self.validate_model()
        logger.info("Successfully loaded model")

    def validate_model(self):
        for name, p in self.named_parameters():
            if not p.requires_grad:
                raise ValueError(f"Parameter {name} does not require grad")

    def forward(self, bags: torch.Tensor, lens: torch.Tensor, return_embeddings=False):
        B, BAG, C, H, W = bags.shape
        flattened_bag = bags.reshape(B * BAG, C, H, W)
        embeddings: torch.Tensor = self.backbone(flattened_bag)
        embeddings = embeddings.reshape(B, BAG, embeddings.shape[-1])

        prediction = self.ABMIL(embeddings, lens)
        if return_embeddings:
            return prediction, embeddings
        return prediction

    def count_parameters(self):
        ABMIL_parameters = sum(
            p.numel() for p in self.ABMIL.parameters() if p.requires_grad
        )
        backbone_parameters = sum(
            p.numel() for p in self.backbone.parameters() if p.requires_grad
        )
        total = ABMIL_parameters + backbone_parameters
        return {
            "ABMIL": ABMIL_parameters / 10**6,
            "backbone": backbone_parameters / 10**6,
            "total": total / 10**6,
        }

    @property
    def device(self):
        return next(self.parameters()).device


def get_model(
    num_features: int,
    num_labels: int,
    config: AppCfg,
    device: torch.device,
    compile: bool = False,
) -> MILModel:
    model = MILModel(
        n_feats=num_features,
        n_out=num_labels,
        dropout_head=config.model.dropout_head,
        embedding_size=config.model.embedding_size,
        encoder_layers=config.model.encoder_layers,
        use_batchnorm=config.model.use_batchnorm,
    ).to(device)
    if compile:
        model: MILModel = torch.compile(model)
    # tqdm.write(f"Running model with:, {model.count_parameters()}, parameters")
    return model
