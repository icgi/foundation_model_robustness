from typing import Optional
from torch import nn
import torch


class MILModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        features: torch.Tensor,
        size: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward method for MIL models.

        Args:
            features (torch.Tensor): Input features of shape (N, F) where N is the number of instances and F is the number of features.
            size (Optional[torch.Tensor]): Optional tensor indicating the size of each bag.

        Returns:
            torch.Tensor: Output logits of shape (1, C) where C is the number of classes.
        """
        ...
        raise NotImplementedError("Subclasses must implement this method.")

    def count_parameters(self) -> int:
        """Counts the number of trainable parameters in the model.

        Returns:
            int: Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_per_tile_score_attn(self, features: torch.Tensor) -> torch.Tensor:
        """Method to get per-tile scores for attention-based MIL models.

        Args:
            features (torch.Tensor): Input features of shape (N, F) where N is the number of instances and F is the number of features.

        Returns:
            torch.Tensor: Per-tile scores of shape (N, 1).
        """
        ...
        raise NotImplementedError("Subclasses must implement this method.")