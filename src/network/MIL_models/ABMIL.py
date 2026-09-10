from typing import Optional

import torch
from torch import nn

from src.network.MIL_models.base_model import MILModel


class ABMILModel(MILModel):
    def __init__(
        self,
        n_feats: int,
        n_out: int,
        dropout_head: float,
        embedding_size: int,
        encoder_layers: int,
        use_batchnorm: bool = True,
    ) -> None:
        """Create a new attention MIL model.

        Args:
            n_feats:  The nuber of features each bag instance has.
            n_out:  The number of output layers of the model.
            encoder:  A network transforming bag instances into feature vectors.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feats, embedding_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(embedding_size) if use_batchnorm else nn.Identity(),
            *[
                nn.Sequential(
                    nn.Linear(embedding_size, embedding_size),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(embedding_size) if use_batchnorm else nn.Identity(),
                )
                for _ in range(encoder_layers - 1)
            ],
        )
        self.n_out = n_out
        self.attention = Attention(embedding_size)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(embedding_size),
            nn.Dropout(dropout_head) if dropout_head > 0.0 else nn.Identity(),
            nn.Linear(embedding_size, n_out),
        )

        self.initialize_weights()

    def forward(
        self,
        bags,
        lens,
        return_embeddings=False,
        return_weighted_embeddings=False,
        return_attention_scores=False,
        **kwargs,
    ):
        assert (
            int(return_weighted_embeddings)
            + int(return_attention_scores)
            + int(return_embeddings)
        ) <= 1, "Cannot return both embeddings and attention scores at the same time as of now."
        assert bags.ndim == 3
        assert bags.shape[0] == lens.shape[0]
        embeddings = self.encoder(bags.reshape(-1, bags.shape[-1]))
        embeddings = embeddings.reshape(bags.shape[0], bags.shape[1], -1)

        # Zero out embeddings for padded tiles
        batch_size, bag_size, emb_dim = embeddings.shape
        tile_indices = torch.arange(bag_size, device=embeddings.device)
        valid_mask = tile_indices.unsqueeze(0) < lens.unsqueeze(1)  # (batch_size, bag_size)
        embeddings = embeddings * valid_mask.unsqueeze(-1)  # Zero out padded positions

        masked_attention_scores = self._masked_attention_scores(embeddings, lens)

        weighted_embedding_sums = (masked_attention_scores * embeddings).sum(-2)

        scores = self.head(weighted_embedding_sums)

        if return_attention_scores:
            return scores, masked_attention_scores

        if return_embeddings:
            return scores, embeddings

        if return_weighted_embeddings:
            return scores, weighted_embedding_sums

        return scores

    def forward_single_tile(self, tile):
        embeddings = self.encoder(tile.squeeze())
        attention = self.attention(embeddings)
        # attention = torch.softmax(attention.squeeze(), dim=0)
        scores = self.head(embeddings)

        return scores, attention

    def get_per_tile_score_attn(self, features: torch.Tensor) -> torch.Tensor:
        """Get per-tile scores for attention-based MIL models.

        Args:
            features (torch.Tensor): Input features of shape (N, F) where N is the number of instances and F is the number of features.
        Returns:
            torch.Tensor: Per-tile scores of shape (N, 1).
        """
        embeddings = self.encoder(features)
        attention_scores = self.attention(embeddings)
        scores = self.head(embeddings)
        return scores, attention_scores

    def _masked_attention_scores(self, embeddings, lens):
        """Calculates attention scores for all bags.

        Returns:
            A tensor containing
            torch.concat([torch.rand(64, 256), torch.rand(64, 23)], -1)
             *  The attention score of instance i of bag j if i < len[j]
             *  0 otherwise
        """
        bs, bag_size = embeddings.shape[0], embeddings.shape[1]
        attention_scores = self.attention(embeddings)

        # a tensor containing a row [0, ..., bag_size-1] for each batch instance
        idx = torch.arange(bag_size).repeat(bs, 1).to(attention_scores.device)

        # False for every instance of bag i with index(instance) >= lens[i]
        attention_mask = (idx < lens.unsqueeze(-1)).unsqueeze(-1)

        mask_value = -1e4 if attention_scores.dtype == torch.float16 else -1e8
        masked_attention = torch.where(
            attention_mask,
            attention_scores,
            torch.full_like(attention_scores, mask_value),
        )
        return torch.softmax(masked_attention, dim=1)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def device(self):
        return self.encoder[0].bias.device


def Attention(n_in: int, n_latent: Optional[int] = None) -> nn.Module:
    """A network calculating an embedding's importance weight."""
    n_latent = n_latent or (n_in + 1) // 2

    return nn.Sequential(nn.Linear(n_in, n_latent), nn.Tanh(), nn.Linear(n_latent, 1))
