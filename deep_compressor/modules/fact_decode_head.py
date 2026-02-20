"""FactDecodeHead: auxiliary loss head for anchor reconstruction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactDecodeHead(nn.Module):
    def __init__(self, perceiver_dim: int, finbert_dim: int):
        super().__init__()
        self.proj = nn.Linear(perceiver_dim, finbert_dim)

    def forward(self, processed_anchors: torch.Tensor, original_anchors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            processed_anchors: (batch, top_k, perceiver_dim) — after Perceiver Stage B
            original_anchors: (batch, top_k, finbert_dim) — original FinBERT anchor embeddings
        Returns:
            mse_loss: scalar reconstruction loss
        """
        reconstructed = self.proj(processed_anchors)
        return F.mse_loss(reconstructed, original_anchors.detach())
