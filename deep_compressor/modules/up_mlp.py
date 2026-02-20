"""UpMLP: maps latent array from Perceiver dimension back to Qwen dimension."""

import torch
import torch.nn as nn


class UpMLP(nn.Module):
    def __init__(self, perceiver_dim: int, qwen_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(perceiver_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, qwen_dim),
            nn.LayerNorm(qwen_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_queries, perceiver_dim)
        Returns:
            (batch, num_queries, qwen_dim)
        """
        return self.net(x)
