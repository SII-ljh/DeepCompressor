"""DownProj: projects Qwen hidden states to Perceiver dimension."""

import torch
import torch.nn as nn


class DownProj(nn.Module):
    def __init__(self, qwen_dim: int, perceiver_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(qwen_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, perceiver_dim),
            nn.LayerNorm(perceiver_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, qwen_dim)
        Returns:
            (batch, seq_len, perceiver_dim)
        """
        return self.net(x)
