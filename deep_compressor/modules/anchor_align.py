"""AnchorAlign: maps FinBERT hidden states to Perceiver space with residual MLP layers."""

import torch
import torch.nn as nn


class AnchorAlign(nn.Module):
    def __init__(self, finbert_dim: int, perceiver_dim: int, num_layers: int = 3):
        super().__init__()
        layers = []
        in_dim = finbert_dim
        for i in range(num_layers):
            out_dim = perceiver_dim
            layers.append(_ResidualMLP(in_dim, out_dim))
            in_dim = out_dim
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(perceiver_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, top_k, finbert_dim)
        Returns:
            (batch, top_k, perceiver_dim)
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class _ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.use_residual = in_dim == out_dim
        if not self.use_residual:
            self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear2(self.act(self.linear1(x)))
        if self.use_residual:
            return x + h
        return self.proj(x) + h
