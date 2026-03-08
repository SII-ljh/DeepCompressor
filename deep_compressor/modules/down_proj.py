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
        # Align dtype with first layer's weights to avoid BFloat16/Float32 mismatch
        target_dtype = next(self.net.parameters()).dtype
        return self.net(x.to(target_dtype))


class IdentityProj(nn.Module):
    """Identity projection — passes input through unchanged (requires qwen_dim == perceiver_dim)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LinearProj(nn.Module):
    """Single linear projection with LayerNorm."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Align dtype with first layer's weights to avoid BFloat16/Float32 mismatch
        target_dtype = next(self.net.parameters()).dtype
        return self.net(x.to(target_dtype))


def build_down_proj(mode: str, qwen_dim: int, perceiver_dim: int,
                    hidden_dim: int, dropout: float) -> nn.Module:
    """Factory function for down-projection modules.

    Args:
        mode: "mlp" | "linear" | "identity"
    """
    if mode == "mlp":
        return DownProj(qwen_dim, perceiver_dim, hidden_dim, dropout)
    elif mode == "linear":
        return LinearProj(qwen_dim, perceiver_dim)
    elif mode == "identity":
        return IdentityProj()
    raise ValueError(f"Unknown down_proj_mode: {mode}")
