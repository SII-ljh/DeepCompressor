"""GuidedPerceiver: three-stage attention compression architecture."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceiverCrossAttention(nn.Module):
    """Cross-attention from queries to key-value source."""

    def __init__(self, dim: int, num_heads: int, head_dim: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.to_k = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.to_v = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.to_out = nn.Linear(num_heads * head_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                kv_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            q: (batch, q_len, dim) — queries
            kv: (batch, kv_len, dim) — key-value source
            kv_mask: (batch, kv_len) — True for valid positions
        Returns:
            (batch, q_len, dim)
        """
        residual = q
        q = self.norm_q(q)
        kv = self.norm_kv(kv)

        B, Q, _ = q.shape
        _, S, _ = kv.shape
        H = self.num_heads

        q_proj = self.to_q(q).view(B, Q, H, self.head_dim).transpose(1, 2)  # (B, H, Q, D)
        k_proj = self.to_k(kv).view(B, S, H, self.head_dim).transpose(1, 2)
        v_proj = self.to_v(kv).view(B, S, H, self.head_dim).transpose(1, 2)

        attn = (q_proj @ k_proj.transpose(-2, -1)) * self.scale  # (B, H, Q, S)

        if kv_mask is not None:
            attn = attn.masked_fill(~kv_mask[:, None, None, :].bool(), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v_proj).transpose(1, 2).reshape(B, Q, H * self.head_dim)
        return residual + self.to_out(out)


class PerceiverSelfAttention(nn.Module):
    """Standard self-attention with pre-norm and residual."""

    def __init__(self, dim: int, num_heads: int, head_dim: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.to_k = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.to_v = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.to_out = nn.Linear(num_heads * head_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            (batch, seq_len, dim)
        """
        residual = x
        x = self.norm(x)

        B, N, _ = x.shape
        H = self.num_heads

        q = self.to_q(x).view(B, N, H, self.head_dim).transpose(1, 2)
        k = self.to_k(x).view(B, N, H, self.head_dim).transpose(1, 2)
        v = self.to_v(x).view(B, N, H, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, H * self.head_dim)
        return residual + self.to_out(out)


class PerceiverFeedForward(nn.Module):
    """Pre-norm feed-forward with residual."""

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class PerceiverBlock(nn.Module):
    """A single Perceiver block: optional cross-attn + self-attn + feed-forward."""

    def __init__(self, dim: int, num_heads: int, head_dim: int, ff_mult: int = 4,
                 dropout: float = 0.1, has_cross_attn: bool = False):
        super().__init__()
        self.has_cross_attn = has_cross_attn
        if has_cross_attn:
            self.cross_attn = PerceiverCrossAttention(dim, num_heads, head_dim, dropout)
        self.self_attn = PerceiverSelfAttention(dim, num_heads, head_dim, dropout)
        self.ff = PerceiverFeedForward(dim, ff_mult, dropout)

    def forward(self, x: torch.Tensor, kv: Optional[torch.Tensor] = None,
                kv_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.has_cross_attn:
            assert kv is not None, "kv required for cross-attention block"
            x = self.cross_attn(x, kv, kv_mask=kv_mask)
        x = self.self_attn(x)
        x = self.ff(x)
        return x


class GuidedPerceiver(nn.Module):
    """Three-stage guided Perceiver for document compression.

    Stage A: global cross-attention from byte_array + self-attention
    Stage B: self-attention
    Stage C: deep reasoning cross-attention back to byte_array + self-attention
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int, ff_mult: int = 4,
                 dropout: float = 0.1,
                 stage_a_cross_layers: int = 2, stage_a_self_layers: int = 2,
                 stage_b_layers: int = 2,
                 stage_c_cross_layers: int = 2, stage_c_self_layers: int = 4,
                 enable_stage_a: bool = True,
                 enable_stage_b: bool = True,
                 enable_stage_c: bool = True):
        super().__init__()
        self.enable_stage_a = enable_stage_a
        self.enable_stage_b = enable_stage_b
        self.enable_stage_c = enable_stage_c

        # Stage A: cross-attn from byte_array + self-attn
        if enable_stage_a:
            self.stage_a_cross = nn.ModuleList([
                PerceiverBlock(dim, num_heads, head_dim, ff_mult, dropout, has_cross_attn=True)
                for _ in range(stage_a_cross_layers)
            ])
            self.stage_a_self = nn.ModuleList([
                PerceiverBlock(dim, num_heads, head_dim, ff_mult, dropout, has_cross_attn=False)
                for _ in range(stage_a_self_layers)
            ])
        else:
            self.stage_a_cross = nn.ModuleList()
            self.stage_a_self = nn.ModuleList()

        # Stage B: self-attention
        if enable_stage_b:
            self.stage_b_self = nn.ModuleList([
                PerceiverBlock(dim, num_heads, head_dim, ff_mult, dropout, has_cross_attn=False)
                for _ in range(stage_b_layers)
            ])
        else:
            self.stage_b_self = nn.ModuleList()

        # Stage C: cross-attn back to byte_array + deep self-attn
        if enable_stage_c:
            self.stage_c_cross = nn.ModuleList([
                PerceiverBlock(dim, num_heads, head_dim, ff_mult, dropout, has_cross_attn=True)
                for _ in range(stage_c_cross_layers)
            ])
            self.stage_c_self = nn.ModuleList([
                PerceiverBlock(dim, num_heads, head_dim, ff_mult, dropout, has_cross_attn=False)
                for _ in range(stage_c_self_layers)
            ])
        else:
            self.stage_c_cross = nn.ModuleList()
            self.stage_c_self = nn.ModuleList()

        self.final_norm = nn.LayerNorm(dim)

    def forward(self, queries: torch.Tensor, byte_array: torch.Tensor,
                byte_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            queries: (batch, num_queries, dim) — from QueryInit
            byte_array: (batch, doc_len, dim) — from DownProj
            byte_mask: (batch, doc_len) — True for valid token positions
        Returns:
            latent_array: (batch, num_queries, dim)
        """
        x = queries

        # Stage A: global compression from byte_array
        if self.enable_stage_a:
            for block in self.stage_a_cross:
                x = block(x, kv=byte_array, kv_mask=byte_mask)
            for block in self.stage_a_self:
                x = block(x)

        # Stage B: self-attention
        if self.enable_stage_b:
            for block in self.stage_b_self:
                x = block(x)

        # Stage C: deep reasoning re-read
        if self.enable_stage_c:
            for block in self.stage_c_cross:
                x = block(x, kv=byte_array, kv_mask=byte_mask)
            for block in self.stage_c_self:
                x = block(x)

        return self.final_norm(x)
