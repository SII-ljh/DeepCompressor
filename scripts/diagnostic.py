#!/usr/bin/env python
"""Diagnostic experiments for Deep Compressor.

Experiment 1 — Overfitting
    Pre-compute frozen-Qwen hidden states for a single batch, then train
    only DownProj -> QueryInit -> Perceiver -> UpMLP for N steps.
    Reports loss curve, per-module gradient norms, and projection-layer
    output statistics.

Experiment 2 — Information Bottleneck
    Train a linear probe and a 2-layer MLP probe that map mean-pooled
    Qwen document features directly to decoder prefix embeddings.
    Comparison with Experiment 1 shows how much value the Perceiver
    cross-attention adds over naive pooling.

Usage:
    python scripts/diagnostic.py \\
        --config configs/tiny_subset.yaml \\
        --data_path data/ntp_train.jsonl \\
        --steps 300
"""

from __future__ import annotations

import argparse
import math
import os
import sys

# Ensure project root is on sys.path (for running as `python scripts/diagnostic.py`)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from deep_compressor.config import DeepCompressorConfig
from deep_compressor.data import NTPDataset, PaddingCollator
from deep_compressor.model import DeepCompressor


# ─── helpers ──────────────────────────────────────────────────────────

def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _grad_norm(module: nn.Module) -> float:
    """Total L2 gradient norm across all parameters of *module*."""
    sq = 0.0
    for p in module.parameters():
        if p.grad is not None:
            sq += p.grad.data.norm(2).item() ** 2
    return math.sqrt(sq)


def _stats_str(t: torch.Tensor) -> str:
    """One-line tensor statistics string."""
    with torch.no_grad():
        return (f"mean={t.mean().item():+.4f}  std={t.std().item():.4f}  "
                f"min={t.min().item():+.4f}  max={t.max().item():+.4f}")


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()}


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 1 — Overfitting
# ═══════════════════════════════════════════════════════════════════════

def run_overfit(
    model: DeepCompressor,
    doc_hidden: torch.Tensor,     # pre-computed, detached
    batch: dict,                  # already on device
    device: torch.device,
    steps: int,
    lr: float,
    log_every: int,
) -> dict:
    """Overfit the compression pipeline on one fixed batch."""

    print("\n" + "=" * 76)
    print(f"  EXPERIMENT 1: Overfitting (1 batch, {steps} steps, lr={lr})")
    print("=" * 76)

    model.train()

    seg_ids    = batch["segment_ids"]
    seg_mask   = batch["segment_attention_mask"]
    seg_labels = batch["segment_labels"]
    doc_mask   = batch["doc_attention_mask"]

    modules = {
        "down_proj":  model.down_proj,
        "query_init": model.query_init,
        "perceiver":  model.perceiver,
        "up_mlp":     model.up_mlp,
    }
    params = [p for m in modules.values()
              for p in m.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=lr, weight_decay=0.01)

    B = doc_hidden.shape[0]
    D = model.config.qwen.hidden_size

    losses: list[float] = []
    grad_norms: dict[str, list[float]] = {n: [] for n in modules}

    hdr = (f"  {'step':>5}  {'loss':>10}  "
           f"{'|down_proj|':>12} {'|query_init|':>13} "
           f"{'|perceiver|':>12} {'|up_mlp|':>10}")
    print(f"\n{hdr}")
    print("  " + "─" * (len(hdr) - 2))

    for step in range(1, steps + 1):
        optimizer.zero_grad()

        byte_array = model.down_proj(doc_hidden)
        queries    = model.query_init(torch.zeros(B, D, device=device))
        latent     = model.perceiver(queries, byte_array, byte_mask=doc_mask)
        prefix     = model.up_mlp(latent)

        out  = model.decode(prefix, seg_ids, seg_mask, labels=seg_labels)
        loss = out.loss
        loss.backward()

        lv = loss.item()
        losses.append(lv)
        gn = {n: _grad_norm(m) for n, m in modules.items()}
        for n in modules:
            grad_norms[n].append(gn[n])

        if step == 1 or step % log_every == 0:
            print(f"  {step:>5}  {lv:>10.4f}  "
                  f"{gn['down_proj']:>12.4e} {gn['query_init']:>13.4e} "
                  f"{gn['perceiver']:>12.4e} {gn['up_mlp']:>10.4e}")

        optimizer.step()

    # ── projection statistics (final) ─────────────────────────────────
    with torch.no_grad():
        byte_array = model.down_proj(doc_hidden)
        queries    = model.query_init(torch.zeros(B, D, device=device))
        latent     = model.perceiver(queries, byte_array, byte_mask=doc_mask)
        prefix     = model.up_mlp(latent)

    print(f"\n  Projection statistics (after {steps} steps):")
    print(f"    DownProj output:  {_stats_str(byte_array)}")
    print(f"    Latent array:     {_stats_str(latent)}")
    print(f"    UpMLP output:     {_stats_str(prefix)}")

    # ── diagnosis ─────────────────────────────────────────────────────
    first, last, best = losses[0], losses[-1], min(losses)
    reduction_pct = (first - last) / first * 100

    W = min(20, max(1, steps // 5))
    if len(losses) > W:
        wins = sum(1 for i in range(len(losses) - W) if losses[i + W] < losses[i])
        trend_pct = wins / (len(losses) - W) * 100
    else:
        trend_pct = 100.0 if last < first else 0.0

    print(f"\n  DIAGNOSIS:")
    print(f"    loss  {first:.4f} -> {last:.4f}  (min {best:.4f}, {reduction_pct:+.1f}%)")
    print(f"    decreasing-window rate ({W}-step windows): {trend_pct:.0f}%")

    if reduction_pct > 10 and trend_pct > 50:
        print("    PASS     — loss consistently decreases; gradient flow is healthy")
    elif reduction_pct > 5:
        print("    MARGINAL — loss decreasing slowly; consider higher LR or more steps")
    else:
        print("    FAIL     — loss not decreasing; check gradient flow or architecture")

    # gradient health
    for name, gn_list in grad_norms.items():
        avg = sum(gn_list) / len(gn_list)
        if avg < 1e-7:
            print(f"    WARNING  {name}: avg |grad| = {avg:.2e} (near zero)")
        elif avg > 100:
            print(f"    WARNING  {name}: avg |grad| = {avg:.2e} (very large)")

    return {"losses": losses, "grad_norms": grad_norms}


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 2 — Information Bottleneck
# ═══════════════════════════════════════════════════════════════════════

class LinearProbe(nn.Module):
    """Learnable base queries + linear doc-conditioned bias."""

    def __init__(self, num_queries: int, dim: int):
        super().__init__()
        self.base = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.proj = nn.Linear(dim, dim)

    def forward(self, doc_pooled: torch.Tensor) -> torch.Tensor:
        """doc_pooled: (B, D) -> prefix: (B, num_queries, D)"""
        return self.base.unsqueeze(0) + self.proj(doc_pooled).unsqueeze(1)


class MLPProbe(nn.Module):
    """Learnable base queries + 2-layer MLP doc-conditioned bias."""

    def __init__(self, num_queries: int, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or dim
        self.base = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim),
        )

    def forward(self, doc_pooled: torch.Tensor) -> torch.Tensor:
        return self.base.unsqueeze(0) + self.mlp(doc_pooled).unsqueeze(1)


def _train_probe(
    tag: str,
    probe: nn.Module,
    model: DeepCompressor,
    doc_pooled: torch.Tensor,
    batch: dict,
    device: torch.device,
    steps: int,
    lr: float,
    log_every: int,
) -> list[float]:
    """Train one probe; return loss history."""

    probe.to(device)
    probe.train()
    n_params = sum(p.numel() for p in probe.parameters())
    optimizer = AdamW(probe.parameters(), lr=lr, weight_decay=0.01)

    seg_ids    = batch["segment_ids"]
    seg_mask   = batch["segment_attention_mask"]
    seg_labels = batch["segment_labels"]

    losses: list[float] = []
    print(f"\n  {tag}  ({n_params:,} params)")
    print(f"  {'step':>5}  {'loss':>10}  {'|grad|':>10}")
    print(f"  {'─' * 5}  {'─' * 10}  {'─' * 10}")

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        prefix = probe(doc_pooled)                       # (B, nq, D)
        out = model.decode(prefix, seg_ids, seg_mask, labels=seg_labels)
        out.loss.backward()

        lv = out.loss.item()
        losses.append(lv)
        if step == 1 or step % log_every == 0:
            print(f"  {step:>5}  {lv:>10.4f}  {_grad_norm(probe):>10.4e}")

        optimizer.step()

    return losses


def run_bottleneck(
    model: DeepCompressor,
    doc_pooled: torch.Tensor,     # (B, D), detached
    batch: dict,                  # already on device
    device: torch.device,
    steps: int,
    lr: float,
    log_every: int,
) -> dict:
    """Train linear & MLP probes and compare."""

    print("\n" + "=" * 76)
    print("  EXPERIMENT 2: Information Bottleneck (Linear vs MLP probe)")
    print("=" * 76)

    nq = model.config.perceiver.num_queries
    D  = model.config.qwen.hidden_size

    # ── random-prefix baseline (no learning) ──────────────────────────
    with torch.no_grad():
        rnd = torch.randn(doc_pooled.shape[0], nq, D, device=device) * 0.02
        rnd_loss = model.decode(
            rnd, batch["segment_ids"],
            batch["segment_attention_mask"],
            labels=batch["segment_labels"],
        ).loss.item()
    print(f"\n  Random-prefix baseline loss: {rnd_loss:.4f}")

    # ── probes ────────────────────────────────────────────────────────
    lin_losses = _train_probe(
        "Linear Probe", LinearProbe(nq, D),
        model, doc_pooled, batch, device, steps, lr, log_every)

    mlp_losses = _train_probe(
        "MLP Probe (2-layer)", MLPProbe(nq, D),
        model, doc_pooled, batch, device, steps, lr, log_every)

    # ── table ─────────────────────────────────────────────────────────
    print(f"\n  {'─' * 60}")
    print(f"  {'Probe':<30}  {'Init':>12}  {'Final':>12}")
    print(f"  {'─' * 60}")
    print(f"  {'Random (no training)':<30}  {rnd_loss:>12.4f}  {'--':>12}")
    print(f"  {'Linear':<30}  {lin_losses[0]:>12.4f}  {lin_losses[-1]:>12.4f}")
    print(f"  {'MLP (2-layer)':<30}  {mlp_losses[0]:>12.4f}  {mlp_losses[-1]:>12.4f}")
    print(f"  {'─' * 60}")

    gap = lin_losses[-1] - mlp_losses[-1]
    print(f"\n  DIAGNOSIS:")
    print(f"    Linear final:  {lin_losses[-1]:.4f}")
    print(f"    MLP final:     {mlp_losses[-1]:.4f}")
    print(f"    Gap:           {gap:.4f}")

    if gap > 0.5:
        print("    Large gap  — nonlinear mapping adds significant value;")
        print("                 Perceiver cross-attention is well-justified")
    elif gap > 0.1:
        print("    Moderate gap — some benefit from nonlinear compression")
    else:
        print("    Minimal gap  — mean-pool + linear captures most information")

    return {
        "random_loss":   rnd_loss,
        "linear_losses": lin_losses,
        "mlp_losses":    mlp_losses,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 3 — Gradient Flow Analysis
# ═══════════════════════════════════════════════════════════════════════

def run_gradient_flow(
    model: DeepCompressor,
    doc_hidden: torch.Tensor,
    batch: dict,
    device: torch.device,
) -> dict:
    """Single forward+backward pass, report per-layer gradient norms."""

    print("\n" + "=" * 76)
    print("  EXPERIMENT 3: Gradient Flow Analysis")
    print("=" * 76)

    model.train()
    model.zero_grad()

    B = doc_hidden.shape[0]
    D = model.config.qwen.hidden_size

    byte_array = model.down_proj(doc_hidden)
    queries = model.query_init(torch.zeros(B, D, device=device))
    latent = model.perceiver(queries, byte_array, byte_mask=batch["doc_attention_mask"])
    prefix = model.up_mlp(latent)
    out = model.decode(prefix, batch["segment_ids"],
                       batch["segment_attention_mask"],
                       labels=batch["segment_labels"])
    out.loss.backward()

    # Collect per-module, per-layer gradient norms
    results = {}
    modules = {
        "down_proj": model.down_proj,
        "query_init": model.query_init,
        "perceiver": model.perceiver,
        "up_mlp": model.up_mlp,
    }

    print(f"\n  {'Module':<30}  {'Layer':<30}  {'|grad|':>12}  {'Status'}")
    print(f"  {'─' * 90}")

    for mod_name, module in modules.items():
        layer_norms = {}
        for name, param in module.named_parameters():
            if param.grad is not None:
                gn = param.grad.data.norm(2).item()
                layer_norms[name] = gn

                if gn < 1e-8:
                    status = "VANISHING"
                elif gn > 100:
                    status = "EXPLODING"
                else:
                    status = "OK"

                # Only print notable layers to keep output concise
                if gn < 1e-7 or gn > 10 or name.endswith("weight"):
                    print(f"  {mod_name:<30}  {name:<30}  {gn:>12.4e}  {status}")

        results[mod_name] = layer_norms

    # Summary
    all_norms = [gn for layer_norms in results.values() for gn in layer_norms.values()]
    if all_norms:
        min_gn = min(all_norms)
        max_gn = max(all_norms)
        mean_gn = sum(all_norms) / len(all_norms)
        print(f"\n  Summary: min={min_gn:.2e}, max={max_gn:.2e}, mean={mean_gn:.2e}")

        vanishing = sum(1 for g in all_norms if g < 1e-7)
        exploding = sum(1 for g in all_norms if g > 100)
        if vanishing > 0:
            print(f"  WARNING: {vanishing} parameters with vanishing gradients (<1e-7)")
        if exploding > 0:
            print(f"  WARNING: {exploding} parameters with exploding gradients (>100)")
        if vanishing == 0 and exploding == 0:
            print("  PASS — No gradient pathologies detected")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 4 — Attention Pattern Analysis
# ═══════════════════════════════════════════════════════════════════════

def run_attention_analysis(
    model: DeepCompressor,
    doc_hidden: torch.Tensor,
    batch: dict,
    device: torch.device,
) -> dict:
    """Hook into Perceiver attention layers and analyze entropy of attention distributions."""

    print("\n" + "=" * 76)
    print("  EXPERIMENT 4: Attention Pattern Analysis")
    print("=" * 76)

    model.eval()

    B = doc_hidden.shape[0]
    D = model.config.qwen.hidden_size

    attn_weights = {}

    # Manually compute attention weights for analysis
    with torch.no_grad():
        byte_array = model.down_proj(doc_hidden)
        queries = model.query_init(torch.zeros(B, D, device=device))

        x = queries
        doc_mask = batch["doc_attention_mask"]

        # Analyze Stage A cross-attention
        if model.perceiver.enable_stage_a:
            for i, block in enumerate(model.perceiver.stage_a_cross):
                if block.has_cross_attn:
                    ca = block.cross_attn
                    q_norm = ca.norm_q(x)
                    kv_norm = ca.norm_kv(byte_array)
                    H = ca.num_heads
                    q_proj = ca.to_q(q_norm).view(B, -1, H, ca.head_dim).transpose(1, 2)
                    k_proj = ca.to_k(kv_norm).view(B, -1, H, ca.head_dim).transpose(1, 2)
                    attn_logits = (q_proj @ k_proj.transpose(-2, -1)) * ca.scale
                    if doc_mask is not None:
                        attn_logits = attn_logits.masked_fill(
                            ~doc_mask[:, None, None, :].bool(), float("-inf"))
                    attn_probs = torch.softmax(attn_logits, dim=-1)
                    attn_weights[f"stage_a_cross_{i}"] = attn_probs
                x = block(x, kv=byte_array, kv_mask=doc_mask)
            for block in model.perceiver.stage_a_self:
                x = block(x)

        # Stage B self-attention
        if model.perceiver.enable_stage_b:
            for i, block in enumerate(model.perceiver.stage_b_self):
                sa = block.self_attn
                x_norm = sa.norm(x)
                H = sa.num_heads
                q = sa.to_q(x_norm).view(B, -1, H, sa.head_dim).transpose(1, 2)
                k = sa.to_k(x_norm).view(B, -1, H, sa.head_dim).transpose(1, 2)
                attn_logits = (q @ k.transpose(-2, -1)) * sa.scale
                attn_probs = torch.softmax(attn_logits, dim=-1)
                attn_weights[f"stage_b_self_{i}"] = attn_probs
                x = block(x)

        # Stage C cross-attention
        if model.perceiver.enable_stage_c:
            for i, block in enumerate(model.perceiver.stage_c_cross):
                if block.has_cross_attn:
                    ca = block.cross_attn
                    q_norm = ca.norm_q(x)
                    kv_norm = ca.norm_kv(byte_array)
                    H = ca.num_heads
                    q_proj = ca.to_q(q_norm).view(B, -1, H, ca.head_dim).transpose(1, 2)
                    k_proj = ca.to_k(kv_norm).view(B, -1, H, ca.head_dim).transpose(1, 2)
                    attn_logits = (q_proj @ k_proj.transpose(-2, -1)) * ca.scale
                    if doc_mask is not None:
                        attn_logits = attn_logits.masked_fill(
                            ~doc_mask[:, None, None, :].bool(), float("-inf"))
                    attn_probs = torch.softmax(attn_logits, dim=-1)
                    attn_weights[f"stage_c_cross_{i}"] = attn_probs
                x = block(x, kv=byte_array, kv_mask=doc_mask)

    # Analyze attention entropy
    print(f"\n  {'Layer':<25}  {'Mean Entropy':>14}  {'Min Entropy':>13}  {'Max Entropy':>13}")
    print(f"  {'─' * 70}")

    results = {}
    for name, attn in attn_weights.items():
        # attn: (B, H, Q, S) — compute entropy per query position
        eps = 1e-10
        entropy = -(attn * (attn + eps).log()).sum(dim=-1)  # (B, H, Q)
        mean_ent = entropy.mean().item()
        min_ent = entropy.min().item()
        max_ent = entropy.max().item()
        results[name] = {"mean": mean_ent, "min": min_ent, "max": max_ent}
        print(f"  {name:<25}  {mean_ent:>14.4f}  {min_ent:>13.4f}  {max_ent:>13.4f}")

    if results:
        all_means = [r["mean"] for r in results.values()]
        avg_entropy = sum(all_means) / len(all_means)
        print(f"\n  Overall average entropy: {avg_entropy:.4f}")
        if avg_entropy < 0.5:
            print("  WARNING: Very low entropy — attention may be too peaked/collapsed")
        elif avg_entropy > 5.0:
            print("  INFO: High entropy — attention is broadly distributed (uniform-like)")
        else:
            print("  PASS — Entropy in healthy range")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 5 — Compression Fidelity
# ═══════════════════════════════════════════════════════════════════════

def run_compression_fidelity(
    model: DeepCompressor,
    doc_hidden: torch.Tensor,
    batch: dict,
    device: torch.device,
) -> dict:
    """Compute cosine similarity between prefix embeddings and original Qwen hidden states."""

    print("\n" + "=" * 76)
    print("  EXPERIMENT 5: Compression Fidelity")
    print("=" * 76)

    model.eval()

    B = doc_hidden.shape[0]
    D = model.config.qwen.hidden_size
    doc_mask = batch["doc_attention_mask"]

    with torch.no_grad():
        byte_array = model.down_proj(doc_hidden)
        queries = model.query_init(torch.zeros(B, D, device=device))
        latent = model.perceiver(queries, byte_array, byte_mask=doc_mask)
        prefix = model.up_mlp(latent)  # (B, nq, D)

        # Mean-pool prefix and doc_hidden for global comparison
        prefix_pooled = prefix.mean(dim=1)  # (B, D)
        mask_f = doc_mask.unsqueeze(-1).float()
        doc_pooled = ((doc_hidden * mask_f).sum(1) /
                      mask_f.sum(1).clamp(min=1))  # (B, D)

        # Cosine similarity between prefix mean and doc mean
        cos_sim_global = torch.nn.functional.cosine_similarity(
            prefix_pooled, doc_pooled, dim=-1)  # (B,)

        # Per-query cosine similarity with closest doc position
        # prefix: (B, nq, D), doc_hidden: (B, doc_len, D)
        prefix_norm = torch.nn.functional.normalize(prefix, dim=-1)
        doc_norm = torch.nn.functional.normalize(doc_hidden, dim=-1)
        # (B, nq, doc_len) cosine similarity matrix
        sim_matrix = torch.bmm(prefix_norm, doc_norm.transpose(1, 2))
        # Mask out padding positions
        if doc_mask is not None:
            sim_matrix = sim_matrix.masked_fill(
                ~doc_mask[:, None, :].bool(), float("-inf"))
        max_sim_per_query = sim_matrix.max(dim=-1).values  # (B, nq)
        mean_max_sim = max_sim_per_query.mean().item()

        # Random baseline
        random_prefix = torch.randn_like(prefix) * 0.02
        random_pooled = random_prefix.mean(dim=1)
        cos_sim_random = torch.nn.functional.cosine_similarity(
            random_pooled, doc_pooled, dim=-1)

    print(f"\n  {'Metric':<40}  {'Value':>12}")
    print(f"  {'─' * 55}")
    print(f"  {'Global cosine sim (prefix↔doc mean)':<40}  {cos_sim_global.mean().item():>12.4f}")
    print(f"  {'Random baseline cosine sim':<40}  {cos_sim_random.mean().item():>12.4f}")
    print(f"  {'Mean max per-query cosine sim':<40}  {mean_max_sim:>12.4f}")
    print(f"  {'Prefix embedding norm (mean)':<40}  {prefix.norm(dim=-1).mean().item():>12.4f}")
    print(f"  {'Doc hidden states norm (mean)':<40}  {doc_hidden.norm(dim=-1).mean().item():>12.4f}")
    print(f"  {'─' * 55}")

    improvement = cos_sim_global.mean().item() - cos_sim_random.mean().item()
    print(f"\n  Improvement over random: {improvement:+.4f}")
    if improvement > 0.1:
        print("  PASS — Prefix captures meaningful document information")
    elif improvement > 0:
        print("  MARGINAL — Some document signal, but weak")
    else:
        print("  FAIL — Prefix not more similar to doc than random")

    return {
        "global_cos_sim": cos_sim_global.mean().item(),
        "random_cos_sim": cos_sim_random.mean().item(),
        "mean_max_query_sim": mean_max_sim,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

ALL_EXPERIMENTS = ["1", "2", "3", "4", "5"]


def main():
    import warnings
    warnings.warn(
        "scripts/diagnostic.py is deprecated. "
        "Use scripts/diagnostics/pre_training.py, mid_training.py, or post_training.py instead.",
        DeprecationWarning, stacklevel=1,
    )

    parser = argparse.ArgumentParser(description="Deep Compressor diagnostics")
    parser.add_argument("--config",     default="configs/tiny_subset.yaml")
    parser.add_argument("--data_path",  default="data/ntp_train.jsonl")
    parser.add_argument("--steps",      type=int,   default=300)
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--log_every",  type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=2)
    parser.add_argument("--experiments", type=str, default="1,2,3,4,5",
                        help="Comma-separated list of experiments to run (1-5)")
    parser.add_argument("--wandb", action="store_true",
                        help="Log results to wandb")
    parser.add_argument("--wandb_project", type=str, default="dc-diagnostic",
                        help="wandb project name")
    args = parser.parse_args()

    experiments = [e.strip() for e in args.experiments.split(",")]

    device = _device()

    # ── wandb init ────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name="diagnostic",
                config={"steps": args.steps, "lr": args.lr,
                        "experiments": experiments},
            )
        except ImportError:
            print("WARNING: wandb not installed, skipping wandb logging")

    # ── setup ─────────────────────────────────────────────────────────
    print(f"Device: {device}")
    config = DeepCompressorConfig.from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config.qwen.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = NTPDataset(args.data_path, tokenizer,
                    max_doc_tokens=config.qwen.max_doc_tokens,
                    segment_len=config.training.ntp_segment_len)
    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
    n = min(args.batch_size, len(ds))
    subset = Subset(ds, list(range(n)))
    batch = next(iter(DataLoader(subset, batch_size=n,
                                 shuffle=False, collate_fn=collator)))

    print(f"Batch: {n} samples")
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f"  {k:<25} {str(tuple(v.shape)):>20}")

    print("\nLoading model...")
    model = DeepCompressor(config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters()) - trainable
    print(f"  Trainable: {trainable:,}  |  Frozen: {frozen:,}")

    model.to(device)
    batch = _to_device(batch, device)

    # ── pre-compute Qwen features (shared by all experiments) ─────────
    print("\nPre-computing Qwen document features (one-time)...")
    with torch.no_grad():
        qwen_out = model.qwen(
            input_ids=batch["doc_input_ids"],
            attention_mask=batch["doc_attention_mask"],
            output_hidden_states=True, use_cache=False,
        )
        doc_hidden = qwen_out.hidden_states[-1].detach()   # (B, doc_len, D)
        mask_f = batch["doc_attention_mask"].unsqueeze(-1).float()
        doc_pooled = ((doc_hidden * mask_f).sum(1)
                      / mask_f.sum(1).clamp(min=1)).detach()  # (B, D)
    del qwen_out
    print(f"  doc_hidden: {tuple(doc_hidden.shape)}")
    print(f"  doc_pooled: {tuple(doc_pooled.shape)}")

    # ── experiments ───────────────────────────────────────────────────
    all_results = {}

    if "1" in experiments:
        torch.manual_seed(config.training.seed)
        exp1 = run_overfit(model, doc_hidden, batch, device,
                           args.steps, args.lr, args.log_every)
        all_results["overfit"] = exp1
        if wandb_run:
            for i, loss in enumerate(exp1["losses"]):
                wandb_run.log({"exp1/loss": loss}, step=i + 1)

    if "2" in experiments:
        torch.manual_seed(config.training.seed)
        exp2 = run_bottleneck(model, doc_pooled, batch, device,
                              args.steps, args.lr, args.log_every)
        all_results["bottleneck"] = exp2
        if wandb_run:
            wandb_run.log({
                "exp2/random_loss": exp2["random_loss"],
                "exp2/linear_final": exp2["linear_losses"][-1],
                "exp2/mlp_final": exp2["mlp_losses"][-1],
            })

    if "3" in experiments:
        torch.manual_seed(config.training.seed)
        exp3 = run_gradient_flow(model, doc_hidden, batch, device)
        all_results["gradient_flow"] = exp3
        if wandb_run:
            for mod_name, layer_norms in exp3.items():
                for layer_name, gn in layer_norms.items():
                    wandb_run.log({f"exp3/{mod_name}/{layer_name}": gn})

    if "4" in experiments:
        torch.manual_seed(config.training.seed)
        exp4 = run_attention_analysis(model, doc_hidden, batch, device)
        all_results["attention"] = exp4
        if wandb_run:
            for name, stats in exp4.items():
                for k, v in stats.items():
                    wandb_run.log({f"exp4/{name}/{k}": v})

    if "5" in experiments:
        torch.manual_seed(config.training.seed)
        exp5 = run_compression_fidelity(model, doc_hidden, batch, device)
        all_results["fidelity"] = exp5
        if wandb_run:
            for k, v in exp5.items():
                wandb_run.log({f"exp5/{k}": v})

    # ── final comparison (if experiments 1 & 2 were both run) ─────────
    if "1" in experiments and "2" in experiments:
        overfit_final = all_results["overfit"]["losses"][-1]
        linear_final  = all_results["bottleneck"]["linear_losses"][-1]
        mlp_final     = all_results["bottleneck"]["mlp_losses"][-1]
        rnd           = all_results["bottleneck"]["random_loss"]

        print("\n" + "=" * 76)
        print("  FINAL COMPARISON")
        print("=" * 76)

        print(f"\n  {'Method':<35}  {'Loss':>10}")
        print(f"  {'─' * 48}")
        print(f"  {'Random prefix (no learning)':<35}  {rnd:>10.4f}")
        print(f"  {'Linear probe':<35}  {linear_final:>10.4f}")
        print(f"  {'MLP probe (2-layer)':<35}  {mlp_final:>10.4f}")
        print(f"  {'Full pipeline (Perceiver)':<35}  {overfit_final:>10.4f}")
        print(f"  {'─' * 48}")

        all_beat_random = linear_final < rnd and mlp_final < rnd
        pipeline_beats_mlp    = overfit_final < mlp_final
        pipeline_beats_linear = overfit_final < linear_final

        print(f"\n  Conclusions:")
        if not all_beat_random:
            print("    [!] Not all probes beat random — training may need more steps")
        if pipeline_beats_mlp:
            print(f"    Full pipeline ({overfit_final:.4f}) < MLP ({mlp_final:.4f}) "
                  f"< Linear ({linear_final:.4f})")
            print("    => Perceiver cross-attention extracts richer document features")
            print("    => Architecture is working as intended")
        elif pipeline_beats_linear:
            print(f"    Linear ({linear_final:.4f}) > Full ({overfit_final:.4f}) "
                  f"> MLP ({mlp_final:.4f})")
            print("    => Pipeline outperforms linear but not MLP — functioning, room to improve")
        else:
            print(f"    Full ({overfit_final:.4f}) >= Linear ({linear_final:.4f})")
            print("    => Pipeline not outperforming simple baseline — investigate architecture")

    if wandb_run:
        wandb_run.finish()

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
