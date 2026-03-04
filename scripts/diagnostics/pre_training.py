#!/usr/bin/env python
"""Pre-training diagnostic experiments for Deep Compressor.

Exp 1 — Overfitting: Train compression pipeline on single batch, verify loss decreases.
Exp 2 — Gradient Flow: Single forward+backward, report per-layer gradient norms.
Exp 3 — Information Bottleneck: Compare linear/MLP probes vs full pipeline.

Usage:
    python scripts/diagnostics/pre_training.py \
        --config configs/macbook_debug.yaml \
        --data_path data/ntp_tiny.jsonl \
        --steps 2000 --probe_steps 300 --experiments 1,2,3
"""

from __future__ import annotations

import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
from torch.optim import AdamW

from scripts.diagnostics.common import (
    base_parser,
    detect_device,
    finish_wandb,
    grad_norm,
    init_wandb,
    load_model,
    log_wandb,
    precompute_qwen_features,
    prepare_ntp_batch,
    stats_str,
)
from deep_compressor.config import DeepCompressorConfig


ALL_EXPERIMENTS = ["1", "2", "3"]


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 1 — Overfitting
# ═══════════════════════════════════════════════════════════════════════

def run_overfit(
    model,
    doc_hidden: torch.Tensor,
    batch: dict,
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

    seg_ids = batch["segment_ids"]
    seg_mask = batch["segment_attention_mask"]
    seg_labels = batch["segment_labels"]
    doc_mask = batch["doc_attention_mask"]

    modules = {
        "down_proj": model.down_proj,
        "query_init": model.query_init,
        "perceiver": model.perceiver,
        "up_mlp": model.up_mlp,
    }
    params = [
        p for m in modules.values() for p in m.parameters() if p.requires_grad
    ]
    optimizer = AdamW(params, lr=lr, weight_decay=0.01)

    B = doc_hidden.shape[0]
    D = model.config.qwen.hidden_size

    losses: list[float] = []
    grad_norms: dict[str, list[float]] = {n: [] for n in modules}

    hdr = (
        f"  {'step':>5}  {'loss':>10}  "
        f"{'|down_proj|':>12} {'|query_init|':>13} "
        f"{'|perceiver|':>12} {'|up_mlp|':>10}"
    )
    print(f"\n{hdr}")
    print("  " + "─" * (len(hdr) - 2))

    for step in range(1, steps + 1):
        optimizer.zero_grad()

        byte_array = model.down_proj(doc_hidden)
        queries = model.query_init(torch.zeros(B, D, device=device))
        latent = model.perceiver(queries, byte_array, byte_mask=doc_mask)
        prefix = model.up_mlp(latent)

        out = model.decode(prefix, seg_ids, seg_mask, labels=seg_labels)
        loss = out.loss
        loss.backward()

        lv = loss.item()
        losses.append(lv)
        gn = {n: grad_norm(m) for n, m in modules.items()}
        for n in modules:
            grad_norms[n].append(gn[n])

        if step == 1 or step % log_every == 0:
            print(
                f"  {step:>5}  {lv:>10.4f}  "
                f"{gn['down_proj']:>12.4e} {gn['query_init']:>13.4e} "
                f"{gn['perceiver']:>12.4e} {gn['up_mlp']:>10.4e}"
            )

        optimizer.step()

    # ── projection statistics (final) ─────────────────────────────────
    with torch.no_grad():
        byte_array = model.down_proj(doc_hidden)
        queries = model.query_init(torch.zeros(B, D, device=device))
        latent = model.perceiver(queries, byte_array, byte_mask=doc_mask)
        prefix = model.up_mlp(latent)

    print(f"\n  Projection statistics (after {steps} steps):")
    print(f"    DownProj output:  {stats_str(byte_array)}")
    print(f"    Latent array:     {stats_str(latent)}")
    print(f"    UpMLP output:     {stats_str(prefix)}")

    # ── diagnosis ─────────────────────────────────────────────────────
    first, last, best = losses[0], losses[-1], min(losses)
    reduction_pct = (first - last) / first * 100

    W = min(20, max(1, steps // 5))
    if len(losses) > W:
        wins = sum(
            1 for i in range(len(losses) - W) if losses[i + W] < losses[i]
        )
        trend_pct = wins / (len(losses) - W) * 100
    else:
        trend_pct = 100.0 if last < first else 0.0

    print(f"\n  DIAGNOSIS:")
    print(
        f"    loss  {first:.4f} -> {last:.4f}  (min {best:.4f}, {reduction_pct:+.1f}%)"
    )
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
#  Experiment 2 — Gradient Flow Analysis
# ═══════════════════════════════════════════════════════════════════════

def run_gradient_flow(
    model,
    doc_hidden: torch.Tensor,
    batch: dict,
    device: torch.device,
) -> dict:
    """Single forward+backward pass, report per-layer gradient norms."""

    print("\n" + "=" * 76)
    print("  EXPERIMENT 2: Gradient Flow Analysis")
    print("=" * 76)

    model.train()
    model.zero_grad()

    B = doc_hidden.shape[0]
    D = model.config.qwen.hidden_size

    byte_array = model.down_proj(doc_hidden)
    queries = model.query_init(torch.zeros(B, D, device=device))
    latent = model.perceiver(
        queries, byte_array, byte_mask=batch["doc_attention_mask"]
    )
    prefix = model.up_mlp(latent)
    out = model.decode(
        prefix,
        batch["segment_ids"],
        batch["segment_attention_mask"],
        labels=batch["segment_labels"],
    )
    out.loss.backward()

    # Collect per-module, per-layer gradient norms
    results = {}
    modules = {
        "down_proj": model.down_proj,
        "query_init": model.query_init,
        "perceiver": model.perceiver,
        "up_mlp": model.up_mlp,
    }

    print(
        f"\n  {'Module':<30}  {'Layer':<30}  {'|grad|':>12}  {'Status'}"
    )
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

                if gn < 1e-7 or gn > 10 or name.endswith("weight"):
                    print(
                        f"  {mod_name:<30}  {name:<30}  {gn:>12.4e}  {status}"
                    )

        results[mod_name] = layer_norms

    # Summary
    all_norms = [
        gn for layer_norms in results.values() for gn in layer_norms.values()
    ]
    if all_norms:
        min_gn = min(all_norms)
        max_gn = max(all_norms)
        mean_gn = sum(all_norms) / len(all_norms)
        print(f"\n  Summary: min={min_gn:.2e}, max={max_gn:.2e}, mean={mean_gn:.2e}")

        vanishing = sum(1 for g in all_norms if g < 1e-7)
        exploding = sum(1 for g in all_norms if g > 100)
        if vanishing > 0:
            print(
                f"  WARNING: {vanishing} parameters with vanishing gradients (<1e-7)"
            )
        if exploding > 0:
            print(
                f"  WARNING: {exploding} parameters with exploding gradients (>100)"
            )
        if vanishing == 0 and exploding == 0:
            print("  PASS — No gradient pathologies detected")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 3 — Information Bottleneck
# ═══════════════════════════════════════════════════════════════════════

class LinearProbe(nn.Module):
    """Learnable base queries + linear doc-conditioned bias."""

    def __init__(self, num_queries: int, dim: int):
        super().__init__()
        self.base = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.proj = nn.Linear(dim, dim)

    def forward(self, doc_pooled: torch.Tensor) -> torch.Tensor:
        return self.base.unsqueeze(0) + self.proj(doc_pooled).unsqueeze(1)


class MLPProbe(nn.Module):
    """Learnable base queries + 2-layer MLP doc-conditioned bias."""

    def __init__(self, num_queries: int, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or dim
        self.base = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, doc_pooled: torch.Tensor) -> torch.Tensor:
        return self.base.unsqueeze(0) + self.mlp(doc_pooled).unsqueeze(1)


def _train_probe(
    tag: str,
    probe: nn.Module,
    model,
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

    seg_ids = batch["segment_ids"]
    seg_mask = batch["segment_attention_mask"]
    seg_labels = batch["segment_labels"]

    losses: list[float] = []
    print(f"\n  {tag}  ({n_params:,} params)")
    print(f"  {'step':>5}  {'loss':>10}  {'|grad|':>10}")
    print(f"  {'─' * 5}  {'─' * 10}  {'─' * 10}")

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        prefix = probe(doc_pooled)
        out = model.decode(prefix, seg_ids, seg_mask, labels=seg_labels)
        out.loss.backward()

        lv = out.loss.item()
        losses.append(lv)
        if step == 1 or step % log_every == 0:
            print(f"  {step:>5}  {lv:>10.4f}  {grad_norm(probe):>10.4e}")

        optimizer.step()

    return losses


def run_bottleneck(
    model,
    doc_pooled: torch.Tensor,
    batch: dict,
    device: torch.device,
    steps: int,
    lr: float,
    log_every: int,
) -> dict:
    """Train linear & MLP probes and compare."""

    print("\n" + "=" * 76)
    print("  EXPERIMENT 3: Information Bottleneck (Linear vs MLP probe)")
    print("=" * 76)

    nq = model.config.perceiver.num_queries
    D = model.config.qwen.hidden_size

    # ── random-prefix baseline (no learning) ──────────────────────────
    with torch.no_grad():
        rnd = torch.randn(doc_pooled.shape[0], nq, D, device=device) * 0.02
        rnd_loss = model.decode(
            rnd,
            batch["segment_ids"],
            batch["segment_attention_mask"],
            labels=batch["segment_labels"],
        ).loss.item()
    print(f"\n  Random-prefix baseline loss: {rnd_loss:.4f}")

    # ── probes ────────────────────────────────────────────────────────
    lin_losses = _train_probe(
        "Linear Probe",
        LinearProbe(nq, D),
        model,
        doc_pooled,
        batch,
        device,
        steps,
        lr,
        log_every,
    )

    mlp_losses = _train_probe(
        "MLP Probe (2-layer)",
        MLPProbe(nq, D),
        model,
        doc_pooled,
        batch,
        device,
        steps,
        lr,
        log_every,
    )

    # ── table ─────────────────────────────────────────────────────────
    print(f"\n  {'─' * 60}")
    print(f"  {'Probe':<30}  {'Init':>12}  {'Final':>12}")
    print(f"  {'─' * 60}")
    print(f"  {'Random (no training)':<30}  {rnd_loss:>12.4f}  {'--':>12}")
    print(
        f"  {'Linear':<30}  {lin_losses[0]:>12.4f}  {lin_losses[-1]:>12.4f}"
    )
    print(
        f"  {'MLP (2-layer)':<30}  {mlp_losses[0]:>12.4f}  {mlp_losses[-1]:>12.4f}"
    )
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
        "random_loss": rnd_loss,
        "linear_losses": lin_losses,
        "mlp_losses": mlp_losses,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = base_parser("Deep Compressor pre-training diagnostics (Exp 1-3)")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Training steps for full pipeline (Exp 1)")
    parser.add_argument("--probe_steps", type=int, default=300,
                        help="Training steps for linear/MLP probes (Exp 3)")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--min_doc_tokens", type=int, default=0,
                        help="Skip docs shorter than this (0=no filter)")
    parser.add_argument(
        "--experiments",
        type=str,
        default="1,2,3",
        help="Comma-separated list of experiments to run (1-3)",
    )
    parser.set_defaults(wandb_project="dc-diagnostic-pre")
    args = parser.parse_args()

    experiments = [e.strip() for e in args.experiments.split(",")]
    device = detect_device()

    # ── wandb init ────────────────────────────────────────────────────
    wandb_run = init_wandb(
        enabled=args.wandb,
        project=args.wandb_project,
        run_name="pre-training-diag",
        config={"steps": args.steps, "probe_steps": args.probe_steps,
                "lr": args.lr, "experiments": experiments},
        entity=args.wandb_entity,
    )

    # ── setup ─────────────────────────────────────────────────────────
    print(f"Device: {device}")
    config = DeepCompressorConfig.from_yaml(args.config)

    print("\nLoading model (random init)...")
    model = load_model(config, device=device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters()) - trainable
    print(f"  Trainable: {trainable:,}  |  Frozen: {frozen:,}")

    print("\nPreparing single NTP batch...")
    batch, tokenizer = prepare_ntp_batch(
        config, args.data_path, args.batch_size, device,
        min_doc_tokens=args.min_doc_tokens,
    )
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f"  {k:<25} {str(tuple(v.shape)):>20}")

    print("\nPre-computing Qwen document features...")
    doc_hidden, doc_pooled = precompute_qwen_features(model, batch, device)
    print(f"  doc_hidden: {tuple(doc_hidden.shape)}")
    print(f"  doc_pooled: {tuple(doc_pooled.shape)}")

    # ── experiments ───────────────────────────────────────────────────
    all_results = {}

    if "1" in experiments:
        torch.manual_seed(config.training.seed)
        exp1 = run_overfit(
            model, doc_hidden, batch, device, args.steps, args.lr, args.log_every
        )
        all_results["overfit"] = exp1
        if wandb_run:
            for i, loss in enumerate(exp1["losses"]):
                log_wandb(wandb_run, {"exp1/loss": loss}, step=i + 1)

    if "2" in experiments:
        torch.manual_seed(config.training.seed)
        exp2 = run_gradient_flow(model, doc_hidden, batch, device)
        all_results["gradient_flow"] = exp2
        if wandb_run:
            for mod_name, layer_norms in exp2.items():
                for layer_name, gn in layer_norms.items():
                    log_wandb(wandb_run, {f"exp2/{mod_name}/{layer_name}": gn})

    if "3" in experiments:
        torch.manual_seed(config.training.seed)
        exp3 = run_bottleneck(
            model, doc_pooled, batch, device, args.probe_steps, args.lr, args.log_every
        )
        all_results["bottleneck"] = exp3
        if wandb_run:
            log_wandb(
                wandb_run,
                {
                    "exp3/random_loss": exp3["random_loss"],
                    "exp3/linear_final": exp3["linear_losses"][-1],
                    "exp3/mlp_final": exp3["mlp_losses"][-1],
                },
            )

    # ── cross comparison (if experiments 1 & 3 were both run) ─────────
    if "overfit" in all_results and "bottleneck" in all_results:
        overfit_final = all_results["overfit"]["losses"][-1]
        linear_final = all_results["bottleneck"]["linear_losses"][-1]
        mlp_final = all_results["bottleneck"]["mlp_losses"][-1]
        rnd = all_results["bottleneck"]["random_loss"]

        print("\n" + "=" * 76)
        print("  CROSS COMPARISON (Exp 1 vs Exp 3)")
        print(f"  Pipeline: {args.steps} steps  |  Probes: {args.probe_steps} steps")
        print("=" * 76)

        print(f"\n  {'Method':<35}  {'Loss':>10}  {'Steps':>7}")
        print(f"  {'─' * 56}")
        print(f"  {'Random prefix (no learning)':<35}  {rnd:>10.4f}  {'--':>7}")
        print(f"  {'Linear probe':<35}  {linear_final:>10.4f}  {args.probe_steps:>7}")
        print(f"  {'MLP probe (2-layer)':<35}  {mlp_final:>10.4f}  {args.probe_steps:>7}")
        print(f"  {'Full pipeline (Perceiver)':<35}  {overfit_final:>10.4f}  {args.steps:>7}")
        print(f"  {'─' * 56}")

        all_beat_random = linear_final < rnd and mlp_final < rnd
        pipeline_beats_mlp = overfit_final < mlp_final
        pipeline_beats_linear = overfit_final < linear_final

        print(f"\n  Conclusions:")
        if not all_beat_random:
            print("    [!] Not all probes beat random — training may need more steps")
        if pipeline_beats_mlp:
            print(
                f"    Full pipeline ({overfit_final:.4f}) < MLP ({mlp_final:.4f}) "
                f"< Linear ({linear_final:.4f})"
            )
            print(
                "    => Perceiver cross-attention extracts richer document features"
            )
            print("    => Architecture is working as intended")
        elif pipeline_beats_linear:
            print(
                f"    Linear ({linear_final:.4f}) > Full ({overfit_final:.4f}) "
                f"> MLP ({mlp_final:.4f})"
            )
            print(
                "    => Pipeline outperforms linear but not MLP — functioning, room to improve"
            )
        else:
            print(f"    Full ({overfit_final:.4f}) >= Linear ({linear_final:.4f})")
            print(
                "    => Pipeline not outperforming simple baseline — investigate architecture"
            )

    finish_wandb(wandb_run)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
