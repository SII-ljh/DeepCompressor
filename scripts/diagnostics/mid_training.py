#!/usr/bin/env python
"""Mid-training diagnostic experiments for Deep Compressor.

Exp 4 — Query Diversity/Utilization: Check query collapse, dead queries, effective rank.
Exp 5 — Stagewise Information Gain: Measure loss reduction contributed by each Perceiver stage.

Can be run standalone (with optional checkpoint) or as a callback during training.

Usage (standalone):
    python scripts/diagnostics/mid_training.py \
        --config configs/macbook_debug.yaml \
        --data_path data/ntp_tiny.jsonl \
        --checkpoint outputs/checkpoint-1000/trainable_weights.pt \
        --experiments 4,5

Usage (callback via train.py):
    python -m deep_compressor.train \
        --config configs/tiny_subset.yaml \
        --data_path data/ntp_tiny.jsonl --stage 1 \
        --diagnostic_every 50 --diagnostic_experiments 4,5
"""

from __future__ import annotations

import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

from scripts.diagnostics.common import (
    base_parser,
    detect_device,
    effective_rank,
    finish_wandb,
    init_wandb,
    load_model,
    log_wandb,
    pairwise_cosine_similarity,
    precompute_qwen_features,
    prepare_ntp_batch,
    to_device,
)
from deep_compressor.config import DeepCompressorConfig


ALL_EXPERIMENTS = ["4", "5"]


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 4 — Query Diversity / Utilization
# ═══════════════════════════════════════════════════════════════════════

def run_query_diversity(
    model,
    batch: dict,
    device: torch.device,
    doc_hidden: torch.Tensor | None = None,
) -> dict:
    """Analyze query diversity and utilization after Perceiver compression.

    Checks for query collapse, dead queries, and effective utilization of
    the latent bottleneck.
    """

    print("\n" + "=" * 76)
    print("  EXPERIMENT 4: Query Diversity / Utilization")
    print("=" * 76)

    model.eval()

    B = batch["doc_input_ids"].shape[0]
    D = model.config.qwen.hidden_size
    doc_mask = batch["doc_attention_mask"]
    num_queries = model.config.effective_num_queries

    with torch.no_grad():
        # Get byte_array from doc_hidden or recompute
        if doc_hidden is not None:
            byte_array = model.down_proj(doc_hidden)
        else:
            byte_array = model.encode_document(
                batch["doc_input_ids"], batch["doc_attention_mask"]
            )

        # Build queries (zero pooled = NTP mode)
        queries = model.query_init(torch.zeros(B, D, device=device))
        latent_array = model.perceiver(
            queries, byte_array, byte_mask=doc_mask
        )

        # Analyze first sample's latent array: (num_queries, perceiver_dim)
        latent = latent_array[0]  # (nq, dim)

        # Pairwise cosine similarity
        cos_sim = pairwise_cosine_similarity(latent)
        # Exclude diagonal
        mask = ~torch.eye(num_queries, device=device, dtype=torch.bool)
        off_diag = cos_sim[mask]

        mean_pairwise = off_diag.mean().item()
        max_pairwise = off_diag.max().item()

        # Effective rank
        eff_rank = effective_rank(latent)

        # Dead query detection (norm < 1% of mean)
        query_norms = latent.norm(dim=-1)
        norm_mean = query_norms.mean().item()
        norm_std = query_norms.std().item()
        threshold = norm_mean * 0.01
        num_dead = (query_norms < threshold).sum().item()

    results = {
        "mean_pairwise_cosine": mean_pairwise,
        "max_pairwise_cosine": max_pairwise,
        "effective_rank": eff_rank,
        "num_dead_queries": num_dead,
        "query_norm_mean": norm_mean,
        "query_norm_std": norm_std,
    }

    # ── report ────────────────────────────────────────────────────────
    print(f"\n  {'Metric':<40}  {'Value':>12}")
    print(f"  {'─' * 55}")
    print(f"  {'Mean pairwise cosine similarity':<40}  {mean_pairwise:>12.4f}")
    print(f"  {'Max pairwise cosine similarity':<40}  {max_pairwise:>12.4f}")
    print(f"  {'Effective rank':<40}  {eff_rank:>12.2f}")
    print(f"  {'Num queries':<40}  {num_queries:>12d}")
    print(f"  {'Dead queries (norm < 1% mean)':<40}  {num_dead:>12d}")
    print(f"  {'Query norm mean':<40}  {norm_mean:>12.4f}")
    print(f"  {'Query norm std':<40}  {norm_std:>12.4f}")
    print(f"  {'─' * 55}")

    # ── diagnosis ─────────────────────────────────────────────────────
    print(f"\n  DIAGNOSIS:")
    warnings = []
    if eff_rank < num_queries * 0.3:
        warnings.append(
            f"  WARNING: Low effective rank ({eff_rank:.1f} < {num_queries * 0.3:.0f}) "
            f"— queries under-utilized"
        )
    if num_dead > 0:
        warnings.append(
            f"  WARNING: {num_dead} dead queries detected — norm near zero"
        )
    if mean_pairwise > 0.9:
        warnings.append(
            f"  WARNING: High mean cosine similarity ({mean_pairwise:.3f} > 0.9) "
            f"— possible query collapse"
        )

    if warnings:
        for w in warnings:
            print(w)
    else:
        print(
            f"  PASS — Queries are diverse (rank={eff_rank:.1f}/{num_queries}, "
            f"cos={mean_pairwise:.3f}, {num_dead} dead)"
        )

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 5 — Stagewise Information Gain
# ═══════════════════════════════════════════════════════════════════════

def run_stagewise_info_gain(
    model,
    batch: dict,
    device: torch.device,
    doc_hidden: torch.Tensor | None = None,
) -> dict:
    """Measure loss at progressive truncation points through the Perceiver.

    Truncation points:
      - Initial queries (no Perceiver processing)
      - After Stage A
      - After Stage A + B
      - After Stage A + B + C (full pipeline)
    """

    print("\n" + "=" * 76)
    print("  EXPERIMENT 5: Stagewise Information Gain")
    print("=" * 76)

    model.eval()

    B = batch["doc_input_ids"].shape[0]
    D = model.config.qwen.hidden_size
    doc_mask = batch["doc_attention_mask"]
    seg_ids = batch["segment_ids"]
    seg_mask = batch["segment_attention_mask"]
    seg_labels = batch["segment_labels"]

    perceiver = model.perceiver

    with torch.no_grad():
        # Get byte_array
        if doc_hidden is not None:
            byte_array = model.down_proj(doc_hidden)
        else:
            byte_array = model.encode_document(
                batch["doc_input_ids"], batch["doc_attention_mask"]
            )

        queries = model.query_init(torch.zeros(B, D, device=device))

        # Helper: compute loss from intermediate latent
        def _loss_from_latent(x: torch.Tensor) -> float:
            normed = perceiver.final_norm(x)
            prefix = model.up_mlp(normed)
            out = model.decode(prefix, seg_ids, seg_mask, labels=seg_labels)
            return out.loss.item()

        # Collect truncation points
        checkpoints = {}
        x = queries

        # Point 0: initial queries (no processing)
        checkpoints["initial"] = _loss_from_latent(x)

        # Stage A
        if perceiver.enable_stage_a:
            for block in perceiver.stage_a_cross:
                x = block(x, kv=byte_array, kv_mask=doc_mask)
            for block in perceiver.stage_a_self:
                x = block(x)
            checkpoints["after_stage_a"] = _loss_from_latent(x)

        # Stage B
        if perceiver.enable_stage_b:
            for block in perceiver.stage_b_self:
                x = block(x)
            checkpoints["after_stage_b"] = _loss_from_latent(x)

        # Stage C
        if perceiver.enable_stage_c:
            for block in perceiver.stage_c_cross:
                x = block(x, kv=byte_array, kv_mask=doc_mask)
            for block in perceiver.stage_c_self:
                x = block(x)
            checkpoints["after_stage_c"] = _loss_from_latent(x)

    # ── report ────────────────────────────────────────────────────────
    stages = list(checkpoints.keys())
    losses = list(checkpoints.values())

    print(f"\n  {'Stage':<25}  {'Loss':>10}  {'Gain':>10}  {'Gain %':>10}")
    print(f"  {'─' * 60}")

    gains = {}
    total_gain = losses[0] - losses[-1] if len(losses) > 1 else 0.0

    for i, (stage, loss) in enumerate(zip(stages, losses)):
        if i == 0:
            gain = 0.0
            gain_pct = 0.0
        else:
            gain = losses[i - 1] - loss
            gain_pct = (gain / max(abs(total_gain), 1e-10)) * 100
        gains[stage] = gain
        print(f"  {stage:<25}  {loss:>10.4f}  {gain:>+10.4f}  {gain_pct:>9.1f}%")

    print(f"  {'─' * 60}")
    print(f"  {'Total gain':<25}  {'':>10}  {total_gain:>+10.4f}  {'100.0':>9}%")

    # ── diagnosis ─────────────────────────────────────────────────────
    print(f"\n  DIAGNOSIS:")
    if total_gain < 0.01:
        print("  WARNING: Negligible total gain — Perceiver may not be learning")
    else:
        # Find which stage contributes most
        gain_items = [(k, v) for k, v in gains.items() if k != "initial"]
        if gain_items:
            best_stage = max(gain_items, key=lambda x: x[1])
            print(f"  Largest contribution: {best_stage[0]} ({best_stage[1]:+.4f})")

        # Check for negative gains (stages hurting performance)
        negative = [(k, v) for k, v in gain_items if v < -0.01]
        if negative:
            for k, v in negative:
                print(f"  WARNING: {k} increases loss by {-v:.4f}")
        else:
            print("  PASS — All active stages contribute positively")

    results = {"checkpoints": checkpoints, "gains": gains, "total_gain": total_gain}

    # Flatten for easy wandb logging
    for stage, loss in checkpoints.items():
        results[f"loss_{stage}"] = loss
    for stage, gain in gains.items():
        results[f"gain_{stage}"] = gain

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Callback API
# ═══════════════════════════════════════════════════════════════════════

def create_mid_training_callback(
    config: DeepCompressorConfig,
    data_path: str,
    experiments: tuple[str, ...] = ("4", "5"),
    run_every: int = 1000,
    wandb_run=None,
    batch_size: int = 2,
):
    """Create a callback function for mid-training diagnostics.

    Returns:
        callable(step, model, accelerator) -> dict | None
    """
    device = detect_device()
    _batch_cache = {}

    def callback(step: int, model, accelerator) -> dict | None:
        if step % run_every != 0:
            return None

        # Prepare batch (cached after first call)
        if "batch" not in _batch_cache:
            batch, _ = prepare_ntp_batch(config, data_path, batch_size, device)
            _batch_cache["batch"] = batch
        batch = _batch_cache["batch"]

        # Unwrap DDP model
        unwrapped = accelerator.unwrap_model(model)
        was_training = unwrapped.training
        unwrapped.eval()

        # Pre-compute features
        doc_hidden, _ = precompute_qwen_features(unwrapped, batch, device)

        results = {}

        if "4" in experiments:
            exp4 = run_query_diversity(unwrapped, batch, device, doc_hidden=doc_hidden)
            for k, v in exp4.items():
                if isinstance(v, (int, float)):
                    results[f"query_diversity/{k}"] = v

        if "5" in experiments:
            exp5 = run_stagewise_info_gain(
                unwrapped, batch, device, doc_hidden=doc_hidden
            )
            for k, v in exp5.items():
                if isinstance(v, (int, float)):
                    results[f"stagewise_gain/{k}"] = v

        # Log to wandb if available
        if wandb_run is not None:
            log_wandb(
                wandb_run,
                {f"diag/{k}": v for k, v in results.items()},
                step=step,
            )

        if was_training:
            unwrapped.train()

        return results

    return callback


# ═══════════════════════════════════════════════════════════════════════
#  Main (standalone mode)
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = base_parser("Deep Compressor mid-training diagnostics (Exp 4-5)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (optional, uses random init if not provided)",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default="4,5",
        help="Comma-separated list of experiments to run (4,5)",
    )
    parser.set_defaults(wandb_project="dc-diagnostic-mid")
    args = parser.parse_args()

    experiments = [e.strip() for e in args.experiments.split(",")]
    device = detect_device()

    wandb_run = init_wandb(
        enabled=args.wandb,
        project=args.wandb_project,
        run_name="mid-training-diag",
        config={"checkpoint": args.checkpoint, "experiments": experiments},
        entity=args.wandb_entity,
    )

    print(f"Device: {device}")
    config = DeepCompressorConfig.from_yaml(args.config)

    print("\nLoading model...")
    model = load_model(config, checkpoint_path=args.checkpoint, device=device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters()) - trainable
    print(f"  Trainable: {trainable:,}  |  Frozen: {frozen:,}")

    print("\nPreparing single NTP batch...")
    batch, tokenizer = prepare_ntp_batch(
        config, args.data_path, args.batch_size, device
    )

    print("\nPre-computing Qwen document features...")
    doc_hidden, doc_pooled = precompute_qwen_features(model, batch, device)
    print(f"  doc_hidden: {tuple(doc_hidden.shape)}")

    # ── experiments ───────────────────────────────────────────────────
    all_results = {}

    if "4" in experiments:
        torch.manual_seed(config.training.seed)
        exp4 = run_query_diversity(model, batch, device, doc_hidden=doc_hidden)
        all_results["query_diversity"] = exp4
        if wandb_run:
            log_wandb(
                wandb_run,
                {f"exp4/{k}": v for k, v in exp4.items() if isinstance(v, (int, float))},
            )

    if "5" in experiments:
        torch.manual_seed(config.training.seed)
        exp5 = run_stagewise_info_gain(model, batch, device, doc_hidden=doc_hidden)
        all_results["stagewise_gain"] = exp5
        if wandb_run:
            log_wandb(
                wandb_run,
                {f"exp5/{k}": v for k, v in exp5.items() if isinstance(v, (int, float))},
            )

    finish_wandb(wandb_run)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
