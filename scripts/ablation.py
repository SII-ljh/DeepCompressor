#!/usr/bin/env python3
"""Ablation experiment runner for Deep Compressor.

Runs multiple ablation configurations against a shared Qwen backbone,
comparing each variant against the full pipeline baseline.

Usage:
  # Run all ablations for Stage 1 (NTP)
  python scripts/ablation.py --stage 1 \
      --data_path data/ablation/ntp_ablation.jsonl \
      --config configs/ablation_base.yaml

  # Run specific ablations
  python scripts/ablation.py --stage 1 \
      --data_path data/ablation/ntp_ablation.jsonl \
      --ablation down_proj_identity,no_stage_c,full_pipeline

  # List all available ablation experiments
  python scripts/ablation.py --list

  # With wandb tracking
  python scripts/ablation.py --stage 1 \
      --data_path data/ablation/ntp_ablation.jsonl \
      --wandb --wandb_project dc-ablation
"""

import argparse
import copy
import gc
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from deep_compressor.config import AblationConfig, DeepCompressorConfig
from deep_compressor.train import _run_training

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Ablation Registry ─────────────────────────────────────────────────

ABLATION_REGISTRY = {
    # Projection ablations
    "down_proj_identity": {"ablation": {"down_proj_mode": "identity"}},
    "down_proj_linear":   {"ablation": {"down_proj_mode": "linear"}},
    "up_proj_identity":   {"ablation": {"up_proj_mode": "identity"}},
    "up_proj_linear":     {"ablation": {"up_proj_mode": "linear"}},

    # QueryInit ablation
    "query_no_question":  {"ablation": {"query_condition_on_question": False}},

    # Perceiver stage ablations
    "no_stage_a":         {"ablation": {"enable_stage_a": False}},
    "no_stage_b":         {"ablation": {"enable_stage_b": False}},
    "no_stage_c":         {"ablation": {"enable_stage_c": False}},

    # Perceiver depth ablations
    "shallow_perceiver":  {"ablation": {
        "override_stage_a_cross_layers": 1,
        "override_stage_a_self_layers": 1,
        "override_stage_b_layers": 1,
        "override_stage_c_cross_layers": 1,
        "override_stage_c_self_layers": 1,
    }},
    "deep_perceiver":     {"ablation": {
        "override_stage_a_cross_layers": 4,
        "override_stage_a_self_layers": 4,
        "override_stage_b_layers": 4,
        "override_stage_c_cross_layers": 4,
        "override_stage_c_self_layers": 8,
    }},

    # Distillation ablations
    "no_kl_distill":      {"ablation": {"enable_kl_distillation": False}},
    "no_mse_distill":     {"ablation": {"enable_hidden_mse_distillation": False}},
    "no_distillation":    {"ablation": {
        "enable_kl_distillation": False,
        "enable_hidden_mse_distillation": False,
    }},

    # num_queries ablations
    "queries_16":         {"ablation": {"override_num_queries": 16}},
    "queries_32":         {"ablation": {"override_num_queries": 32}},
    "queries_128":        {"ablation": {"override_num_queries": 128}},

    # Full pipeline (baseline control)
    "full_pipeline":      {},
}


def _apply_overrides(config: DeepCompressorConfig, overrides: dict) -> None:
    """Apply ablation overrides to a config in-place, then re-validate."""
    ablation_overrides = overrides.get("ablation", {})
    for key, value in ablation_overrides.items():
        if not hasattr(config.ablation, key):
            raise ValueError(f"Unknown ablation field: {key}")
        setattr(config.ablation, key, value)
    # Re-trigger validation after overrides
    config.__post_init__()


def _list_ablations():
    """Print all available ablation experiments."""
    print("\nAvailable ablation experiments:")
    print(f"  {'Name':<25}  {'Description'}")
    print(f"  {'─' * 70}")
    descriptions = {
        "down_proj_identity": "Skip down-projection (identity)",
        "down_proj_linear": "Single linear down-projection",
        "up_proj_identity": "Skip up-projection (identity)",
        "up_proj_linear": "Single linear up-projection",
        "query_no_question": "QueryInit without question conditioning",
        "no_stage_a": "Disable Perceiver Stage A (global cross-attn)",
        "no_stage_b": "Disable Perceiver Stage B (anchor/self-attn)",
        "no_stage_c": "Disable Perceiver Stage C (deep reasoning)",
        "shallow_perceiver": "All stages with 1 layer each",
        "deep_perceiver": "All stages with 4+ layers each",
        "no_kl_distill": "Disable KL distillation loss",
        "no_mse_distill": "Disable hidden-state MSE loss",
        "no_distillation": "Disable all distillation losses",
        "queries_16": "Use 16 queries instead of default",
        "queries_32": "Use 32 queries instead of default",
        "queries_128": "Use 128 queries instead of default",
        "full_pipeline": "Full pipeline (baseline control)",
    }
    for name in ABLATION_REGISTRY:
        desc = descriptions.get(name, "")
        print(f"  {name:<25}  {desc}")
    print()


def run_ablation_suite(
    args,
    ablation_names: list,
):
    """Run a suite of ablation experiments and collect results."""

    results = {}

    for i, name in enumerate(ablation_names):
        if name not in ABLATION_REGISTRY:
            logger.error(f"Unknown ablation: {name}")
            continue

        overrides = ABLATION_REGISTRY[name]

        logger.info(f"\n{'=' * 60}")
        logger.info(f"  Ablation {i+1}/{len(ablation_names)}: {name}")
        logger.info(f"  Overrides: {overrides}")
        logger.info(f"{'=' * 60}")

        # Build fresh config each time
        config = DeepCompressorConfig.from_yaml(args.config)
        config.training.stage = args.stage
        config.training.output_dir = os.path.join(
            args.output_dir, f"ablation_{name}")

        # Apply ablation overrides
        _apply_overrides(config, overrides)

        # Build wandb config
        from types import SimpleNamespace
        if args.wandb:
            wandb_conf = SimpleNamespace(
                enabled=True,
                project=args.wandb_project,
                entity=args.wandb_entity,
                run_name=f"ablation-{name}",
                tags=["ablation", name, f"stage{args.stage}"],
                offline=False,
            )
        else:
            wandb_conf = SimpleNamespace(enabled=False)

        t0 = time.time()
        try:
            last_metrics = _run_training(
                config=config,
                data_path=args.data_path,
                eval_data_path=args.eval_data_path,
                max_eval_samples=args.max_eval_samples,
                max_train_samples=args.max_train_samples,
                wandb_conf=wandb_conf,
            )
            elapsed = time.time() - t0
            results[name] = {
                "metrics": last_metrics or {},
                "elapsed": elapsed,
                "status": "OK",
            }
            logger.info(f"  {name}: completed in {elapsed:.0f}s")
            if last_metrics:
                for k, v in last_metrics.items():
                    logger.info(f"    {k}: {v:.4f}")
        except Exception as e:
            elapsed = time.time() - t0
            results[name] = {
                "metrics": {},
                "elapsed": elapsed,
                "status": f"FAILED: {e}",
            }
            logger.error(f"  {name}: FAILED after {elapsed:.0f}s — {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


def _print_comparison_table(results: dict, stage: int):
    """Print a formatted comparison table of all ablation results."""
    print("\n" + "=" * 80)
    print("  ABLATION COMPARISON")
    print("=" * 80)

    if stage == 1:
        metric_key = "loss"
        metric_label = "Loss"
        better = "lower"
    else:
        metric_key = "f1"
        metric_label = "F1"
        better = "higher"

    print(f"\n  {'Ablation':<25}  {metric_label:>10}  {'Time':>8}  {'Status':<15}")
    print(f"  {'─' * 65}")

    baseline_val = None
    for name, result in results.items():
        metrics = result["metrics"]
        status = result["status"]
        elapsed = result["elapsed"]

        val = metrics.get(metric_key)
        if val is not None:
            val_str = f"{val:.4f}"
            if name == "full_pipeline":
                baseline_val = val
        else:
            val_str = "N/A"

        print(f"  {name:<25}  {val_str:>10}  {elapsed:>7.0f}s  {status:<15}")

    # Show deltas from baseline
    if baseline_val is not None:
        print(f"\n  Delta from full_pipeline ({metric_label}={baseline_val:.4f}):")
        for name, result in results.items():
            if name == "full_pipeline":
                continue
            val = result["metrics"].get(metric_key)
            if val is not None:
                delta = val - baseline_val
                direction = "worse" if (stage == 1 and delta > 0) or (stage == 2 and delta < 0) else "better"
                print(f"    {name:<25}  {delta:>+.4f}  ({direction})")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Ablation experiment runner for Deep Compressor")
    parser.add_argument("--config", type=str,
                        default="configs/ablation_base.yaml",
                        help="Base config YAML")
    parser.add_argument("--stage", type=int, default=1,
                        help="Training stage (1 or 2)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to training data")
    parser.add_argument("--eval_data_path", type=str, default=None,
                        help="Path to eval data")
    parser.add_argument("--ablation", type=str, default=None,
                        help="Comma-separated list of ablation names to run")
    parser.add_argument("--list", action="store_true",
                        help="List all available ablation experiments")
    parser.add_argument("--output_dir", type=str, default="outputs/ablation",
                        help="Base output directory")
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=0)
    # wandb options
    parser.add_argument("--wandb", action="store_true",
                        help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="dc-ablation")
    parser.add_argument("--wandb_entity", type=str, default=None)
    args = parser.parse_args()

    if args.list:
        _list_ablations()
        return

    if args.data_path is None:
        parser.error("--data_path is required (unless using --list)")

    # Determine which ablations to run
    if args.ablation:
        ablation_names = [n.strip() for n in args.ablation.split(",")]
    else:
        ablation_names = list(ABLATION_REGISTRY.keys())

    logger.info(f"Running {len(ablation_names)} ablation(s): {ablation_names}")

    results = run_ablation_suite(args, ablation_names)
    _print_comparison_table(results, args.stage)


if __name__ == "__main__":
    main()
