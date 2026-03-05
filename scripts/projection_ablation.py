#!/usr/bin/env python3
"""Projection layer ablation across multiple Qwen3 models.

Compares 5 projection configurations (down_proj x up_proj combinations)
for each specified Qwen3 model size. Supports incremental execution:
run one config at a time and results are merged into a single JSON file.

Projection configurations:
  mlp_mlp           MLP down + MLP up (baseline)
  identity_identity Identity down + Identity up (no projection)
  linear_linear     Linear down + Linear up
  linear_mlp        Linear down + MLP up
  mlp_linear        MLP down + Linear up

Supported models: 0.6B, 1.7B, 4B, 8B

Each config runs: NTP Stage 1 -> QA Stage 2 -> Eval on dev set.

Usage:
  # Run all projection configs for one model
  python scripts/projection_ablation.py --model 0.6B

  # Run specific configs for a specific model
  python scripts/projection_ablation.py --model 4B --configs mlp_mlp,linear_linear

  # Run all configs for all models
  python scripts/projection_ablation.py --model all

  # Fast smoke test
  python scripts/projection_ablation.py --model 0.6B --fast

  # List available configs and models
  python scripts/projection_ablation.py --list

Results are saved incrementally to outputs/proj_ablation/results.json.
Each result entry records the model size and projection config, so
running different models/configs at different times will merge cleanly.
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deep_compressor.config import (
    QWEN3_REGISTRY,
    AblationConfig,
    DeepCompressorConfig,
    FinBERTConfig,
    LossConfig,
    PerceiverConfig,
    ProjectionConfig,
    QwenConfig,
    TrainingConfig,
)
from deep_compressor.data import NTPDataset, PaddingCollator, QADataset
from deep_compressor.model import DeepCompressor
from deep_compressor.train import train_stage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("proj_ablation")

# ═══════════════════════════════════════════════════════════════════════
# Model registry: size label -> HuggingFace name
# ═══════════════════════════════════════════════════════════════════════
MODEL_SIZES = {
    "0.6B": "Qwen/Qwen3-0.6B",
    "1.7B": "Qwen/Qwen3-1.7B",
    "4B":   "Qwen/Qwen3-4B",
    "8B":   "Qwen/Qwen3-8B",
}

# ═══════════════════════════════════════════════════════════════════════
# 5 projection configurations to compare
# ═══════════════════════════════════════════════════════════════════════
ABLATION_CONFIGS = [
    {
        "name": "mlp_mlp",
        "desc": "MLP down + MLP up (baseline)",
        "down_proj_mode": "mlp",
        "up_proj_mode": "mlp",
    },
    {
        "name": "identity_identity",
        "desc": "Identity down + Identity up (no projection)",
        "down_proj_mode": "identity",
        "up_proj_mode": "identity",
    },
    {
        "name": "linear_linear",
        "desc": "Linear down + Linear up",
        "down_proj_mode": "linear",
        "up_proj_mode": "linear",
    },
    {
        "name": "linear_mlp",
        "desc": "Linear down + MLP up",
        "down_proj_mode": "linear",
        "up_proj_mode": "mlp",
    },
    {
        "name": "mlp_linear",
        "desc": "MLP down + Linear up",
        "down_proj_mode": "mlp",
        "up_proj_mode": "linear",
    },
]


def _resolve_model_path(model_size: str, args) -> str:
    """Resolve model path: try local models/ dir first, fall back to HuggingFace name."""
    local_path = Path(args.model_dir) / f"Qwen3-{model_size}"
    if local_path.exists():
        return str(local_path)
    return MODEL_SIZES[model_size]


def _distill_layers_for(num_layers: int):
    """Compute evenly-spaced distillation layer indices for a given layer count."""
    step = num_layers // 4
    return [step - 1, 2 * step - 1, 3 * step - 1, num_layers - 1]


def build_config(acfg, model_path, model_size, args, stage):
    """Build DeepCompressorConfig for a given projection config, model, and stage."""
    specs = QWEN3_REGISTRY[MODEL_SIZES[model_size]]
    hidden_size = specs["hidden_size"]
    num_layers = specs["num_hidden_layers"]

    max_steps = args.ntp_steps if stage == 1 else args.qa_steps
    warmup = max(10, max_steps // 20)

    # MLP hidden dim scales with model hidden_size (no bottleneck)
    mlp_hidden = hidden_size

    return DeepCompressorConfig(
        qwen=QwenConfig(
            model_name_or_path=model_path,
            max_doc_tokens=512,
            max_question_tokens=64,
            max_answer_tokens=64,
        ),
        finbert=FinBERTConfig(enabled=False),
        perceiver=PerceiverConfig(
            num_queries=64,
            num_heads=16,
            stage_a_cross_layers=2,
            stage_a_self_layers=2,
            stage_b_layers=2,
            stage_c_cross_layers=2,
            stage_c_self_layers=4,
            ff_mult=4,
            anchor_score_scale_init=1.0,
            dropout=0.1,
        ),
        projection=ProjectionConfig(
            down_hidden=mlp_hidden,
            up_hidden=mlp_hidden,
            dropout=0.1,
        ),
        loss=LossConfig(
            kl_temperature=2.0,
            hidden_distill_ramp_steps=200,
            hidden_distill_layers=_distill_layers_for(num_layers),
            qa_ce_weight=1.0,
            kl_weight=0.0,
            hidden_mse_weight=0.0,
            anchor_recon_weight=0.0,
        ),
        training=TrainingConfig(
            stage=stage,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_steps=max_steps,
            warmup_steps=warmup,
            weight_decay=0.01,
            max_grad_norm=1.0,
            scheduler="cosine",
            seed=42,
            log_every=max(1, max_steps // 20),
            eval_every=max(1, max_steps // 2),
            save_every=999999,  # no checkpoints during ablation
            output_dir=f"outputs/proj_ablation/{model_size}/{acfg['name']}",
            ntp_segment_len=64,
            gradient_checkpointing=True,
            mixed_precision=args.mixed_precision,
        ),
        ablation=AblationConfig(
            down_proj_mode=acfg["down_proj_mode"],
            up_proj_mode=acfg["up_proj_mode"],
        ),
    )


def run_one_config(acfg, model_size, model_path, tokenizer, args):
    """Train NTP -> QA -> Eval for one projection configuration on one model."""
    name = acfg["name"]
    run_id = f"{model_size}/{name}"
    logger.info(f"\n{'='*70}")
    logger.info(f"  [{run_id}] {acfg['desc']}")
    logger.info(f"{'='*70}")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    t0 = time.time()
    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
    is_cuda = torch.cuda.is_available()
    num_workers = 4 if is_cuda else 0
    pin_memory = is_cuda

    # ── Stage 1: NTP ──
    logger.info(f"  [{run_id}] Stage 1: NTP ({args.ntp_steps} steps)")
    ntp_config = build_config(acfg, model_path, model_size, args, stage=1)
    ntp_model = DeepCompressor(ntp_config)
    if ntp_config.training.gradient_checkpointing:
        ntp_model.qwen.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in ntp_model.parameters() if p.requires_grad)
    proj_params = sum(
        p.numel()
        for n in ("down_proj", "up_mlp")
        for p in getattr(ntp_model, n).parameters()
    )
    logger.info(f"  [{run_id}] Model: Qwen3-{model_size} "
                f"(hidden={ntp_config.qwen.hidden_size}, "
                f"layers={ntp_config.qwen.num_hidden_layers})")
    logger.info(f"  [{run_id}] Trainable: {trainable:,}  (projection: {proj_params:,})")

    ntp_ds = NTPDataset(args.ntp_data, tokenizer,
                        max_doc_tokens=512, segment_len=64)
    n_total = len(ntp_ds)
    n_val = min(100, n_total // 10) if n_total > 10 else 0
    ntp_train = Subset(ntp_ds, list(range(n_total - n_val)))
    ntp_val = Subset(ntp_ds, list(range(n_total - n_val, n_total))) if n_val else None

    ntp_loader = DataLoader(ntp_train, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collator, num_workers=num_workers,
                            pin_memory=pin_memory)
    ntp_eval_loader = None
    if ntp_val:
        ntp_eval_loader = DataLoader(ntp_val, batch_size=args.batch_size,
                                     shuffle=False, collate_fn=collator,
                                     num_workers=num_workers, pin_memory=pin_memory)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    ntp_accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )

    ntp_model, ntp_metrics = train_stage(
        ntp_config, ntp_model, ntp_loader, ntp_accelerator,
        mode="ntp", eval_loader=ntp_eval_loader, tokenizer=tokenizer,
    )

    ntp_loss = ntp_metrics["loss"] if ntp_metrics else float("nan")
    ntp_ppl = ntp_metrics["perplexity"] if ntp_metrics else float("nan")
    t_ntp = time.time() - t0

    # Extract trained weights for Stage 2
    unwrapped_ntp = ntp_accelerator.unwrap_model(ntp_model)
    trained_state = {k: v.cpu().clone() for k, v in unwrapped_ntp.state_dict().items()
                     if not k.startswith("qwen.")}

    # Cleanup Stage 1
    del ntp_model, ntp_accelerator, ntp_loader, ntp_eval_loader
    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()

    # ── Stage 2: QA ──
    logger.info(f"  [{run_id}] Stage 2: QA ({args.qa_steps} steps)")
    qa_config = build_config(acfg, model_path, model_size, args, stage=2)
    qa_model = DeepCompressor(qa_config)
    if qa_config.training.gradient_checkpointing:
        qa_model.qwen.gradient_checkpointing_enable()

    # Load NTP-trained weights
    qa_model.load_state_dict(trained_state, strict=False)
    del trained_state

    qa_ds = QADataset(args.qa_train, tokenizer,
                      max_doc_tokens=512, max_question_tokens=64,
                      max_answer_tokens=64)
    qa_loader = DataLoader(qa_ds, batch_size=args.batch_size, shuffle=True,
                           collate_fn=collator, num_workers=num_workers,
                           pin_memory=pin_memory)

    qa_eval_loader = None
    if args.qa_dev:
        qa_eval_ds = QADataset(args.qa_dev, tokenizer,
                               max_doc_tokens=512, max_question_tokens=64,
                               max_answer_tokens=64)
        if args.max_eval_samples and len(qa_eval_ds) > args.max_eval_samples:
            qa_eval_ds = Subset(qa_eval_ds, list(range(args.max_eval_samples)))
        qa_eval_loader = DataLoader(qa_eval_ds, batch_size=args.batch_size,
                                    shuffle=False, collate_fn=collator,
                                    num_workers=num_workers, pin_memory=pin_memory)
        logger.info(f"  [{run_id}] QA train: {len(qa_ds):,}  dev: {len(qa_eval_ds):,}")

    ddp_kwargs2 = DistributedDataParallelKwargs(find_unused_parameters=True)
    qa_accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[ddp_kwargs2],
    )

    qa_model, qa_metrics = train_stage(
        qa_config, qa_model, qa_loader, qa_accelerator,
        mode="qa", eval_loader=qa_eval_loader, tokenizer=tokenizer,
    )

    elapsed = time.time() - t0

    # Extract final metrics
    em = qa_metrics.get("exact_match", 0.0) if qa_metrics else 0.0
    f1 = qa_metrics.get("f1", 0.0) if qa_metrics else 0.0

    logger.info(f"  [{run_id}] DONE  EM={em:.2%}  F1={f1:.4f}  "
                f"NTP_ppl={ntp_ppl:.2f}  time={elapsed:.0f}s")

    # Cleanup Stage 2
    del qa_model, qa_accelerator, qa_loader, qa_eval_loader
    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()

    return {
        "model": f"Qwen3-{model_size}",
        "model_size": model_size,
        "hidden_size": qa_config.qwen.hidden_size,
        "num_layers": qa_config.qwen.num_hidden_layers,
        "name": name,
        "desc": acfg["desc"],
        "down_proj_mode": acfg["down_proj_mode"],
        "up_proj_mode": acfg["up_proj_mode"],
        "trainable_params": trainable,
        "proj_params": proj_params,
        "ntp_loss": ntp_loss,
        "ntp_ppl": ntp_ppl,
        "ntp_time": t_ntp,
        "em": em,
        "f1": f1,
        "elapsed": elapsed,
    }


def _result_key(r):
    """Unique key for a result entry (model + projection config)."""
    return f"{r['model_size']}_{r['name']}"


def save_results(all_results, json_path):
    """Incrementally save results, merging with any existing file."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    existing = []
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = []
    # Merge: new results overwrite old ones with the same model+config key
    merged = {_result_key(r): r for r in existing}
    for r in all_results:
        merged[_result_key(r)] = r
    with open(json_path, "w") as f:
        json.dump(list(merged.values()), f, indent=2, ensure_ascii=False)
    return list(merged.values())


def print_comparison(all_results, json_path):
    """Print comparison table grouped by model, sorted by F1 within each group."""
    # Group by model
    by_model = {}
    for r in all_results:
        by_model.setdefault(r["model_size"], []).append(r)

    print("\n" + "=" * 100)
    print("  PROJECTION ABLATION RESULTS")
    print("=" * 100)

    for model_size in sorted(by_model.keys(),
                             key=lambda s: list(MODEL_SIZES.keys()).index(s)
                             if s in MODEL_SIZES else 99):
        results = by_model[model_size]
        ranked = sorted(results, key=lambda r: r["f1"], reverse=True)
        hidden = ranked[0]["hidden_size"] if ranked else "?"
        n_layers = ranked[0]["num_layers"] if ranked else "?"

        print(f"\n  Qwen3-{model_size} (hidden={hidden}, layers={n_layers})")
        print(f"  {'#':<3} {'Config':<22} {'Down':<10} {'Up':<10} "
              f"{'Proj Params':<13} {'NTP PPL':<9} {'EM':<8} {'F1':<8} {'Time':<7}")
        print("  " + "-" * 90)

        baseline = next((r for r in ranked if r["name"] == "mlp_mlp"), None)

        for i, r in enumerate(ranked):
            proj_str = f"{r['proj_params']:,}" if r['proj_params'] > 0 else "0"
            mark = " *" if i == 0 else ""
            print(f"  {i+1:<3} {r['name']:<22} {r['down_proj_mode']:<10} "
                  f"{r['up_proj_mode']:<10} {proj_str:<13} "
                  f"{r['ntp_ppl']:<9.2f} {r['em']:.2%}{'':>2} {r['f1']:.4f}  "
                  f"{r['elapsed']:.0f}s{mark}")

        if baseline and len(ranked) > 1:
            best = ranked[0]
            if best["name"] != baseline["name"]:
                delta_f1 = best["f1"] - baseline["f1"]
                delta_em = best["em"] - baseline["em"]
                print(f"\n    Best vs baseline (mlp_mlp): "
                      f"EM {delta_em:+.2%}  F1 {delta_f1:+.4f}")

    # Save
    merged = save_results(all_results, json_path)
    print(f"\n  Results saved to {json_path} ({len(merged)} entries total)")
    print("=" * 100)


def list_all():
    """Print available models and projection configs."""
    print("\nAvailable Qwen3 models:")
    print("-" * 55)
    for size, name in MODEL_SIZES.items():
        specs = QWEN3_REGISTRY[name]
        print(f"  {size:6s}  {name:20s}  hidden={specs['hidden_size']}, "
              f"layers={specs['num_hidden_layers']}")

    print("\nProjection configurations:")
    print("-" * 55)
    for cfg in ABLATION_CONFIGS:
        print(f"  {cfg['name']:<22s}  {cfg['desc']}")

    print(f"\nTotal combinations: {len(MODEL_SIZES)} models x "
          f"{len(ABLATION_CONFIGS)} configs = "
          f"{len(MODEL_SIZES) * len(ABLATION_CONFIGS)}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Projection layer ablation across Qwen3 models")

    # Model selection
    parser.add_argument("--model", type=str, default="0.6B",
                        help="Model size: 0.6B, 1.7B, 4B, 8B, or 'all' (default: 0.6B)")
    parser.add_argument("--model_dir", type=str, default="models",
                        help="Local directory for downloaded models (default: models)")

    # Data paths
    parser.add_argument("--ntp_data", type=str,
                        default="data/ablation/ntp_ablation.jsonl")
    parser.add_argument("--qa_train", type=str,
                        default="data/ablation/qa_ablation_train.json")
    parser.add_argument("--qa_dev", type=str, default=None,
                        help="QA dev set for evaluation (default: auto-detect)")

    # Training steps
    parser.add_argument("--ntp_steps", type=int, default=300,
                        help="NTP pretraining steps per config (default: 300)")
    parser.add_argument("--qa_steps", type=int, default=300,
                        help="QA fine-tuning steps per config (default: 300)")
    parser.add_argument("--lr", type=float, default=5e-4)

    # Batch / hardware
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Micro-batch size per GPU (default: 16)")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])

    # Eval
    parser.add_argument("--max_eval_samples", type=int, default=200,
                        help="Max QA dev samples for evaluation (default: 200)")

    # Config selection
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated config names (default: all 5)")

    # Fast mode
    parser.add_argument("--fast", action="store_true",
                        help="Smoke-test mode: 100 steps, 50 eval samples")

    # Output
    parser.add_argument("--output", type=str,
                        default="outputs/proj_ablation/results.json",
                        help="Path to save results JSON")

    # Info
    parser.add_argument("--list", action="store_true",
                        help="List available models and configs, then exit")

    args = parser.parse_args()

    if args.list:
        list_all()
        return

    # --fast overrides
    if args.fast:
        args.ntp_steps = 100
        args.qa_steps = 100
        args.max_eval_samples = 50
        logger.info("Fast mode: ntp_steps=100, qa_steps=100, max_eval_samples=50")

    # Resolve model sizes
    if args.model.lower() == "all":
        model_sizes = list(MODEL_SIZES.keys())
    else:
        model_sizes = [s.strip() for s in args.model.split(",")]
        for s in model_sizes:
            if s not in MODEL_SIZES:
                logger.error(f"Unknown model size: '{s}'. "
                             f"Available: {', '.join(MODEL_SIZES.keys())}")
                sys.exit(1)

    # Auto-detect QA dev path
    if args.qa_dev is None:
        candidate = args.qa_train.replace("_train.", "_dev.")
        if os.path.exists(candidate):
            args.qa_dev = candidate
            logger.info(f"Auto-detected QA dev: {args.qa_dev}")
        else:
            logger.warning("No QA dev set found -- eval will be skipped")

    # Validate data paths
    for label, path in [("ntp_data", args.ntp_data), ("qa_train", args.qa_train)]:
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            logger.error("Run 'python scripts/prepare_data.py && "
                         "python scripts/prepare_data.py --make-ablation' first")
            sys.exit(1)

    # Filter configs
    configs = ABLATION_CONFIGS
    if args.configs:
        names = set(args.configs.split(","))
        configs = [c for c in configs if c["name"] in names]
        if not configs:
            logger.error(f"No matching configs. Available: "
                         f"{[c['name'] for c in ABLATION_CONFIGS]}")
            sys.exit(1)

    total_runs = len(model_sizes) * len(configs)
    logger.info(f"Models: {model_sizes}")
    logger.info(f"Configs: {[c['name'] for c in configs]}")
    logger.info(f"Total runs: {total_runs}")

    eff_batch = args.batch_size * args.grad_accum
    logger.info(f"batch_size={args.batch_size} x grad_accum={args.grad_accum} "
                f"= eff_batch {eff_batch}")
    logger.info(f"mixed_precision={args.mixed_precision}")
    logger.info(f"NTP steps={args.ntp_steps}, QA steps={args.qa_steps}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.0f} GB)")

    all_results = []
    run_idx = 0

    for model_size in model_sizes:
        model_path = _resolve_model_path(model_size, args)
        logger.info(f"\n{'#'*70}")
        logger.info(f"  Loading tokenizer for Qwen3-{model_size}: {model_path}")
        logger.info(f"{'#'*70}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        for acfg in configs:
            run_idx += 1
            logger.info(f"\n>>> Run {run_idx}/{total_runs}: "
                        f"Qwen3-{model_size} / {acfg['name']}")
            result = run_one_config(acfg, model_size, model_path, tokenizer, args)
            all_results.append(result)

            # Save incrementally after each run
            save_results(all_results, args.output)

            # Print running comparison
            if len(all_results) > 1:
                print_comparison(all_results, args.output)

    # Final comparison
    print_comparison(all_results, args.output)


if __name__ == "__main__":
    main()
