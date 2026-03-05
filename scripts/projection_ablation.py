#!/usr/bin/env python3
"""Projection layer ablation on real data with GPU optimization.

Compares 5 projection configurations using the ablation dataset.
Leverages Accelerate for bf16 mixed precision and large batch sizes.

Each config runs: NTP Stage 1 → QA Stage 2 → Eval on dev set.

Usage:
  # On H100 (default settings optimized for 80GB VRAM)
  python scripts/projection_ablation.py \
      --ntp_data data/ablation/ntp_ablation.jsonl \
      --qa_train data/ablation/qa_ablation_train.json \
      --qa_dev   data/ablation/qa_ablation_dev.json

  # Quick test (fewer steps)
  python scripts/projection_ablation.py \
      --ntp_data data/ablation/ntp_ablation.jsonl \
      --qa_train data/ablation/qa_ablation_train.json \
      --qa_dev   data/ablation/qa_ablation_dev.json \
      --ntp_steps 500 --qa_steps 500

  # Run subset of configs
  python scripts/projection_ablation.py ... --configs A_mlp768,B_mlp1024
"""

import argparse
import copy
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deep_compressor.config import (
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
from deep_compressor.eval import evaluate_qa
from deep_compressor.model import DeepCompressor
from deep_compressor.train import train_stage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("proj_ablation")

# ═══════════════════════════════════════════════════════════════════════
# 5 projection configurations to compare
# ═══════════════════════════════════════════════════════════════════════
ABLATION_CONFIGS = [
    {
        "name": "A_mlp768",
        "desc": "MLP(768) baseline（当前默认，有瓶颈）",
        "down_proj_mode": "mlp",
        "up_proj_mode": "mlp",
        "down_hidden": 768,
        "up_hidden": 768,
    },
    {
        "name": "B_mlp1024",
        "desc": "MLP(1024) 去瓶颈保留非线性",
        "down_proj_mode": "mlp",
        "up_proj_mode": "mlp",
        "down_hidden": 1024,
        "up_hidden": 1024,
    },
    {
        "name": "C_linear",
        "desc": "Linear + LayerNorm",
        "down_proj_mode": "linear",
        "up_proj_mode": "linear",
        "down_hidden": 768,
        "up_hidden": 768,
    },
    {
        "name": "D_identity",
        "desc": "Identity 全去掉（0投影参数）",
        "down_proj_mode": "identity",
        "up_proj_mode": "identity",
        "down_hidden": 768,
        "up_hidden": 768,
    },
    {
        "name": "E_hybrid",
        "desc": "Identity↓ + MLP(1024)↑",
        "down_proj_mode": "identity",
        "up_proj_mode": "mlp",
        "down_hidden": 768,
        "up_hidden": 1024,
    },
]


def build_config(acfg, args, stage):
    """Build DeepCompressorConfig for a given projection config and training stage."""
    max_steps = args.ntp_steps if stage == 1 else args.qa_steps
    warmup = max(10, max_steps // 20)

    return DeepCompressorConfig(
        qwen=QwenConfig(
            model_name_or_path=args.model_path,
            hidden_size=1024,
            num_hidden_layers=28,
            vocab_size=151936,
            max_doc_tokens=512,
            max_question_tokens=64,
            max_answer_tokens=64,
        ),
        finbert=FinBERTConfig(enabled=False),
        perceiver=PerceiverConfig(
            perceiver_dim=1024,
            num_queries=64,
            num_heads=16,
            head_dim=64,
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
            down_hidden=acfg["down_hidden"],
            up_hidden=acfg["up_hidden"],
            dropout=0.1,
        ),
        loss=LossConfig(
            kl_temperature=2.0,
            hidden_distill_ramp_steps=200,
            hidden_distill_layers=[7, 14, 21, 27],
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
            eval_every=max(1, max_steps // 5),
            save_every=999999,  # no checkpoints during ablation
            output_dir=f"outputs/proj_ablation/{acfg['name']}",
            ntp_segment_len=64,
            gradient_checkpointing=True,
            mixed_precision=args.mixed_precision,
        ),
        ablation=AblationConfig(
            down_proj_mode=acfg["down_proj_mode"],
            up_proj_mode=acfg["up_proj_mode"],
        ),
    )


def run_one_config(acfg, tokenizer, args):
    """Train NTP → QA → Eval for one projection configuration."""
    name = acfg["name"]
    logger.info(f"\n{'='*70}")
    logger.info(f"  [{name}] {acfg['desc']}")
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
    logger.info(f"  [{name}] Stage 1: NTP ({args.ntp_steps} steps)")
    ntp_config = build_config(acfg, args, stage=1)
    ntp_model = DeepCompressor(ntp_config)
    if ntp_config.training.gradient_checkpointing:
        ntp_model.qwen.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in ntp_model.parameters() if p.requires_grad)
    proj_params = sum(
        p.numel()
        for n in ("down_proj", "up_mlp")
        for p in getattr(ntp_model, n).parameters()
    )
    logger.info(f"  [{name}] Trainable: {trainable:,}  (projection: {proj_params:,})")

    ntp_ds = NTPDataset(args.ntp_data, tokenizer,
                        max_doc_tokens=512, segment_len=64)
    n_total = len(ntp_ds)
    n_val = min(500, n_total // 10) if n_total > 10 else 0
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
    logger.info(f"  [{name}] Stage 2: QA ({args.qa_steps} steps)")
    qa_config = build_config(acfg, args, stage=2)
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
        qa_eval_loader = DataLoader(qa_eval_ds, batch_size=args.batch_size,
                                    shuffle=False, collate_fn=collator,
                                    num_workers=num_workers, pin_memory=pin_memory)
        logger.info(f"  [{name}] QA train: {len(qa_ds):,}  dev: {len(qa_eval_ds):,}")

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

    logger.info(f"  [{name}] DONE  EM={em:.2%}  F1={f1:.4f}  "
                f"NTP_ppl={ntp_ppl:.2f}  time={elapsed:.0f}s")

    # Cleanup Stage 2
    del qa_model, qa_accelerator, qa_loader, qa_eval_loader
    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()

    return {
        "name": name,
        "desc": acfg["desc"],
        "trainable_params": trainable,
        "proj_params": proj_params,
        "ntp_loss": ntp_loss,
        "ntp_ppl": ntp_ppl,
        "ntp_time": t_ntp,
        "em": em,
        "f1": f1,
        "elapsed": elapsed,
    }


def print_comparison(all_results):
    """Print final comparison table sorted by F1."""
    print("\n" + "=" * 95)
    print("  PROJECTION ABLATION RESULTS (Real Data)")
    print("=" * 95)

    ranked = sorted(all_results, key=lambda r: r["f1"], reverse=True)

    print(f"\n  {'#':<3} {'Config':<20} {'Proj Params':<12} "
          f"{'NTP PPL':<9} {'EM':<8} {'F1':<8} {'Time':<7}")
    print("  " + "-" * 70)
    for i, r in enumerate(ranked):
        proj_str = f"{r['proj_params']:,}" if r['proj_params'] > 0 else "0"
        mark = " *" if i == 0 else ""
        print(f"  {i+1:<3} {r['name']:<20} {proj_str:<12} "
              f"{r['ntp_ppl']:<9.2f} {r['em']:.2%}{'':>2} {r['f1']:.4f}  "
              f"{r['elapsed']:.0f}s{mark}")

    best = ranked[0]
    worst = ranked[-1]
    baseline = next((r for r in all_results if r["name"] == "A_mlp768"), ranked[-1])

    print(f"\n  BEST:     {best['name']} — {best['desc']}")
    print(f"            EM={best['em']:.2%}  F1={best['f1']:.4f}")
    if best["name"] != baseline["name"]:
        delta_f1 = best["f1"] - baseline["f1"]
        delta_em = best["em"] - baseline["em"]
        print(f"            vs baseline: EM {delta_em:+.2%}  F1 {delta_f1:+.4f}")

    print(f"\n  WORST:    {worst['name']} — {worst['desc']}")
    print(f"            EM={worst['em']:.2%}  F1={worst['f1']:.4f}")

    # JSON dump for further analysis
    json_path = "outputs/proj_ablation/results.json"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {json_path}")
    print("=" * 95)


def main():
    parser = argparse.ArgumentParser(
        description="Projection layer ablation on real data")

    # Data paths
    parser.add_argument("--ntp_data", type=str,
                        default="data/ablation/ntp_ablation.jsonl")
    parser.add_argument("--qa_train", type=str,
                        default="data/ablation/qa_ablation_train.json")
    parser.add_argument("--qa_dev", type=str, default=None,
                        help="QA dev set for evaluation (default: auto-detect)")
    parser.add_argument("--model_path", type=str, default="models/Qwen3-0.6B")

    # Training steps
    parser.add_argument("--ntp_steps", type=int, default=2000,
                        help="NTP pretraining steps per config")
    parser.add_argument("--qa_steps", type=int, default=2000,
                        help="QA fine-tuning steps per config")
    parser.add_argument("--lr", type=float, default=5e-4)

    # H100 optimization
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Micro-batch size per GPU (default: 32 for H100)")
    parser.add_argument("--grad_accum", type=int, default=2,
                        help="Gradient accumulation steps (eff_batch = batch_size * grad_accum)")
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision mode (bf16 recommended for H100)")

    # Config selection
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated config names (default: all 5)")

    args = parser.parse_args()

    # Auto-detect QA dev path
    if args.qa_dev is None:
        candidate = args.qa_train.replace("_train.", "_dev.")
        if os.path.exists(candidate):
            args.qa_dev = candidate
            logger.info(f"Auto-detected QA dev: {args.qa_dev}")
        else:
            logger.warning("No QA dev set found — eval will be skipped")

    # Validate data paths
    for name, path in [("ntp_data", args.ntp_data), ("qa_train", args.qa_train)]:
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            logger.error("Run 'python scripts/prepare_data.py && "
                         "python scripts/prepare_data.py --make-ablation' first")
            sys.exit(1)

    eff_batch = args.batch_size * args.grad_accum
    logger.info(f"batch_size={args.batch_size} × grad_accum={args.grad_accum} "
                f"= eff_batch {eff_batch}")
    logger.info(f"mixed_precision={args.mixed_precision}")
    logger.info(f"NTP steps={args.ntp_steps}, QA steps={args.qa_steps}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.0f} GB)")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Filter configs
    configs = ABLATION_CONFIGS
    if args.configs:
        names = set(args.configs.split(","))
        configs = [c for c in configs if c["name"] in names]
    logger.info(f"Running {len(configs)} configs: {[c['name'] for c in configs]}")

    all_results = []
    for i, acfg in enumerate(configs):
        logger.info(f"\n>>> Config {i+1}/{len(configs)}: {acfg['name']}")
        result = run_one_config(acfg, tokenizer, args)
        all_results.append(result)

        # Print running comparison after each config
        if len(all_results) > 1:
            print_comparison(all_results)

    # Final comparison
    print_comparison(all_results)


if __name__ == "__main__":
    main()
