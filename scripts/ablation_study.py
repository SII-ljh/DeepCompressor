#!/usr/bin/env python3
"""Phased ablation study for Deep Compressor (Qwen3-0.6B).

Phase 1: Independent module ablations (9 configs, ~3h on H100)
  Each config changes ONE component from the baseline, isolating its effect.
  Results guide Phase 2 design.

Phase 1 Configs:
  baseline        Full pipeline (control)                     ~173M trainable
  no_stage_a      Remove Stage A (global cross-attn)          ~114M (-34%)
  no_stage_b      Remove Stage B (self-attn refinement)       ~148M (-15%)
  no_stage_c      Remove Stage C (deep reasoning re-read)      ~89M (-49%)
  no_query_cond   Remove question conditioning in QueryInit   ~172M (-0.6%)
  proj_identity   Identity projection (remove DownProj/UpMLP) ~169M (-2.4%)
  queries_16      16 queries (4x compression ratio)           ~173M
  queries_128     128 queries (0.5x compression ratio)        ~173M
  shallow         1 layer per stage (vs 2-4)                   ~77M (-56%)

Training budget per config:
  NTP:  2000 steps x batch 16 = 32K samples, ~16M tokens (22% of ablation data)
  QA:   1000 steps x batch 16 = 16K samples  (~1.4 passes through ablation QA)
  Eval: 500 QA dev samples for EM/F1

Usage:
  # Run all Phase 1 ablations
  python scripts/ablation_study.py

  # Run specific configs
  python scripts/ablation_study.py --configs baseline,no_stage_c,shallow

  # Quick smoke test (~30 min total)
  python scripts/ablation_study.py --fast

  # List all configs
  python scripts/ablation_study.py --list

  # Custom data paths
  python scripts/ablation_study.py \
      --ntp_data data/ablation/ntp_ablation.jsonl \
      --qa_train data/ablation/qa_ablation_train.json \
      --qa_dev   data/ablation/qa_ablation_dev.json

Results saved incrementally to outputs/ablation_study/phase1_results.json.
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
from transformers import AutoModelForCausalLM, AutoTokenizer

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
from deep_compressor.model import DeepCompressor
from deep_compressor.train import train_stage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ablation_study")


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: 9 independent ablation configs
# Each changes exactly ONE thing from the baseline.
# ═══════════════════════════════════════════════════════════════════════
PHASE1_CONFIGS = [
    {
        "name": "baseline",
        "desc": "Full pipeline (control)",
        "ablation": {},
        "perceiver": {},
        "projection": {},
    },
    {
        "name": "no_stage_a",
        "desc": "Remove Stage A (global cross-attn from doc)",
        "ablation": {"enable_stage_a": False},
        "perceiver": {},
        "projection": {},
    },
    {
        "name": "no_stage_b",
        "desc": "Remove Stage B (self-attn refinement)",
        "ablation": {"enable_stage_b": False},
        "perceiver": {},
        "projection": {},
    },
    {
        "name": "no_stage_c",
        "desc": "Remove Stage C (deep reasoning re-read)",
        "ablation": {"enable_stage_c": False},
        "perceiver": {},
        "projection": {},
    },
    {
        "name": "no_query_cond",
        "desc": "Remove question conditioning in QueryInit",
        "ablation": {"query_condition_on_question": False},
        "perceiver": {},
        "projection": {},
    },
    {
        "name": "proj_identity",
        "desc": "Identity projection (no DownProj/UpMLP)",
        "ablation": {"down_proj_mode": "identity", "up_proj_mode": "identity"},
        "perceiver": {},
        "projection": {},
    },
    {
        "name": "queries_16",
        "desc": "16 queries (4x higher compression)",
        "ablation": {"override_num_queries": 16},
        "perceiver": {},
        "projection": {},
    },
    {
        "name": "queries_128",
        "desc": "128 queries (2x lower compression)",
        "ablation": {"override_num_queries": 128},
        "perceiver": {},
        "projection": {},
    },
    {
        "name": "shallow",
        "desc": "Shallow perceiver (1 layer per stage)",
        "ablation": {
            "override_stage_a_cross_layers": 1,
            "override_stage_a_self_layers": 1,
            "override_stage_b_layers": 1,
            "override_stage_c_cross_layers": 1,
            "override_stage_c_self_layers": 1,
        },
        "perceiver": {},
        "projection": {},
    },
]


def build_config(acfg, args, stage):
    """Build DeepCompressorConfig with ablation overrides for one experiment."""
    max_steps = args.ntp_steps if stage == 1 else args.qa_steps
    warmup = max(10, max_steps // 10)

    ablation_kwargs = {
        "down_proj_mode": "mlp",
        "up_proj_mode": "mlp",
        "query_condition_on_question": True,
        "enable_stage_a": True,
        "enable_stage_b": True,
        "enable_stage_c": True,
    }
    ablation_kwargs.update(acfg.get("ablation", {}))

    perceiver_kwargs = {
        "num_queries": 64,
        "num_heads": 16,
        "stage_a_cross_layers": 2,
        "stage_a_self_layers": 2,
        "stage_b_layers": 2,
        "stage_c_cross_layers": 2,
        "stage_c_self_layers": 4,
        "ff_mult": 4,
        "anchor_score_scale_init": 1.0,
        "dropout": 0.1,
    }
    perceiver_kwargs.update(acfg.get("perceiver", {}))

    projection_kwargs = {
        "down_hidden": 1024,   # = hidden_size, no bottleneck
        "up_hidden": 1024,
        "dropout": 0.1,
    }
    projection_kwargs.update(acfg.get("projection", {}))

    return DeepCompressorConfig(
        qwen=QwenConfig(
            model_name_or_path=args.model_path,
            max_doc_tokens=512,
            max_question_tokens=64,
            max_answer_tokens=64,
        ),
        finbert=FinBERTConfig(enabled=False),
        perceiver=PerceiverConfig(**perceiver_kwargs),
        projection=ProjectionConfig(**projection_kwargs),
        loss=LossConfig(
            kl_temperature=2.0,
            hidden_distill_ramp_steps=200,
            hidden_distill_layers=[7, 14, 21, 27],
            qa_ce_weight=1.0,
            kl_weight=0.0,       # no distillation for ablation (simplify)
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
            eval_every=max(1, max_steps // 4),
            save_every=999999,
            output_dir=f"outputs/ablation_study/{acfg['name']}",
            ntp_segment_len=64,
            gradient_checkpointing=True,
            mixed_precision=args.mixed_precision,
        ),
        ablation=AblationConfig(**ablation_kwargs),
    )


def run_one_ablation(acfg, qwen_model, tokenizer, args):
    """Train NTP -> QA -> Eval for one ablation config."""
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

    # Clone frozen Qwen to avoid cross-run interference from Accelerate
    ntp_model = DeepCompressor(ntp_config, qwen_model=qwen_model)
    if ntp_config.training.gradient_checkpointing:
        ntp_model.qwen.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in ntp_model.parameters() if p.requires_grad)
    proj_params = sum(
        p.numel()
        for n in ("down_proj", "up_mlp")
        for p in getattr(ntp_model, n).parameters()
    )
    perceiver_params = sum(p.numel() for p in ntp_model.perceiver.parameters())
    query_params = sum(p.numel() for p in ntp_model.query_init.parameters())

    logger.info(f"  [{name}] Trainable: {trainable:,}  "
                f"(perceiver: {perceiver_params:,}, "
                f"projection: {proj_params:,}, "
                f"query_init: {query_params:,})")

    ntp_ds = NTPDataset(args.ntp_data, tokenizer,
                        max_doc_tokens=512, segment_len=64)
    n_total = len(ntp_ds)
    n_val = min(200, n_total // 10) if n_total > 10 else 0
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

    # Extract trainable weights for Stage 2
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
    qa_model = DeepCompressor(qa_config, qwen_model=qwen_model)
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
        "phase": 1,
        "trainable_params": trainable,
        "perceiver_params": perceiver_params,
        "proj_params": proj_params,
        "query_params": query_params,
        "ntp_loss": ntp_loss,
        "ntp_ppl": ntp_ppl,
        "ntp_time": round(t_ntp, 1),
        "em": em,
        "f1": f1,
        "elapsed": round(elapsed, 1),
    }


def _result_key(r):
    return f"p{r.get('phase', 1)}_{r['name']}"


def save_results(all_results, json_path):
    """Incrementally save results, merging with existing file."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    existing = []
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = []
    merged = {_result_key(r): r for r in existing}
    for r in all_results:
        merged[_result_key(r)] = r
    with open(json_path, "w") as f:
        json.dump(list(merged.values()), f, indent=2, ensure_ascii=False)
    return list(merged.values())


def print_comparison(all_results, json_path):
    """Print Phase 1 comparison table sorted by F1."""
    print("\n" + "=" * 110)
    print("  PHASE 1 ABLATION RESULTS — Qwen3-0.6B (D=1024)")
    print("=" * 110)

    baseline = next((r for r in all_results if r["name"] == "baseline"), None)
    ranked = sorted(all_results, key=lambda r: r["f1"], reverse=True)

    print(f"\n  {'#':<3} {'Config':<18} {'Trainable':<12} {'Perceiver':<12} "
          f"{'NTP PPL':<10} {'EM':<8} {'F1':<8} {'dF1':<8} {'Time'}")
    print("  " + "-" * 103)

    for i, r in enumerate(ranked):
        delta_f1 = ""
        if baseline and r["name"] != "baseline":
            d = r["f1"] - baseline["f1"]
            delta_f1 = f"{d:+.4f}"

        mark = " <-- best" if i == 0 else ""
        print(f"  {i+1:<3} {r['name']:<18} {r['trainable_params']:>11,} "
              f"{r['perceiver_params']:>11,} "
              f"{r['ntp_ppl']:<10.2f} {r['em']:.2%}{'':>2} {r['f1']:.4f}  "
              f"{delta_f1:<8} {r['elapsed']:.0f}s{mark}")

    if baseline:
        print(f"\n  Baseline: NTP_ppl={baseline['ntp_ppl']:.2f}  "
              f"EM={baseline['em']:.2%}  F1={baseline['f1']:.4f}")

    # Key findings
    if len(ranked) >= 3 and baseline:
        worst = ranked[-1]
        print(f"\n  Most impactful removal: {worst['name']} "
              f"(F1 {worst['f1'] - baseline['f1']:+.4f})")
        # Find least impactful (closest to baseline, excluding baseline itself)
        non_base = [r for r in ranked if r["name"] != "baseline"]
        if non_base:
            least = min(non_base, key=lambda r: abs(r["f1"] - baseline["f1"]))
            print(f"  Least impactful removal: {least['name']} "
                  f"(F1 {least['f1'] - baseline['f1']:+.4f})")

    merged = save_results(all_results, json_path)
    print(f"\n  Results saved to {json_path} ({len(merged)} entries)")
    print("=" * 110)


def list_configs():
    """Print all Phase 1 ablation configs."""
    print("\nPhase 1: Independent Module Ablations (Qwen3-0.6B)")
    print("-" * 70)
    for cfg in PHASE1_CONFIGS:
        changes = cfg.get("ablation", {})
        change_str = ", ".join(f"{k}={v}" for k, v in changes.items()) if changes else "(default)"
        print(f"  {cfg['name']:<18} {cfg['desc']}")
        print(f"  {'':18} overrides: {change_str}")
    print(f"\nTotal: {len(PHASE1_CONFIGS)} configs")
    print(f"Est. time: ~20 min/config x {len(PHASE1_CONFIGS)} = "
          f"~{len(PHASE1_CONFIGS) * 20 // 60}h {len(PHASE1_CONFIGS) * 20 % 60}min on H100")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Phased ablation study for Deep Compressor (Qwen3-0.6B)")

    # Model
    parser.add_argument("--model_path", type=str, default="models/Qwen3-0.6B",
                        help="Path to Qwen3-0.6B model")

    # Data paths
    parser.add_argument("--ntp_data", type=str,
                        default="data/ablation/ntp_ablation.jsonl")
    parser.add_argument("--qa_train", type=str,
                        default="data/ablation/qa_ablation_train.json")
    parser.add_argument("--qa_dev", type=str, default=None,
                        help="QA dev set (default: auto-detect)")

    # Training budget
    parser.add_argument("--ntp_steps", type=int, default=2000,
                        help="NTP steps per config (default: 2000)")
    parser.add_argument("--qa_steps", type=int, default=1000,
                        help="QA steps per config (default: 1000)")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])

    # Eval
    parser.add_argument("--max_eval_samples", type=int, default=500,
                        help="Max QA dev samples (default: 500)")

    # Config selection
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated config names (default: all)")
    parser.add_argument("--fast", action="store_true",
                        help="Quick smoke test: 500 NTP + 300 QA + 100 eval")

    # Output
    parser.add_argument("--output", type=str,
                        default="outputs/ablation_study/phase1_results.json")

    # Info
    parser.add_argument("--list", action="store_true",
                        help="List configs and exit")

    args = parser.parse_args()

    if args.list:
        list_configs()
        return

    # --fast overrides
    if args.fast:
        args.ntp_steps = 500
        args.qa_steps = 300
        args.max_eval_samples = 100
        logger.info("Fast mode: ntp=500, qa=300, eval=100")

    # Auto-detect QA dev
    if args.qa_dev is None:
        candidate = args.qa_train.replace("_train.", "_dev.")
        if os.path.exists(candidate):
            args.qa_dev = candidate
            logger.info(f"Auto-detected QA dev: {args.qa_dev}")
        else:
            logger.warning("No QA dev set found -- QA eval will be skipped")

    # Validate data paths
    for label, path in [("ntp_data", args.ntp_data), ("qa_train", args.qa_train)]:
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            logger.error("Run: python scripts/prepare_data.py --make-ablation")
            sys.exit(1)

    # Select configs
    configs = PHASE1_CONFIGS
    if args.configs:
        names = set(args.configs.split(","))
        configs = [c for c in configs if c["name"] in names]
        if not configs:
            logger.error(f"No matching configs. Available: "
                         f"{[c['name'] for c in PHASE1_CONFIGS]}")
            sys.exit(1)

    # Print experiment summary
    ntp_samples = args.ntp_steps * args.batch_size
    qa_samples = args.qa_steps * args.batch_size
    logger.info(f"{'='*70}")
    logger.info(f"  Phase 1 Ablation Study — Qwen3-0.6B")
    logger.info(f"  Configs: {[c['name'] for c in configs]}")
    logger.info(f"  NTP: {args.ntp_steps} steps x {args.batch_size} batch = "
                f"{ntp_samples:,} samples (~{ntp_samples * 512 // 1_000_000}M tokens)")
    logger.info(f"  QA:  {args.qa_steps} steps x {args.batch_size} batch = "
                f"{qa_samples:,} samples")
    logger.info(f"  Eval: {args.max_eval_samples} QA dev samples")
    logger.info(f"  LR={args.lr}, mixed_precision={args.mixed_precision}")
    logger.info(f"{'='*70}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.0f} GB)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pre-load frozen Qwen model once, share across all configs
    logger.info(f"Loading frozen Qwen model from {args.model_path} ...")
    qwen_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float32,
    )
    for p in qwen_model.parameters():
        p.requires_grad = False
    logger.info(f"Qwen model loaded: {sum(p.numel() for p in qwen_model.parameters()):,} params (frozen)")

    # Run ablations
    all_results = []
    for i, acfg in enumerate(configs):
        logger.info(f"\n>>> [{i+1}/{len(configs)}] {acfg['name']}: {acfg['desc']}")
        result = run_one_ablation(acfg, qwen_model, tokenizer, args)
        all_results.append(result)

        # Save incrementally after each run
        save_results(all_results, args.output)
        if len(all_results) > 1:
            print_comparison(all_results, args.output)

    # Final comparison
    print_comparison(all_results, args.output)

    # Suggest Phase 2
    print("\n  Phase 2 suggestions (based on Phase 1 results):")
    print("  - For impactful modules: test finer-grained depth/width variations")
    print("  - For negligible modules: test combined removal (e.g. no_stage_b + proj_identity)")
    print("  - If queries_16 ~ baseline: you can compress 4x more aggressively")
    print("  - Add distillation ablation (requires teacher logits): --kl and --mse toggles")
    print()


if __name__ == "__main__":
    main()
