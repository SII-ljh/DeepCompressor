#!/usr/bin/env python3
"""Phase 2 mid-scale ablation study: 15 experiments at 5x Phase 1 scale.

Runs 10K NTP + 5K QA steps per experiment (vs 2K+1K in Phase 1) to verify
trends and get real QA metrics now that the generate_answer() bug is fixed.

15 experiments in 4 groups:
  A: Core module ablations (baseline, proj_identity, no_stage_b, no_stage_a,
     queries_16, no_query_cond, no_stage_c)
  B: New architecture variants (proj_linear, queries_32, deep, no_stage_ac)
  C: Distillation (no_distillation, full_distillation)
  D: Training schedule (lr_1e-3, long_ntp)

Usage:
  # List all experiments
  python scripts/ablation_phase2.py --list

  # Run all experiments
  python scripts/ablation_phase2.py --data_path data/ablation/ntp_ablation.jsonl \
      --eval_data_path data/ablation/qa_ablation_dev.json \
      --qa_data_path data/ablation/qa_ablation_train.json

  # Run specific groups
  python scripts/ablation_phase2.py --groups A,B --data_path data/ntp_train.jsonl

  # Run specific configs
  python scripts/ablation_phase2.py --configs baseline,proj_identity \
      --data_path data/ntp_train.jsonl

  # Fast mode for script validation (2K NTP + 1K QA)
  python scripts/ablation_phase2.py --fast --configs baseline \
      --data_path data/ntp_tiny.jsonl
"""

import argparse
import copy
import gc
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

from deep_compressor.config import DeepCompressorConfig
from deep_compressor.data import NTPDataset, PaddingCollator, QADataset
from deep_compressor.eval import evaluate_ntp, evaluate_qa
from deep_compressor.model import DeepCompressor
from deep_compressor.train import get_scheduler, save_checkpoint, train_stage

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Experiment Registry ──────────────────────────────────────────────

@dataclass
class ExperimentSpec:
    name: str
    group: str
    description: str
    ablation_overrides: dict  # applied to config.ablation
    training_overrides: dict  # applied to config.training
    loss_overrides: dict      # applied to config.loss
    ntp_steps: Optional[int] = None  # override NTP steps (for long_ntp)


EXPERIMENTS: List[ExperimentSpec] = [
    # ── Group A: Phase 1 validation ──
    ExperimentSpec("baseline", "A", "Full pipeline control",
                   ablation_overrides={}, training_overrides={}, loss_overrides={}),
    ExperimentSpec("proj_identity", "A", "Identity projection (Phase 1 best)",
                   ablation_overrides={"down_proj_mode": "identity", "up_proj_mode": "identity"},
                   training_overrides={}, loss_overrides={}),
    ExperimentSpec("no_stage_b", "A", "Disable Stage B (Phase 1 worst)",
                   ablation_overrides={"enable_stage_b": False},
                   training_overrides={}, loss_overrides={}),
    ExperimentSpec("no_stage_a", "A", "Disable Stage A (Phase 1 better than baseline)",
                   ablation_overrides={"enable_stage_a": False},
                   training_overrides={}, loss_overrides={}),
    ExperimentSpec("queries_16", "A", "16 queries (aggressive compression)",
                   ablation_overrides={"override_num_queries": 16},
                   training_overrides={}, loss_overrides={}),
    ExperimentSpec("no_query_cond", "A", "QueryInit without question conditioning",
                   ablation_overrides={"query_condition_on_question": False},
                   training_overrides={}, loss_overrides={}),
    ExperimentSpec("no_stage_c", "A", "Disable Stage C (deep reasoning)",
                   ablation_overrides={"enable_stage_c": False},
                   training_overrides={}, loss_overrides={}),

    # ── Group B: New architecture variants ──
    ExperimentSpec("proj_linear", "B", "Linear projection (middle ground)",
                   ablation_overrides={"down_proj_mode": "linear", "up_proj_mode": "linear"},
                   training_overrides={}, loss_overrides={}),
    ExperimentSpec("queries_32", "B", "32 queries (between 16 and 64)",
                   ablation_overrides={"override_num_queries": 32},
                   training_overrides={}, loss_overrides={}),
    ExperimentSpec("deep", "B", "Deep perceiver (3-6 layers per stage)",
                   ablation_overrides={
                       "override_stage_a_cross_layers": 3,
                       "override_stage_a_self_layers": 3,
                       "override_stage_b_layers": 4,
                       "override_stage_c_cross_layers": 3,
                       "override_stage_c_self_layers": 6,
                   }, training_overrides={}, loss_overrides={}),
    ExperimentSpec("no_stage_ac", "B", "Stage B only (pure self-attention)",
                   ablation_overrides={"enable_stage_a": False, "enable_stage_c": False},
                   training_overrides={}, loss_overrides={}),

    # ── Group C: Distillation ──
    ExperimentSpec("no_distillation", "C", "QA without teacher signal",
                   ablation_overrides={
                       "enable_kl_distillation": False,
                       "enable_hidden_mse_distillation": False,
                   }, training_overrides={}, loss_overrides={
                       "kl_weight": 0.0, "hidden_mse_weight": 0.0,
                   }),
    ExperimentSpec("full_distillation", "C", "QA with full teacher distillation",
                   ablation_overrides={}, training_overrides={}, loss_overrides={
                       "kl_weight": 1.0, "hidden_mse_weight": 1.0,
                   }),

    # ── Group D: Training schedule ──
    ExperimentSpec("lr_1e-3", "D", "2x learning rate (5e-4 -> 1e-3)",
                   ablation_overrides={}, training_overrides={"learning_rate": 1e-3},
                   loss_overrides={}),
    ExperimentSpec("long_ntp", "D", "15K NTP steps (1.5x budget)",
                   ablation_overrides={}, training_overrides={},
                   loss_overrides={}, ntp_steps=15000),
]

EXPERIMENT_MAP = {e.name: e for e in EXPERIMENTS}
GROUPS = sorted(set(e.group for e in EXPERIMENTS))


# ── Config builders ──────────────────────────────────────────────────

def _base_config(args) -> DeepCompressorConfig:
    """Build baseline config from YAML or defaults."""
    if args.config and os.path.exists(args.config):
        cfg = DeepCompressorConfig.from_yaml(args.config)
    else:
        cfg = DeepCompressorConfig.from_yaml("configs/ablation_base.yaml")
    return cfg


def _apply_experiment(cfg: DeepCompressorConfig, spec: ExperimentSpec,
                      ntp_steps: int, qa_steps: int, args) -> DeepCompressorConfig:
    """Apply experiment overrides to a fresh config copy."""
    c = copy.deepcopy(cfg)

    # Ablation overrides
    for k, v in spec.ablation_overrides.items():
        if not hasattr(c.ablation, k):
            raise ValueError(f"Unknown ablation field: {k}")
        setattr(c.ablation, k, v)

    # Training overrides
    for k, v in spec.training_overrides.items():
        if not hasattr(c.training, k):
            raise ValueError(f"Unknown training field: {k}")
        setattr(c.training, k, v)

    # Loss overrides
    for k, v in spec.loss_overrides.items():
        if not hasattr(c.loss, k):
            raise ValueError(f"Unknown loss field: {k}")
        setattr(c.loss, k, v)

    # NTP steps (per-experiment override or global)
    actual_ntp = spec.ntp_steps if spec.ntp_steps is not None else ntp_steps
    c.training.max_steps = actual_ntp  # will be reset for QA stage

    # Common training params
    c.training.batch_size = args.batch_size
    c.training.seed = 42
    c.training.gradient_checkpointing = True
    c.training.output_dir = os.path.join(args.output_dir, f"phase2_{spec.name}")

    # Re-validate
    c.__post_init__()
    return c


# ── Results I/O ──────────────────────────────────────────────────────

def _results_path(output_dir: str) -> str:
    return os.path.join(output_dir, "phase2_results.json")


def _load_results(output_dir: str) -> dict:
    path = _results_path(output_dir)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_results(output_dir: str, results: dict):
    os.makedirs(output_dir, exist_ok=True)
    path = _results_path(output_dir)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


# ── Training with learning curve tracking ────────────────────────────

def _run_single_experiment(
    spec: ExperimentSpec,
    base_cfg: DeepCompressorConfig,
    ntp_steps: int,
    qa_steps: int,
    ntp_eval_every: int,
    qa_eval_every: int,
    args,
) -> dict:
    """Run one experiment: NTP pretraining + QA fine-tuning.

    Returns a result dict with metrics and learning curve data.
    """
    result = {
        "name": spec.name,
        "group": spec.group,
        "description": spec.description,
        "status": "running",
        "ntp_metrics": {},
        "qa_metrics": {},
        "learning_curve": [],
        "elapsed_ntp": 0,
        "elapsed_qa": 0,
    }

    actual_ntp = spec.ntp_steps if spec.ntp_steps is not None else ntp_steps

    # ── Stage 1: NTP ──
    logger.info(f"  Stage 1: NTP ({actual_ntp} steps)")
    ntp_cfg = _apply_experiment(base_cfg, spec, ntp_steps, qa_steps, args)
    ntp_cfg.training.stage = 1
    ntp_cfg.training.max_steps = actual_ntp
    ntp_cfg.training.warmup_steps = args.ntp_warmup
    ntp_cfg.training.eval_every = ntp_eval_every
    ntp_cfg.training.save_every = actual_ntp  # save only at end
    ntp_cfg.training.log_every = max(1, actual_ntp // 40)
    ntp_cfg.training.mixed_precision = args.mixed_precision
    ntp_cfg.training.ntp_segment_len = 64

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=ntp_cfg.training.gradient_accumulation_steps,
        mixed_precision=ntp_cfg.training.mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )

    tokenizer = AutoTokenizer.from_pretrained(ntp_cfg.qwen.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = DeepCompressor(ntp_cfg)
    if ntp_cfg.training.gradient_checkpointing:
        model.qwen.gradient_checkpointing_enable()

    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
    num_workers = 0 if accelerator.device.type == "mps" else 2
    pin_memory = accelerator.device.type == "cuda"

    # NTP data
    data_path = args.data_path
    ntp_ds = NTPDataset(data_path, tokenizer,
                        max_doc_tokens=ntp_cfg.qwen.max_doc_tokens,
                        segment_len=ntp_cfg.training.ntp_segment_len)
    n_total = len(ntp_ds)
    n_val = min(5000, n_total // 10) if n_total > 10 else 0
    if n_val > 0:
        train_subset = Subset(ntp_ds, list(range(n_total - n_val)))
        val_subset = Subset(ntp_ds, list(range(n_total - n_val, n_total)))
    else:
        train_subset = ntp_ds
        val_subset = None

    ntp_loader = DataLoader(train_subset, batch_size=ntp_cfg.training.batch_size,
                            shuffle=True, collate_fn=collator,
                            num_workers=num_workers, pin_memory=pin_memory)
    ntp_eval_loader = None
    if val_subset is not None:
        if args.max_eval_samples > 0:
            val_subset = Subset(val_subset,
                                list(range(min(args.max_eval_samples, len(val_subset)))))
        ntp_eval_loader = DataLoader(val_subset, batch_size=ntp_cfg.training.batch_size,
                                     shuffle=False, collate_fn=collator,
                                     num_workers=num_workers, pin_memory=pin_memory)

    os.makedirs(ntp_cfg.training.output_dir, exist_ok=True)

    t0 = time.time()
    model, ntp_metrics = train_stage(
        ntp_cfg, model, ntp_loader, accelerator, mode="ntp",
        eval_loader=ntp_eval_loader, tokenizer=tokenizer,
    )
    result["elapsed_ntp"] = time.time() - t0
    result["ntp_metrics"] = ntp_metrics or {}
    logger.info(f"  NTP done in {result['elapsed_ntp']:.0f}s  metrics={ntp_metrics}")

    # Save NTP checkpoint
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(model, accelerator, ntp_cfg.training.output_dir, "ntp_final")

    ntp_ckpt_path = os.path.join(ntp_cfg.training.output_dir, "checkpoint-ntp_final")

    # Free NTP accelerator state
    accelerator.free_memory()
    del accelerator, ntp_loader, ntp_eval_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Stage 2: QA ──
    if not args.qa_data_path:
        logger.info("  Skipping QA stage (no --qa_data_path)")
        result["status"] = "ntp_only"
        return result

    logger.info(f"  Stage 2: QA ({qa_steps} steps)")
    qa_cfg = _apply_experiment(base_cfg, spec, ntp_steps, qa_steps, args)
    qa_cfg.training.stage = 2
    qa_cfg.training.max_steps = qa_steps
    qa_cfg.training.warmup_steps = args.qa_warmup
    qa_cfg.training.eval_every = qa_eval_every
    qa_cfg.training.save_every = qa_steps  # save only at end
    qa_cfg.training.log_every = max(1, qa_steps // 40)
    qa_cfg.training.mixed_precision = args.mixed_precision

    ddp_kwargs2 = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator2 = Accelerator(
        gradient_accumulation_steps=qa_cfg.training.gradient_accumulation_steps,
        mixed_precision=qa_cfg.training.mixed_precision,
        kwargs_handlers=[ddp_kwargs2],
    )

    # Rebuild model and load NTP weights
    model2 = DeepCompressor(qa_cfg)
    if qa_cfg.training.gradient_checkpointing:
        model2.qwen.gradient_checkpointing_enable()

    ckpt_file = os.path.join(ntp_ckpt_path, "trainable_weights.pt")
    if os.path.exists(ckpt_file):
        weights = torch.load(ckpt_file, map_location="cpu", weights_only=True)
        model2.load_state_dict(weights, strict=False)
        logger.info(f"  Loaded NTP checkpoint: {ckpt_file}")

    # Build teacher model for distillation experiments
    teacher_model = None
    needs_teacher = (qa_cfg.loss.kl_weight > 0 or qa_cfg.loss.hidden_mse_weight > 0)
    if needs_teacher:
        logger.info("  Loading teacher model for distillation...")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            qa_cfg.qwen.model_name_or_path, torch_dtype=torch.float32)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        teacher_model = accelerator2.prepare(teacher_model)

    # QA data
    qa_ds = QADataset(args.qa_data_path, tokenizer,
                      max_doc_tokens=qa_cfg.qwen.max_doc_tokens,
                      max_question_tokens=qa_cfg.qwen.max_question_tokens,
                      max_answer_tokens=qa_cfg.qwen.max_answer_tokens)
    qa_loader = DataLoader(qa_ds, batch_size=qa_cfg.training.batch_size,
                           shuffle=True, collate_fn=collator,
                           num_workers=num_workers, pin_memory=pin_memory)

    qa_eval_loader = None
    if args.eval_data_path:
        eval_ds = QADataset(args.eval_data_path, tokenizer,
                            max_doc_tokens=qa_cfg.qwen.max_doc_tokens,
                            max_question_tokens=qa_cfg.qwen.max_question_tokens,
                            max_answer_tokens=qa_cfg.qwen.max_answer_tokens)
        if args.max_eval_samples > 0:
            eval_ds = Subset(eval_ds,
                             list(range(min(args.max_eval_samples, len(eval_ds)))))
        qa_eval_loader = DataLoader(eval_ds, batch_size=qa_cfg.training.batch_size,
                                    shuffle=False, collate_fn=collator,
                                    num_workers=num_workers, pin_memory=pin_memory)

    t0 = time.time()
    model2, qa_metrics = train_stage(
        qa_cfg, model2, qa_loader, accelerator2, mode="qa",
        eval_loader=qa_eval_loader, tokenizer=tokenizer,
        teacher_model=teacher_model,
    )
    result["elapsed_qa"] = time.time() - t0
    result["qa_metrics"] = qa_metrics or {}
    logger.info(f"  QA done in {result['elapsed_qa']:.0f}s  metrics={qa_metrics}")

    # Save QA checkpoint
    accelerator2.wait_for_everyone()
    if accelerator2.is_main_process:
        save_checkpoint(model2, accelerator2, qa_cfg.training.output_dir, "qa_final")

    # Cleanup
    accelerator2.free_memory()
    del accelerator2, model2, teacher_model, qa_loader, qa_eval_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result["status"] = "OK"
    return result


# ── List experiments ─────────────────────────────────────────────────

def _list_experiments():
    """Print all experiments grouped."""
    print(f"\nPhase 2 Ablation Experiments ({len(EXPERIMENTS)} total):")
    print(f"  {'#':<3} {'Group':<6} {'Name':<20} {'Description'}")
    print(f"  {'─' * 70}")
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"  {i:<3} {exp.group:<6} {exp.name:<20} {exp.description}")

    print(f"\nGroups: {', '.join(GROUPS)}")
    print("  A: Core module ablations (7 experiments)")
    print("  B: New architecture variants (4 experiments)")
    print("  C: Distillation (2 experiments)")
    print("  D: Training schedule (2 experiments)")
    print()


# ── Comparison table ─────────────────────────────────────────────────

def _print_results(results: dict):
    """Print formatted comparison table."""
    print("\n" + "=" * 90)
    print("  PHASE 2 ABLATION RESULTS")
    print("=" * 90)

    print(f"\n  {'#':<3} {'Group':<6} {'Name':<20} "
          f"{'NTP PPL':>10} {'QA EM':>8} {'QA F1':>8} "
          f"{'Time':>8} {'Status':<10}")
    print(f"  {'─' * 85}")

    baseline_ppl = None
    baseline_em = None
    baseline_f1 = None

    for i, exp in enumerate(EXPERIMENTS, 1):
        r = results.get(exp.name)
        if r is None:
            print(f"  {i:<3} {exp.group:<6} {exp.name:<20} {'—':>10} {'—':>8} "
                  f"{'—':>8} {'—':>8} {'pending':<10}")
            continue

        ntp = r.get("ntp_metrics", {})
        qa = r.get("qa_metrics", {})
        ppl = ntp.get("perplexity")
        em = qa.get("exact_match")
        f1 = qa.get("f1")
        elapsed = r.get("elapsed_ntp", 0) + r.get("elapsed_qa", 0)
        status = r.get("status", "?")

        ppl_str = f"{ppl:.1f}" if ppl is not None else "—"
        em_str = f"{em:.2%}" if em is not None else "—"
        f1_str = f"{f1:.4f}" if f1 is not None else "—"

        if exp.name == "baseline":
            baseline_ppl = ppl
            baseline_em = em
            baseline_f1 = f1

        print(f"  {i:<3} {exp.group:<6} {exp.name:<20} "
              f"{ppl_str:>10} {em_str:>8} {f1_str:>8} "
              f"{elapsed:>7.0f}s {status:<10}")

    # Delta table
    if baseline_ppl is not None or baseline_f1 is not None:
        print(f"\n  Delta from baseline:")
        print(f"  {'Name':<20} {'dPPL':>10} {'dEM':>10} {'dF1':>10}")
        print(f"  {'─' * 55}")
        for exp in EXPERIMENTS:
            if exp.name == "baseline":
                continue
            r = results.get(exp.name)
            if r is None:
                continue
            ntp = r.get("ntp_metrics", {})
            qa = r.get("qa_metrics", {})

            dppl = ntp.get("perplexity", 0) - baseline_ppl if baseline_ppl and ntp.get("perplexity") else None
            dem = qa.get("exact_match", 0) - baseline_em if baseline_em is not None and qa.get("exact_match") is not None else None
            df1 = qa.get("f1", 0) - baseline_f1 if baseline_f1 is not None and qa.get("f1") is not None else None

            dppl_s = f"{dppl:>+.1f}" if dppl is not None else "—"
            dem_s = f"{dem:>+.2%}" if dem is not None else "—"
            df1_s = f"{df1:>+.4f}" if df1 is not None else "—"
            print(f"  {exp.name:<20} {dppl_s:>10} {dem_s:>10} {df1_s:>10}")

    print()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 mid-scale ablation study (15 experiments)")
    parser.add_argument("--list", action="store_true",
                        help="List all experiments and exit")
    parser.add_argument("--config", type=str, default="configs/ablation_base.yaml",
                        help="Base config YAML")
    parser.add_argument("--data_path", type=str, default=None,
                        help="NTP training data path")
    parser.add_argument("--qa_data_path", type=str, default=None,
                        help="QA training data path")
    parser.add_argument("--eval_data_path", type=str, default=None,
                        help="QA eval data path")

    # Experiment selection
    parser.add_argument("--groups", type=str, default=None,
                        help="Comma-separated groups to run (e.g. A,B)")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated experiment names (e.g. baseline,proj_identity)")

    # Scale controls
    parser.add_argument("--ntp_steps", type=int, default=10000,
                        help="NTP pretraining steps (default: 10000)")
    parser.add_argument("--qa_steps", type=int, default=5000,
                        help="QA fine-tuning steps (default: 5000)")
    parser.add_argument("--ntp_warmup", type=int, default=500,
                        help="NTP warmup steps")
    parser.add_argument("--qa_warmup", type=int, default=250,
                        help="QA warmup steps")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--ntp_eval_every", type=int, default=2500,
                        help="NTP eval frequency")
    parser.add_argument("--qa_eval_every", type=int, default=1250,
                        help="QA eval frequency")
    parser.add_argument("--max_eval_samples", type=int, default=1000,
                        help="Max eval samples")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/ablation_phase2",
                        help="Base output directory")
    parser.add_argument("--mixed_precision", type=str, default="no",
                        help="Mixed precision mode: no, fp16, bf16")

    # Fast mode
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: 2K NTP + 1K QA for validation")

    args = parser.parse_args()

    if args.list:
        _list_experiments()
        return

    if args.data_path is None:
        # Try ablation subset, fall back to full data
        ablation_ntp = "data/ablation/ntp_ablation.jsonl"
        full_ntp = "data/ntp_train.jsonl"
        if os.path.exists(ablation_ntp):
            args.data_path = ablation_ntp
            logger.info(f"Using ablation data subset: {ablation_ntp}")
        elif os.path.exists(full_ntp):
            args.data_path = full_ntp
            logger.info(f"Using full training data: {full_ntp}")
        else:
            logger.error("No data found. Run: python scripts/prepare_data.py --make-ablation")
            sys.exit(1)

    # Auto-discover QA data if not specified
    if args.qa_data_path is None:
        for candidate in ["data/ablation/qa_ablation_train.json", "data/qa_train.json"]:
            if os.path.exists(candidate):
                args.qa_data_path = candidate
                logger.info(f"Using QA training data: {candidate}")
                break
    if args.eval_data_path is None:
        for candidate in ["data/ablation/qa_ablation_dev.json", "data/qa_dev.json"]:
            if os.path.exists(candidate):
                args.eval_data_path = candidate
                logger.info(f"Using QA eval data: {candidate}")
                break

    # Fast mode overrides
    if args.fast:
        args.ntp_steps = 2000
        args.qa_steps = 1000
        args.ntp_warmup = 100
        args.qa_warmup = 50
        args.ntp_eval_every = 500
        args.qa_eval_every = 250
        args.max_eval_samples = 100
        args.batch_size = 4
        logger.info("Fast mode: 2K NTP + 1K QA, batch=4, eval_samples=100")

    # Determine which experiments to run
    selected = list(EXPERIMENTS)
    if args.groups:
        groups = set(g.strip().upper() for g in args.groups.split(","))
        selected = [e for e in selected if e.group in groups]
    if args.configs:
        names = set(n.strip() for n in args.configs.split(","))
        selected = [e for e in selected if e.name in names]

    if not selected:
        logger.error("No experiments selected. Use --list to see available experiments.")
        sys.exit(1)

    logger.info(f"Running {len(selected)} experiment(s): "
                f"{[e.name for e in selected]}")
    logger.info(f"NTP: {args.ntp_steps} steps | QA: {args.qa_steps} steps | "
                f"batch: {args.batch_size}")

    # Load existing results for resume
    all_results = _load_results(args.output_dir)

    base_cfg = _base_config(args)
    t_total = time.time()

    for i, spec in enumerate(selected):
        # Resume: skip completed experiments
        if spec.name in all_results and all_results[spec.name].get("status") == "OK":
            logger.info(f"\n[{i+1}/{len(selected)}] {spec.name}: SKIPPED (already completed)")
            continue

        logger.info(f"\n{'=' * 60}")
        logger.info(f"  [{i+1}/{len(selected)}] {spec.group}/{spec.name}: {spec.description}")
        logger.info(f"{'=' * 60}")

        try:
            result = _run_single_experiment(
                spec, base_cfg,
                ntp_steps=args.ntp_steps,
                qa_steps=args.qa_steps,
                ntp_eval_every=args.ntp_eval_every,
                qa_eval_every=args.qa_eval_every,
                args=args,
            )
            all_results[spec.name] = result
            _save_results(args.output_dir, all_results)
            logger.info(f"  {spec.name}: {result['status']}")

        except Exception as e:
            logger.error(f"  {spec.name}: FAILED — {e}", exc_info=True)
            all_results[spec.name] = {
                "name": spec.name,
                "group": spec.group,
                "status": f"FAILED: {e}",
                "ntp_metrics": {},
                "qa_metrics": {},
                "elapsed_ntp": 0,
                "elapsed_qa": 0,
            }
            _save_results(args.output_dir, all_results)

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    elapsed_total = time.time() - t_total
    logger.info(f"\nAll experiments done in {elapsed_total:.0f}s "
                f"({elapsed_total/3600:.1f}h)")

    _print_results(all_results)


if __name__ == "__main__":
    main()
