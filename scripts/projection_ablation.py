#!/usr/bin/env python3
"""Quick ablation: compare 5 projection layer configurations.

Runs in a single process, reuses tokenizer across configs.
Each config gets fresh model weights + independent NTP/QA training.

Usage:
  python scripts/projection_ablation.py
  python scripts/projection_ablation.py --ntp_steps 150 --qa_steps 1500
"""

import argparse
import gc
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

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
from deep_compressor.data import NTPDataset, PaddingCollator
from deep_compressor.eval import compute_exact_match, compute_f1
from deep_compressor.model import DeepCompressor
from overfit_test import (
    FINANCIAL_QA,
    QADatasetWithEOS,
    evaluate,
    make_temp_data,
    train_loop,
)

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


def build_config(acfg, ntp_steps):
    """Build DeepCompressorConfig with specific projection settings."""
    return DeepCompressorConfig(
        qwen=QwenConfig(
            model_name_or_path="models/Qwen3-0.6B",
            hidden_size=1024,
            num_hidden_layers=28,
            vocab_size=151936,
            max_doc_tokens=256,
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
            dropout=0.0,
        ),
        projection=ProjectionConfig(
            down_hidden=acfg["down_hidden"],
            up_hidden=acfg["up_hidden"],
            dropout=0.0,
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
            stage=1,
            learning_rate=1e-3,
            batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=ntp_steps,
            warmup_steps=10,
            weight_decay=0.0,
            max_grad_norm=1.0,
            scheduler="cosine",
            seed=42,
            log_every=50,
            eval_every=9999,
            save_every=9999,
            output_dir="outputs/proj_ablation",
            ntp_segment_len=64,
            gradient_checkpointing=True,
            mixed_precision="no",
        ),
        ablation=AblationConfig(
            down_proj_mode=acfg["down_proj_mode"],
            up_proj_mode=acfg["up_proj_mode"],
        ),
    )


def run_one_config(acfg, tokenizer, device, ntp_path, ntp_steps, qa_steps, lr):
    """Train and evaluate one projection configuration. Returns result dict."""
    name = acfg["name"]
    logger.info(f"\n{'='*60}")
    logger.info(f"  Config: {name} — {acfg['desc']}")
    logger.info(f"{'='*60}")

    # Deterministic init
    torch.manual_seed(42)
    if device.type == "mps":
        torch.mps.manual_seed(42)

    config = build_config(acfg, ntp_steps)
    model = DeepCompressor(config)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    proj_params = 0
    for mod_name in ("down_proj", "up_mlp"):
        mod = getattr(model, mod_name, None)
        if mod is not None:
            proj_params += sum(p.numel() for p in mod.parameters())
    logger.info(f"  Trainable: {trainable:,}  (projection: {proj_params:,})")

    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
    t0 = time.time()

    # ── NTP ──
    ntp_ds = NTPDataset(ntp_path, tokenizer, max_doc_tokens=256, segment_len=64)
    ntp_loader = DataLoader(ntp_ds, batch_size=2, shuffle=True,
                            collate_fn=collator, num_workers=0)
    ntp_losses, _ = train_loop(model, ntp_loader, config, device,
                               max_steps=ntp_steps, stage_name="ntp", lr=lr)

    # ── QA ──
    qa_ds = QADatasetWithEOS(FINANCIAL_QA, tokenizer,
                             max_doc_tokens=256, max_question_tokens=64,
                             max_answer_tokens=64)
    qa_loader = DataLoader(qa_ds, batch_size=2, shuffle=True,
                           collate_fn=collator, num_workers=0)
    qa_losses, _ = train_loop(model, qa_loader, config, device,
                              max_steps=qa_steps, stage_name="qa", lr=lr)

    # ── Evaluate ──
    results = evaluate(model, FINANCIAL_QA, tokenizer, device)
    elapsed = time.time() - t0

    total_em = sum(r["em"] for r in results) / len(results)
    total_f1 = sum(r["f1"] for r in results) / len(results)
    ntp_final = sum(ntp_losses[-10:]) / max(len(ntp_losses[-10:]), 1)
    qa_final = sum(qa_losses[-10:]) / max(len(qa_losses[-10:]), 1)

    # Per-type breakdown
    type_a = results[:3]   # simple numeric
    type_b = results[3:8]  # sentence-level
    type_c = results[8:]   # reasoning

    em_a = sum(r["em"] for r in type_a) / len(type_a)
    em_b = sum(r["em"] for r in type_b) / len(type_b)
    em_c = sum(r["em"] for r in type_c) / len(type_c)
    f1_a = sum(r["f1"] for r in type_a) / len(type_a)
    f1_b = sum(r["f1"] for r in type_b) / len(type_b)
    f1_c = sum(r["f1"] for r in type_c) / len(type_c)

    logger.info(f"  [{name}] EM={total_em:.0%} F1={total_f1:.2%} "
                f"NTP={ntp_final:.3f} QA={qa_final:.3f} {elapsed:.0f}s")

    # Cleanup
    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "name": name,
        "desc": acfg["desc"],
        "trainable_params": trainable,
        "proj_params": proj_params,
        "ntp_final_loss": ntp_final,
        "qa_final_loss": qa_final,
        "em": total_em,
        "f1": total_f1,
        "em_a": em_a, "f1_a": f1_a,
        "em_b": em_b, "f1_b": f1_b,
        "em_c": em_c, "f1_c": f1_c,
        "elapsed": elapsed,
        "per_sample": results,
    }


def print_comparison(all_results):
    """Print final comparison table."""
    print("\n" + "=" * 90)
    print("  PROJECTION ABLATION RESULTS")
    print("=" * 90)

    # Sort by F1 descending
    ranked = sorted(all_results, key=lambda r: r["f1"], reverse=True)

    print(f"\n  {'Config':<20} {'Proj Params':<12} {'QA Loss':<9} "
          f"{'EM':<7} {'F1':<7} {'A(num)':<8} {'B(sent)':<8} {'C(reason)':<8} {'Time':<6}")
    print("  " + "-" * 86)
    for r in ranked:
        proj_str = f"{r['proj_params']:,}" if r['proj_params'] > 0 else "0"
        print(f"  {r['name']:<20} {proj_str:<12} {r['qa_final_loss']:<9.4f} "
              f"{r['em']:.0%}{'':>3} {r['f1']:.2%}  "
              f"{r['f1_a']:.2f}{'':>3} {r['f1_b']:.2f}{'':>3} {r['f1_c']:.2f}{'':>3} "
              f"{r['elapsed']:.0f}s")

    best = ranked[0]
    print(f"\n  BEST: {best['name']} — {best['desc']}")
    print(f"        EM={best['em']:.0%}  F1={best['f1']:.2%}")

    # Per-sample detail for top 2
    print(f"\n  Per-sample comparison (top 2 vs baseline):")
    baseline = next(r for r in all_results if r["name"] == "A_mlp768")
    top1 = ranked[0]
    top2 = ranked[1] if len(ranked) > 1 else None

    print(f"\n  {'#':<3} {'Gold':<18} "
          f"{'baseline':<20} {'#1 '+top1['name']:<20}", end="")
    if top2 and top2["name"] != baseline["name"]:
        print(f" {'#2 '+top2['name']:<20}", end="")
    print()
    print("  " + "-" * 80)

    for i in range(len(FINANCIAL_QA)):
        gold = FINANCIAL_QA[i]["answer"][:16]
        bp = baseline["per_sample"][i]
        t1p = top1["per_sample"][i]
        b_mark = "Y" if bp["em"] > 0 else " "
        t1_mark = "Y" if t1p["em"] > 0 else " "
        b_pred = bp["pred"][:16] if bp["pred"] else "(empty)"
        t1_pred = t1p["pred"][:16] if t1p["pred"] else "(empty)"

        line = f"  {i+1:<3} {gold:<18} {b_pred:<16}[{b_mark}] {t1_pred:<16}[{t1_mark}]"

        if top2 and top2["name"] != baseline["name"]:
            t2p = top2["per_sample"][i]
            t2_mark = "Y" if t2p["em"] > 0 else " "
            t2_pred = t2p["pred"][:16] if t2p["pred"] else "(empty)"
            line += f" {t2_pred:<16}[{t2_mark}]"

        print(line)

    print("\n" + "=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Projection layer ablation")
    parser.add_argument("--ntp_steps", type=int, default=100)
    parser.add_argument("--qa_steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated config names to run (default: all)")
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    logger.info(f"Device: {device}")
    logger.info(f"NTP steps: {args.ntp_steps}, QA steps: {args.qa_steps}")

    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Filter configs if requested
    configs = ABLATION_CONFIGS
    if args.configs:
        names = set(args.configs.split(","))
        configs = [c for c in configs if c["name"] in names]
        logger.info(f"Running {len(configs)} configs: {[c['name'] for c in configs]}")

    # Need DataLoader import
    from torch.utils.data import DataLoader  # noqa: F811

    with tempfile.TemporaryDirectory() as tmpdir:
        ntp_path, _ = make_temp_data(tmpdir)

        all_results = []
        for i, acfg in enumerate(configs):
            logger.info(f"\n>>> Running config {i+1}/{len(configs)}: {acfg['name']}")
            result = run_one_config(
                acfg, tokenizer, device, ntp_path,
                args.ntp_steps, args.qa_steps, args.lr,
            )
            all_results.append(result)

    print_comparison(all_results)


if __name__ == "__main__":
    # DataLoader needs to be importable at module level for run_one_config
    from torch.utils.data import DataLoader  # noqa: F811
    main()
