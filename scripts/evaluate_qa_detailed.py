#!/usr/bin/env python3
"""Detailed QA evaluation across all experiments with per-sample analysis.

Auto-discovers checkpoint-best directories under outputs/, loads each model,
evaluates on both held-out dev set and a sample from the training set.

- **stdout**: summary statistics only (comparison table, bar charts, ranking)
- **markdown report**: full per-sample analysis saved to results/qa_eval_report.md

Usage:
    # Basic (500 dev + 100 train peek)
    python scripts/evaluate_qa_detailed.py

    # Full dev set
    python scripts/evaluate_qa_detailed.py --max_eval_samples 0

    # Specific checkpoints
    python scripts/evaluate_qa_detailed.py \
        --checkpoints outputs/qa_q256_8gpu,outputs/qa_q1024_8gpu

    # Multi-GPU
    accelerate launch --multi_gpu --num_processes 4 \
        scripts/evaluate_qa_detailed.py --max_eval_samples 1000
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

from deep_compressor.config import DeepCompressorConfig
from deep_compressor.data import PaddingCollator, QADataset
from deep_compressor.eval import (
    compute_exact_match,
    compute_f1,
    compute_rouge_l,
)
from deep_compressor.model import DeepCompressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Terminal colors ──────────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"

W = 100  # terminal output width


def bar(value: float, max_val: float, width: int = 30) -> str:
    if max_val <= 0:
        return ""
    filled = min(int(round(value / max_val * width)), width)
    return "\u2588" * filled + "\u2591" * (width - filled)


# ── Data containers ──────────────────────────────────────────────────────────


@dataclass
class SampleResult:
    idx: int
    question: str
    gold: str
    prediction: str
    context_preview: str
    em: float
    f1: float
    rouge_l: float
    source: str  # "dev" or "train"


@dataclass
class ModelResult:
    name: str
    q_value: int
    config_path: str
    checkpoint_path: str
    # Dev metrics
    dev_loss: float = 0.0
    dev_ppl: float = 0.0
    dev_em: float = 0.0
    dev_f1: float = 0.0
    dev_rouge_l: float = 0.0
    dev_n: int = 0
    dev_empty_preds: int = 0
    # Train metrics
    train_loss: float = 0.0
    train_ppl: float = 0.0
    train_em: float = 0.0
    train_f1: float = 0.0
    train_rouge_l: float = 0.0
    train_n: int = 0
    train_empty_preds: int = 0
    # Per-sample
    samples: List[SampleResult] = field(default_factory=list)


# ── Checkpoint discovery ─────────────────────────────────────────────────────


def discover_checkpoints(base_dir: str = "outputs") -> List[Tuple[Path, Path, int]]:
    """Return (checkpoint_dir, config_path, q_value) sorted by Q."""
    base = Path(base_dir)
    if not base.exists():
        return []

    found = []
    for exp_dir in sorted(base.iterdir()):
        if not exp_dir.is_dir():
            continue
        best = exp_dir / "checkpoint-best" / "trainable_weights.pt"
        final = exp_dir / "checkpoint-final" / "trainable_weights.pt"
        if best.exists():
            ckpt_dir = exp_dir / "checkpoint-best"
        elif final.exists():
            ckpt_dir = exp_dir / "checkpoint-final"
        else:
            continue

        name = exp_dir.name
        config_path = Path("configs") / f"{name}.yaml"
        if not config_path.exists():
            logger.warning(f"No config for {name}, skipping")
            continue

        m = re.search(r"_q(\d+)", name)
        q_value = int(m.group(1)) if m else -1
        found.append((ckpt_dir, config_path, q_value))

    found.sort(key=lambda x: x[2])
    return found


# ── Model loading ────────────────────────────────────────────────────────────


def load_model(checkpoint_dir: Path, config_path: Path,
               accelerator: Accelerator) -> Tuple[DeepCompressor, DeepCompressorConfig]:
    config = DeepCompressorConfig.from_yaml(str(config_path))
    model = DeepCompressor(config)
    weights = torch.load(
        checkpoint_dir / "trainable_weights.pt",
        map_location="cpu", weights_only=True,
    )
    model.load_state_dict(weights, strict=False)
    model = accelerator.prepare(model)
    return model, config


# ── Core evaluation ──────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_qa_detailed(
    model, eval_loader: DataLoader, tokenizer,
    accelerator: Accelerator, max_new_tokens: int = 64,
    source_label: str = "dev",
) -> Tuple[Dict[str, float], List[SampleResult]]:
    model.eval()
    unwrapped = accelerator.unwrap_model(model)

    all_em, all_f1, all_rouge, all_loss = [], [], [], []
    samples = []
    empty_count = 0
    global_idx = 0

    total_batches = len(eval_loader)
    for batch_idx, batch in enumerate(eval_loader):
        if accelerator.is_main_process and batch_idx % 20 == 0:
            pct = 100 * batch_idx / max(total_batches, 1)
            print(f"\r  [{source_label}] batch {batch_idx}/{total_batches} "
                  f"({pct:.0f}%)", end="", flush=True)

        # Encode + compress
        byte_array = unwrapped.encode_document(
            batch["doc_input_ids"], batch["doc_attention_mask"])
        queries = unwrapped.encode_question(
            batch["q_input_ids"], batch["q_attention_mask"])
        latent = unwrapped.compress(
            queries, byte_array, byte_mask=batch["doc_attention_mask"])
        prefix_embeds = unwrapped.up_mlp(latent)

        # Teacher-forcing loss
        suffix_ids = torch.cat([batch["q_input_ids"], batch["answer_ids"]], dim=1)
        suffix_mask = torch.cat(
            [batch["q_attention_mask"],
             torch.ones_like(batch["answer_ids"])], dim=1)
        q_labels = torch.full_like(batch["q_input_ids"], -100)
        full_labels = torch.cat([q_labels, batch["answer_labels"]], dim=1)
        outputs = unwrapped.decode(prefix_embeds, suffix_ids, suffix_mask,
                                   labels=full_labels)
        batch_loss = outputs.loss.detach().item()

        # Generate answers
        gen_ids = unwrapped.generate_answer(
            prefix_embeds, batch["q_input_ids"], batch["q_attention_mask"],
            tokenizer=tokenizer, max_new_tokens=max_new_tokens)
        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        golds = batch["answer_text"]
        questions = batch.get("question_text", [""] * len(preds))
        doc_previews = tokenizer.batch_decode(
            batch["doc_input_ids"][:, :120], skip_special_tokens=True)

        for i, (pred, gold) in enumerate(zip(preds, golds)):
            if not pred.strip():
                empty_count += 1
            em = compute_exact_match(pred, gold)
            f1 = compute_f1(pred, gold)
            rl = compute_rouge_l(pred, gold)
            all_em.append(em)
            all_f1.append(f1)
            all_rouge.append(rl)
            all_loss.append(batch_loss)

            q_text = questions[i] if isinstance(questions, list) and i < len(questions) else ""
            ctx = doc_previews[i] if i < len(doc_previews) else ""
            samples.append(SampleResult(
                idx=global_idx, question=q_text, gold=gold,
                prediction=pred, context_preview=ctx,
                em=em, f1=f1, rouge_l=rl, source=source_label,
            ))
            global_idx += 1

    if accelerator.is_main_process:
        print()  # newline after progress

    n = max(len(all_em), 1)
    avg_loss = sum(all_loss) / max(len(all_loss), 1)
    metrics = {
        "loss": avg_loss,
        "perplexity": torch.exp(torch.tensor(avg_loss)).item(),
        "exact_match": sum(all_em) / n,
        "f1": sum(all_f1) / n,
        "rouge_l": sum(all_rouge) / n,
        "n_samples": len(all_em),
        "empty_preds": empty_count,
    }
    return metrics, samples


# ── Baseline (raw Qwen) evaluation ───────────────────────────────────────────


def discover_baseline_models(models_dir: str = "models") -> List[Tuple[str, str]]:
    """Return list of (model_path, display_name) for raw models under models/."""
    base = Path(models_dir)
    if not base.exists():
        return []
    found = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        # Check it looks like a HF model dir
        if (d / "config.json").exists():
            found.append((str(d), d.name))
    return found


@torch.no_grad()
def evaluate_baseline_qa(
    qwen_model, eval_loader: DataLoader, tokenizer,
    accelerator: Accelerator, max_new_tokens: int = 64,
    source_label: str = "dev",
) -> Tuple[Dict[str, float], List[SampleResult]]:
    """Evaluate raw Qwen (no compression) on QA as upper bound.

    Feeds full [doc_tokens | question_tokens] as input, generates answer.
    """
    qwen_model.eval()
    unwrapped = accelerator.unwrap_model(qwen_model)

    all_em, all_f1, all_rouge, all_loss = [], [], [], []
    samples = []
    empty_count = 0
    global_idx = 0

    total_batches = len(eval_loader)
    for batch_idx, batch in enumerate(eval_loader):
        if accelerator.is_main_process and batch_idx % 20 == 0:
            pct = 100 * batch_idx / max(total_batches, 1)
            print(f"\r  [{source_label}] batch {batch_idx}/{total_batches} "
                  f"({pct:.0f}%)", end="", flush=True)

        doc_ids = batch["doc_input_ids"]
        doc_mask = batch["doc_attention_mask"]
        q_ids = batch["q_input_ids"]
        q_mask = batch["q_attention_mask"]
        ans_ids = batch["answer_ids"]
        ans_labels = batch["answer_labels"]

        # Concatenate: [doc | question | answer] for teacher-forcing loss
        input_ids = torch.cat([doc_ids, q_ids, ans_ids], dim=1)
        attn_mask = torch.cat([doc_mask, q_mask,
                               torch.ones_like(ans_ids)], dim=1)
        # Labels: -100 for doc+question, real labels for answer
        ignore_doc = torch.full_like(doc_ids, -100)
        ignore_q = torch.full_like(q_ids, -100)
        labels = torch.cat([ignore_doc, ignore_q, ans_labels], dim=1)

        outputs = unwrapped(
            input_ids=input_ids, attention_mask=attn_mask,
            labels=labels, use_cache=False,
        )
        batch_loss = outputs.loss.detach().item()

        # Generate: feed [doc | question], generate answer
        gen_input = torch.cat([doc_ids, q_ids], dim=1)
        gen_mask = torch.cat([doc_mask, q_mask], dim=1)
        gen_out = unwrapped.generate(
            input_ids=gen_input, attention_mask=gen_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )
        # Strip input prefix from generated output
        gen_only = gen_out[:, gen_input.shape[1]:]
        preds = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        golds = batch["answer_text"]
        questions = batch.get("question_text", [""] * len(preds))
        doc_previews = tokenizer.batch_decode(
            doc_ids[:, :120], skip_special_tokens=True)

        for i, (pred, gold) in enumerate(zip(preds, golds)):
            if not pred.strip():
                empty_count += 1
            em = compute_exact_match(pred, gold)
            f1 = compute_f1(pred, gold)
            rl = compute_rouge_l(pred, gold)
            all_em.append(em)
            all_f1.append(f1)
            all_rouge.append(rl)
            all_loss.append(batch_loss)

            q_text = questions[i] if isinstance(questions, list) and i < len(questions) else ""
            ctx = doc_previews[i] if i < len(doc_previews) else ""
            samples.append(SampleResult(
                idx=global_idx, question=q_text, gold=gold,
                prediction=pred, context_preview=ctx,
                em=em, f1=f1, rouge_l=rl, source=source_label,
            ))
            global_idx += 1

    if accelerator.is_main_process:
        print()

    n = max(len(all_em), 1)
    avg_loss = sum(all_loss) / max(len(all_loss), 1)
    metrics = {
        "loss": avg_loss,
        "perplexity": torch.exp(torch.tensor(avg_loss)).item(),
        "exact_match": sum(all_em) / n,
        "f1": sum(all_f1) / n,
        "rouge_l": sum(all_rouge) / n,
        "n_samples": len(all_em),
        "empty_preds": empty_count,
    }
    return metrics, samples


# ── Stdout: summary statistics ───────────────────────────────────────────────


def print_summary(results: List[ModelResult]):
    """Print concise summary to terminal."""

    # ── Comparison table ──
    print(f"\n{'=' * W}")
    print(f"  {BOLD}COMPARISON TABLE{RESET}")
    print(f"{'=' * W}")

    # Sub-header
    print(f"  {'':12}  {'─── Dev Set ───────────────────':^33}  "
          f"{'─── Train Peek ────────────────':^33}")
    header = (f"  {'Model':<12}  {'Loss':>6} {'PPL':>7} "
              f"{'EM':>7} {'F1':>7} {'RL':>7} {'Empty':>6}  "
              f"{'Loss':>6} {'PPL':>7} "
              f"{'EM':>7} {'F1':>7} {'RL':>7} {'Empty':>6}")
    print(f"{BOLD}{header}{RESET}")
    print(f"  {'─' * (W - 4)}")

    # Separate baselines (q_value <= 0) and compressed models
    baselines = [r for r in results if r.q_value <= 0]
    compressed = [r for r in results if r.q_value > 0]

    best_f1 = max((r.dev_f1 for r in results), default=0)
    best_em = max((r.dev_em for r in results), default=0)

    def _print_row(r):
        name = r.name if r.q_value <= 0 else f"Q={r.q_value}"
        em_s = f"{r.dev_em:>6.2%}"
        f1_s = f"{r.dev_f1:>7.4f}"
        if r.dev_f1 == best_f1 and len(results) > 1:
            f1_s = f"{GREEN}{f1_s}{RESET}"
        if r.dev_em == best_em and len(results) > 1:
            em_s = f"{GREEN}{em_s}{RESET}"
        de = f"{r.dev_empty_preds}" if r.dev_n > 0 else "-"
        te = f"{r.train_empty_preds}" if r.train_n > 0 else "-"
        print(f"  {name:<12}  {r.dev_loss:>6.3f} {r.dev_ppl:>7.2f} "
              f"{em_s} {f1_s} {r.dev_rouge_l:>7.4f} {de:>6}  "
              f"{r.train_loss:>6.3f} {r.train_ppl:>7.2f} "
              f"{r.train_em:>6.2%} {r.train_f1:>7.4f} {r.train_rouge_l:>7.4f} {te:>6}")

    # Baselines first (upper bound)
    if baselines:
        for r in baselines:
            _print_row(r)
        print(f"  {'─' * (W - 4)}")

    # Compressed models
    for r in compressed:
        _print_row(r)

    print(f"  {'─' * (W - 4)}")

    # ── Overfitting analysis ──
    print(f"\n  {BOLD}Overfitting (train_F1 - dev_F1):{RESET}")
    for r in results:
        if r.train_n == 0:
            continue
        label = r.name if r.q_value <= 0 else f"Q={r.q_value}"
        gap = r.train_f1 - r.dev_f1
        if gap > 0.10:
            tag = f"{RED}HIGH{RESET}"
        elif gap > 0.05:
            tag = f"{YELLOW}MODERATE{RESET}"
        else:
            tag = f"{GREEN}LOW{RESET}"
        print(f"    {label:<12} gap = {gap:+.4f}  {tag}")

    # ── Bar charts ──
    def _label(r):
        return r.name if r.q_value <= 0 else f"Q={r.q_value}"

    print(f"\n  {BOLD}Dev F1:{RESET}")
    max_f1 = max((r.dev_f1 for r in results), default=0.001)
    for r in results:
        b = bar(r.dev_f1, max_f1, 35)
        print(f"    {_label(r):<12} {CYAN}{b}{RESET} {r.dev_f1:.4f}")

    print(f"\n  {BOLD}Dev EM:{RESET}")
    max_em = max((r.dev_em for r in results), default=0.001)
    for r in results:
        b = bar(r.dev_em, max_em, 35)
        print(f"    {_label(r):<12} {MAGENTA}{b}{RESET} {r.dev_em:.2%}")

    print(f"\n  {BOLD}Dev PPL (lower=better):{RESET}")
    max_ppl = max((r.dev_ppl for r in results), default=1)
    for r in results:
        b = bar(r.dev_ppl, max_ppl, 35)
        print(f"    {_label(r):<12} {YELLOW}{b}{RESET} {r.dev_ppl:.2f}")

    # ── Ranking ──
    print(f"\n{'=' * W}")
    print(f"  {BOLD}RANKING (by Dev F1){RESET}")
    print(f"{'=' * W}")
    ranked = sorted(results, key=lambda r: r.dev_f1, reverse=True)
    medals = {0: "1.", 1: "2.", 2: "3."}
    for i, r in enumerate(ranked):
        m = medals.get(i, f"{i+1}.")
        label = r.name if r.q_value <= 0 else f"Q={r.q_value}"
        baseline_tag = " (baseline)" if r.q_value <= 0 else ""
        print(f"  {m:<4} {label:<12}  "
              f"F1={r.dev_f1:.4f}  EM={r.dev_em:.2%}  "
              f"PPL={r.dev_ppl:.2f}  Loss={r.dev_loss:.4f}  "
              f"({r.dev_n} samples){baseline_tag}")
    print(f"{'=' * W}\n")


# ── Markdown report ──────────────────────────────────────────────────────────


def write_markdown_report(results: List[ModelResult], report_path: str,
                          top_n: int = 10, args=None):
    """Write full per-sample analysis to a markdown file."""
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"# Deep Compressor QA Evaluation Report\n\n")
        f.write(f"Generated: {ts}\n\n")

        # Config
        if args:
            f.write("## Evaluation Config\n\n")
            f.write(f"| Parameter | Value |\n|---|---|\n")
            f.write(f"| Dev data | `{args.eval_data}` |\n")
            f.write(f"| Train data | `{args.train_data}` |\n")
            f.write(f"| Max dev samples | {args.max_eval_samples if args.max_eval_samples > 0 else 'ALL'} |\n")
            f.write(f"| Train peek samples | {args.train_peek_samples} |\n")
            f.write(f"| Models evaluated | {len(results)} |\n\n")

        # ── Summary table ──
        f.write("## Summary\n\n")
        def _md_name(r):
            return f"**{r.name}** (baseline)" if r.q_value <= 0 else f"Q={r.q_value}"

        f.write("### Dev Set\n\n")
        f.write("| Model | Loss | PPL | EM | F1 | ROUGE-L | Empty Preds | N |\n")
        f.write("|-------|------|-----|----|----|---------|-------------|---|\n")
        for r in results:
            f.write(f"| {_md_name(r)} | {r.dev_loss:.4f} | {r.dev_ppl:.2f} | "
                    f"{r.dev_em:.2%} | {r.dev_f1:.4f} | {r.dev_rouge_l:.4f} | "
                    f"{r.dev_empty_preds} ({100*r.dev_empty_preds/max(r.dev_n,1):.1f}%) | "
                    f"{r.dev_n} |\n")

        if any(r.train_n > 0 for r in results):
            f.write("\n### Train Peek\n\n")
            f.write("| Model | Loss | PPL | EM | F1 | ROUGE-L | Empty Preds | N |\n")
            f.write("|-------|------|-----|----|----|---------|-------------|---|\n")
            for r in results:
                if r.train_n == 0:
                    continue
                f.write(f"| {_md_name(r)} | {r.train_loss:.4f} | {r.train_ppl:.2f} | "
                        f"{r.train_em:.2%} | {r.train_f1:.4f} | {r.train_rouge_l:.4f} | "
                        f"{r.train_empty_preds} ({100*r.train_empty_preds/max(r.train_n,1):.1f}%) | "
                        f"{r.train_n} |\n")

        # ── Overfitting ──
        if any(r.train_n > 0 for r in results):
            f.write("\n### Overfitting Analysis\n\n")
            f.write("| Model | Dev F1 | Train F1 | Gap | Level |\n")
            f.write("|-------|--------|----------|-----|-------|\n")
            for r in results:
                if r.train_n == 0:
                    continue
                gap = r.train_f1 - r.dev_f1
                level = "HIGH" if gap > 0.10 else ("MODERATE" if gap > 0.05 else "LOW")
                f.write(f"| {_md_name(r)} | {r.dev_f1:.4f} | {r.train_f1:.4f} | "
                        f"{gap:+.4f} | {level} |\n")

        # ── Per-model sample details ──
        ranked = sorted(results, key=lambda r: r.dev_f1, reverse=True)
        for r in ranked:
            title = r.name if r.q_value <= 0 else f"Q={r.q_value}"
            baseline_note = " (Baseline — no compression)" if r.q_value <= 0 else ""
            f.write(f"\n---\n\n## {title}{baseline_note}\n\n")
            f.write(f"- **Model / Checkpoint**: `{r.checkpoint_path}`\n")
            if r.config_path:
                f.write(f"- **Config**: `{r.config_path}`\n")
            f.write(f"- **Dev**: Loss={r.dev_loss:.4f}  PPL={r.dev_ppl:.2f}  "
                    f"EM={r.dev_em:.2%}  F1={r.dev_f1:.4f}  ROUGE-L={r.dev_rouge_l:.4f}\n")
            if r.train_n > 0:
                f.write(f"- **Train**: Loss={r.train_loss:.4f}  PPL={r.train_ppl:.2f}  "
                        f"EM={r.train_em:.2%}  F1={r.train_f1:.4f}  ROUGE-L={r.train_rouge_l:.4f}\n")
            f.write(f"- **Empty predictions**: dev={r.dev_empty_preds}/{r.dev_n}")
            if r.train_n > 0:
                f.write(f", train={r.train_empty_preds}/{r.train_n}")
            f.write("\n\n")

            for source_label in ["dev", "train"]:
                source_samples = [s for s in r.samples if s.source == source_label]
                if not source_samples:
                    continue

                tag = "Dev Set" if source_label == "dev" else "Train Set (peek)"
                sorted_by_f1 = sorted(source_samples, key=lambda s: s.f1, reverse=True)

                # Best N
                best = sorted_by_f1[:top_n]
                f.write(f"### Best {len(best)} — {tag}\n\n")
                _write_sample_table(f, best)

                # Worst N
                worst = sorted_by_f1[-top_n:]
                worst.reverse()
                f.write(f"### Worst {len(worst)} — {tag}\n\n")
                _write_sample_table(f, worst)

                # Empty predictions (if any)
                empty = [s for s in source_samples if not s.prediction.strip()]
                if empty:
                    shown = empty[:5]
                    f.write(f"### Empty Predictions ({len(empty)} total) — {tag}\n\n")
                    _write_sample_table(f, shown)

    logger.info(f"Markdown report saved to {report_path}")


def _write_sample_table(f, samples: List[SampleResult]):
    """Write a list of samples as a readable markdown section."""
    for i, s in enumerate(samples, 1):
        em_icon = "\u2705" if s.em == 1.0 else "\u274c"
        f.write(f"**#{i}** {em_icon} EM={s.em:.0f} | F1={s.f1:.4f} | ROUGE-L={s.rouge_l:.4f}\n\n")
        # Use blockquote for context
        ctx = s.context_preview[:200].replace("\n", " ")
        f.write(f"> **Context**: {ctx}...\n\n")
        f.write(f"- **Question**: {s.question}\n")
        f.write(f"- **Gold**: {s.gold}\n")
        pred_display = s.prediction if s.prediction.strip() else "*(empty)*"
        f.write(f"- **Prediction**: {pred_display}\n\n")
    f.write("\n")


# ── CSV export ───────────────────────────────────────────────────────────────


def save_csv(results: List[ModelResult], output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "q_value", "checkpoint",
        "dev_loss", "dev_ppl", "dev_em", "dev_f1", "dev_rouge_l",
        "dev_empty_preds", "dev_n",
        "train_loss", "train_ppl", "train_em", "train_f1", "train_rouge_l",
        "train_empty_preds", "train_n",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(results, key=lambda x: x.q_value):
            writer.writerow({
                "model": r.name, "q_value": r.q_value,
                "checkpoint": r.checkpoint_path,
                "dev_loss": f"{r.dev_loss:.6f}",
                "dev_ppl": f"{r.dev_ppl:.4f}",
                "dev_em": f"{r.dev_em:.6f}",
                "dev_f1": f"{r.dev_f1:.6f}",
                "dev_rouge_l": f"{r.dev_rouge_l:.6f}",
                "dev_empty_preds": r.dev_empty_preds,
                "dev_n": r.dev_n,
                "train_loss": f"{r.train_loss:.6f}",
                "train_ppl": f"{r.train_ppl:.4f}",
                "train_em": f"{r.train_em:.6f}",
                "train_f1": f"{r.train_f1:.6f}",
                "train_rouge_l": f"{r.train_rouge_l:.6f}",
                "train_empty_preds": r.train_empty_preds,
                "train_n": r.train_n,
            })
    logger.info(f"CSV saved to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Detailed QA evaluation across all experiments")
    parser.add_argument("--eval_data", type=str,
                        default="data/qa_large_dev.json",
                        help="Held-out dev set (default: data/qa_large_dev.json)")
    parser.add_argument("--train_data", type=str,
                        default="data/qa_large_train.json",
                        help="Training set for peek (default: data/qa_large_train.json)")
    parser.add_argument("--max_eval_samples", type=int, default=500,
                        help="Max dev samples (0=all, default=500)")
    parser.add_argument("--train_peek_samples", type=int, default=100,
                        help="Training samples to peek (default=100)")
    parser.add_argument("--checkpoints", type=str, default=None,
                        help="Comma-separated checkpoint dirs (default: auto-discover)")
    parser.add_argument("--outputs_dir", type=str, default="outputs",
                        help="Base outputs dir (default: outputs)")
    parser.add_argument("--models_dir", type=str, default="models",
                        help="Dir containing baseline models (default: models)")
    parser.add_argument("--no_baselines", action="store_true",
                        help="Skip baseline model evaluation")
    parser.add_argument("--top_n", type=int, default=10,
                        help="Best/worst samples per model in report (default=10)")
    parser.add_argument("--batch_size", type=int, default=0,
                        help="Override batch size (0=use config default)")
    parser.add_argument("--report", type=str, default="results/qa_eval_report.md",
                        help="Markdown report path (default: results/qa_eval_report.md)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Also save CSV summary")
    args = parser.parse_args()

    # ── Discover checkpoints ──
    if args.checkpoints:
        entries = []
        for p in args.checkpoints.split(","):
            p = p.strip()
            exp_dir = Path(p)
            if (exp_dir / "checkpoint-best" / "trainable_weights.pt").exists():
                ckpt_dir = exp_dir / "checkpoint-best"
            elif (exp_dir / "checkpoint-final" / "trainable_weights.pt").exists():
                ckpt_dir = exp_dir / "checkpoint-final"
            elif (exp_dir / "trainable_weights.pt").exists():
                ckpt_dir = exp_dir
            else:
                logger.warning(f"No checkpoint in {p}, skipping")
                continue
            name = exp_dir.name
            config_path = Path("configs") / f"{name}.yaml"
            if not config_path.exists():
                logger.warning(f"No config for {name}, skipping")
                continue
            m = re.search(r"_q(\d+)", name)
            q_val = int(m.group(1)) if m else -1
            entries.append((ckpt_dir, config_path, q_val))
        entries.sort(key=lambda x: x[2])
    else:
        entries = discover_checkpoints(args.outputs_dir)

    if not entries:
        logger.error("No checkpoints found. Train some models first.")
        return

    # ── Initialize ──
    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    if is_main:
        baseline_models = [] if args.no_baselines else discover_baseline_models(args.models_dir)
        print(f"\n{'=' * W}")
        print(f"  {BOLD}DEEP COMPRESSOR — QA EVALUATION{RESET}")
        print(f"{'=' * W}")
        print(f"  Compressed:  {len(entries)}  ({', '.join(f'Q={q}' for _, _, q in entries)})")
        if baseline_models:
            print(f"  Baselines:   {len(baseline_models)}  ({', '.join(n for _, n in baseline_models)})")
        print(f"  Dev data:    {args.eval_data}")
        print(f"  Dev samples: {args.max_eval_samples if args.max_eval_samples > 0 else 'ALL'}")
        print(f"  Train peek:  {args.train_peek_samples}")
        print(f"  Report:      {args.report}")
        print(f"  Device:      {accelerator.device}")
        print()

    # ── Load tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(
        "models/Qwen3-0.6B", trust_remote_code=True, fix_mistral_regex=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Shared data loading helpers ──
    # Use first compressed model's config for data params (doc/question/answer lengths).
    # Baselines and compressed models eval on the same data split for fair comparison.
    ref_config = DeepCompressorConfig.from_yaml(str(entries[0][1]))
    max_doc = ref_config.qwen.max_doc_tokens
    max_q = ref_config.qwen.max_question_tokens
    max_ans = ref_config.qwen.max_answer_tokens

    # ── Evaluate each model ──
    all_results: List[ModelResult] = []

    # ── Baseline models (raw Qwen, no compression) ──
    if not args.no_baselines:
        baseline_models = discover_baseline_models(args.models_dir)
        for model_path, display_name in baseline_models:
            if is_main:
                print(f"{'─' * W}")
                print(f"  {BOLD}{display_name} (baseline){RESET}  ({model_path})")
                print(f"  Loading model...")

            qwen = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
            qwen.eval()
            for p in qwen.parameters():
                p.requires_grad = False
            qwen = accelerator.prepare(qwen)

            collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
            nw = 0 if accelerator.device.type == "mps" else 2
            pin = accelerator.device.type == "cuda"
            bs = args.batch_size if args.batch_size > 0 else 8  # conservative for full-context

            mr = ModelResult(
                name=display_name, q_value=0,
                config_path="", checkpoint_path=model_path,
            )

            # Dev
            if is_main:
                print(f"  Loading dev data...")
            dev_ds = QADataset(
                args.eval_data, tokenizer,
                max_doc_tokens=max_doc, max_question_tokens=max_q,
                max_answer_tokens=max_ans,
            )
            if args.max_eval_samples > 0 and args.max_eval_samples < len(dev_ds):
                dev_ds = Subset(dev_ds, list(range(args.max_eval_samples)))

            dev_loader = DataLoader(dev_ds, batch_size=bs, shuffle=False,
                                    collate_fn=collator, num_workers=nw, pin_memory=pin)
            dev_loader = accelerator.prepare(dev_loader)

            dev_m, dev_s = evaluate_baseline_qa(
                qwen, dev_loader, tokenizer, accelerator,
                max_new_tokens=max_ans, source_label="dev")

            mr.dev_loss = dev_m["loss"]
            mr.dev_ppl = dev_m["perplexity"]
            mr.dev_em = dev_m["exact_match"]
            mr.dev_f1 = dev_m["f1"]
            mr.dev_rouge_l = dev_m["rouge_l"]
            mr.dev_n = dev_m["n_samples"]
            mr.dev_empty_preds = dev_m["empty_preds"]
            mr.samples.extend(dev_s)

            if is_main:
                print(f"  {GREEN}Dev:{RESET}   Loss={mr.dev_loss:.4f}  PPL={mr.dev_ppl:.2f}  "
                      f"EM={mr.dev_em:.2%}  F1={mr.dev_f1:.4f}  "
                      f"Empty={mr.dev_empty_preds}/{mr.dev_n}")

            # Train peek
            if args.train_peek_samples > 0 and os.path.exists(args.train_data):
                train_ds = QADataset(
                    args.train_data, tokenizer,
                    max_doc_tokens=max_doc, max_question_tokens=max_q,
                    max_answer_tokens=max_ans,
                )
                if args.train_peek_samples < len(train_ds):
                    train_ds = Subset(train_ds, list(range(args.train_peek_samples)))

                train_loader = DataLoader(train_ds, batch_size=bs, shuffle=False,
                                          collate_fn=collator, num_workers=nw, pin_memory=pin)
                train_loader = accelerator.prepare(train_loader)

                tr_m, tr_s = evaluate_baseline_qa(
                    qwen, train_loader, tokenizer, accelerator,
                    max_new_tokens=max_ans, source_label="train")

                mr.train_loss = tr_m["loss"]
                mr.train_ppl = tr_m["perplexity"]
                mr.train_em = tr_m["exact_match"]
                mr.train_f1 = tr_m["f1"]
                mr.train_rouge_l = tr_m["rouge_l"]
                mr.train_n = tr_m["n_samples"]
                mr.train_empty_preds = tr_m["empty_preds"]
                mr.samples.extend(tr_s)

                if is_main:
                    print(f"  {YELLOW}Train:{RESET} Loss={mr.train_loss:.4f}  PPL={mr.train_ppl:.2f}  "
                          f"EM={mr.train_em:.2%}  F1={mr.train_f1:.4f}  "
                          f"Empty={mr.train_empty_preds}/{mr.train_n}")

            all_results.append(mr)
            del qwen
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Compressed models (DeepCompressor) ──
    for ckpt_dir, config_path, q_value in entries:
        if is_main:
            print(f"{'─' * W}")
            print(f"  {BOLD}Q={q_value}{RESET}  ({ckpt_dir})")

        model, config = load_model(ckpt_dir, config_path, accelerator)
        bs = args.batch_size if args.batch_size > 0 else config.training.batch_size
        max_ans = config.qwen.max_answer_tokens
        collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
        nw = 0 if accelerator.device.type == "mps" else 2
        pin = accelerator.device.type == "cuda"

        mr = ModelResult(
            name=f"Q={q_value}", q_value=q_value,
            config_path=str(config_path), checkpoint_path=str(ckpt_dir),
        )

        # Dev
        if is_main:
            print(f"  Loading dev data...")
        dev_ds = QADataset(
            args.eval_data, tokenizer,
            max_doc_tokens=config.qwen.max_doc_tokens,
            max_question_tokens=config.qwen.max_question_tokens,
            max_answer_tokens=config.qwen.max_answer_tokens,
        )
        if args.max_eval_samples > 0 and args.max_eval_samples < len(dev_ds):
            dev_ds = Subset(dev_ds, list(range(args.max_eval_samples)))

        dev_loader = DataLoader(dev_ds, batch_size=bs, shuffle=False,
                                collate_fn=collator, num_workers=nw, pin_memory=pin)
        dev_loader = accelerator.prepare(dev_loader)

        dev_m, dev_s = evaluate_qa_detailed(
            model, dev_loader, tokenizer, accelerator,
            max_new_tokens=max_ans, source_label="dev")

        mr.dev_loss = dev_m["loss"]
        mr.dev_ppl = dev_m["perplexity"]
        mr.dev_em = dev_m["exact_match"]
        mr.dev_f1 = dev_m["f1"]
        mr.dev_rouge_l = dev_m["rouge_l"]
        mr.dev_n = dev_m["n_samples"]
        mr.dev_empty_preds = dev_m["empty_preds"]
        mr.samples.extend(dev_s)

        if is_main:
            print(f"  {GREEN}Dev:{RESET}   Loss={mr.dev_loss:.4f}  PPL={mr.dev_ppl:.2f}  "
                  f"EM={mr.dev_em:.2%}  F1={mr.dev_f1:.4f}  "
                  f"Empty={mr.dev_empty_preds}/{mr.dev_n}")

        # Train peek
        if args.train_peek_samples > 0 and os.path.exists(args.train_data):
            train_ds = QADataset(
                args.train_data, tokenizer,
                max_doc_tokens=config.qwen.max_doc_tokens,
                max_question_tokens=config.qwen.max_question_tokens,
                max_answer_tokens=config.qwen.max_answer_tokens,
            )
            if args.train_peek_samples < len(train_ds):
                train_ds = Subset(train_ds, list(range(args.train_peek_samples)))

            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=False,
                                      collate_fn=collator, num_workers=nw, pin_memory=pin)
            train_loader = accelerator.prepare(train_loader)

            tr_m, tr_s = evaluate_qa_detailed(
                model, train_loader, tokenizer, accelerator,
                max_new_tokens=max_ans, source_label="train")

            mr.train_loss = tr_m["loss"]
            mr.train_ppl = tr_m["perplexity"]
            mr.train_em = tr_m["exact_match"]
            mr.train_f1 = tr_m["f1"]
            mr.train_rouge_l = tr_m["rouge_l"]
            mr.train_n = tr_m["n_samples"]
            mr.train_empty_preds = tr_m["empty_preds"]
            mr.samples.extend(tr_s)

            if is_main:
                print(f"  {YELLOW}Train:{RESET} Loss={mr.train_loss:.4f}  PPL={mr.train_ppl:.2f}  "
                      f"EM={mr.train_em:.2%}  F1={mr.train_f1:.4f}  "
                      f"Empty={mr.train_empty_preds}/{mr.train_n}")

        all_results.append(mr)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Output (main process only) ──
    if not is_main:
        return

    # Terminal: summary only
    print_summary(all_results)

    # File: full report
    write_markdown_report(all_results, args.report, top_n=args.top_n, args=args)
    print(f"  Report saved to: {BOLD}{args.report}{RESET}")

    if args.csv:
        save_csv(all_results, args.csv)
        print(f"  CSV saved to:    {BOLD}{args.csv}{RESET}")

    print()


if __name__ == "__main__":
    main()
