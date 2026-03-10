#!/usr/bin/env python3
"""Detailed QA evaluation across all experiments with per-sample analysis.

Auto-discovers checkpoint-best directories under outputs/, loads each model,
evaluates on both held-out dev set and a sample from the training set, then
prints a comprehensive comparison table followed by ranked per-model samples
(best 10 and worst 10 by F1).

Usage:
    # Basic: evaluate all experiments on dev set (500 samples) + train peek (100 samples)
    python scripts/evaluate_qa_detailed.py

    # Full dev set evaluation
    python scripts/evaluate_qa_detailed.py --max_eval_samples 0

    # Custom paths
    python scripts/evaluate_qa_detailed.py \
        --eval_data data/qa_large_dev.json \
        --train_data data/qa_large_train.json \
        --max_eval_samples 500 \
        --train_peek_samples 100

    # Evaluate specific checkpoints only
    python scripts/evaluate_qa_detailed.py \
        --checkpoints outputs/qa_q256_8gpu,outputs/qa_q1024_8gpu

    # Multi-GPU
    accelerate launch --multi_gpu --num_processes 4 \
        scripts/evaluate_qa_detailed.py --max_eval_samples 1000

    # Save results to CSV
    python scripts/evaluate_qa_detailed.py --output results/qa_detailed.csv
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from deep_compressor.config import DeepCompressorConfig
from deep_compressor.data import PaddingCollator, QADataset
from deep_compressor.eval import (
    compute_exact_match,
    compute_f1,
    compute_rouge_l,
    normalize_text,
)
from deep_compressor.model import DeepCompressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Pretty-print helpers ─────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
UNDERLINE = "\033[4m"

W = 110  # total output width


def hline(char="─"):
    return char * W


def section(title: str):
    pad = (W - len(title) - 4) // 2
    return f"\n{'━' * pad}  {BOLD}{title}{RESET}  {'━' * pad}"


def bar(value: float, max_val: float, width: int = 30) -> str:
    """Render a horizontal bar chart character."""
    if max_val <= 0:
        return ""
    filled = int(round(value / max_val * width))
    filled = min(filled, width)
    return "█" * filled + "░" * (width - filled)


# ── Per-sample result container ──────────────────────────────────────────────


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
    # Aggregate metrics (dev set)
    dev_loss: float = 0.0
    dev_ppl: float = 0.0
    dev_em: float = 0.0
    dev_f1: float = 0.0
    dev_rouge_l: float = 0.0
    dev_n: int = 0
    # Aggregate metrics (train peek)
    train_loss: float = 0.0
    train_ppl: float = 0.0
    train_em: float = 0.0
    train_f1: float = 0.0
    train_rouge_l: float = 0.0
    train_n: int = 0
    # Per-sample results
    samples: List[SampleResult] = field(default_factory=list)


# ── Checkpoint discovery ─────────────────────────────────────────────────────


def discover_checkpoints(base_dir: str = "outputs") -> List[Tuple[Path, Path, int]]:
    """Return list of (checkpoint_dir, config_path, q_value) sorted by Q."""
    base = Path(base_dir)
    if not base.exists():
        return []

    found = []
    for exp_dir in sorted(base.iterdir()):
        if not exp_dir.is_dir():
            continue
        # Prefer checkpoint-best, fallback to checkpoint-final
        best = exp_dir / "checkpoint-best" / "trainable_weights.pt"
        final = exp_dir / "checkpoint-final" / "trainable_weights.pt"
        if best.exists():
            ckpt_dir = exp_dir / "checkpoint-best"
        elif final.exists():
            ckpt_dir = exp_dir / "checkpoint-final"
        else:
            continue

        # Find matching config
        name = exp_dir.name  # e.g. "qa_q1024_8gpu"
        config_path = Path("configs") / f"{name}.yaml"
        if not config_path.exists():
            logger.warning(f"No config found for {name}, skipping")
            continue

        # Extract Q value
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


# ── Core evaluation (returns per-sample details) ─────────────────────────────


@torch.no_grad()
def evaluate_qa_detailed(
    model,
    eval_loader: DataLoader,
    tokenizer,
    accelerator: Accelerator,
    max_new_tokens: int = 64,
    source_label: str = "dev",
) -> Tuple[Dict[str, float], List[SampleResult]]:
    """Evaluate QA and return both aggregates and per-sample results.

    Returns:
        (metrics_dict, list_of_SampleResult)
    """
    model.eval()
    unwrapped = accelerator.unwrap_model(model)

    all_em, all_f1, all_rouge, all_loss = [], [], [], []
    samples = []
    global_idx = 0

    total_batches = len(eval_loader)
    for batch_idx, batch in enumerate(eval_loader):
        if accelerator.is_main_process and batch_idx % 20 == 0:
            pct = 100 * batch_idx / max(total_batches, 1)
            print(f"\r  [{source_label}] batch {batch_idx}/{total_batches} ({pct:.0f}%)",
                  end="", flush=True)

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
        q_len = batch["q_input_ids"].shape[1]
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

        # Decode doc preview (first 120 tokens)
        doc_previews = tokenizer.batch_decode(
            batch["doc_input_ids"][:, :120], skip_special_tokens=True)

        for i, (pred, gold) in enumerate(zip(preds, golds)):
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
    }
    return metrics, samples


# ── Pretty output functions ──────────────────────────────────────────────────


def print_comparison_table(results: List[ModelResult]):
    """Print a comprehensive comparison table."""
    print(section("COMPARISON TABLE"))
    print()

    # Header
    h1 = (f"  {'Model':<12} │ {'Loss':>7} {'PPL':>7} │ "
           f"{'EM':>7} {'F1':>7} {'ROUGE-L':>7} │ "
           f"{'Loss':>7} {'PPL':>7} │ "
           f"{'EM':>7} {'F1':>7} {'ROUGE-L':>7}")
    h0 = (f"  {'':12} │ {'── Dev Set ──────────':^24} │ "
           f"{'── Train Peek ─────────────':^24} │ "
           f"{'':7}")
    print(f"{DIM}{h0}{RESET}")
    print(f"{BOLD}{h1}{RESET}")
    print(f"  {'─'*12}─┼─{'─'*23}─┼─{'─'*23}─┼─{'─'*23}")

    # Find best values for highlighting
    best_dev_f1 = max((r.dev_f1 for r in results), default=0)
    best_dev_em = max((r.dev_em for r in results), default=0)

    for r in results:
        name = f"Q={r.q_value}"

        # Highlight best model
        em_str = f"{r.dev_em:>6.2%}"
        f1_str = f"{r.dev_f1:>7.4f}"
        if r.dev_f1 == best_dev_f1:
            f1_str = f"{GREEN}{f1_str}{RESET}"
        if r.dev_em == best_dev_em:
            em_str = f"{GREEN}{em_str}{RESET}"

        row = (f"  {name:<12} │ {r.dev_loss:>7.4f} {r.dev_ppl:>7.2f} │ "
               f"{em_str} {f1_str} {r.dev_rouge_l:>7.4f} │ "
               f"{r.train_loss:>7.4f} {r.train_ppl:>7.2f} │ "
               f"{r.train_em:>6.2%} {r.train_f1:>7.4f} {r.train_rouge_l:>7.4f}")
        print(row)

    print(f"  {'─'*12}─┴─{'─'*23}─┴─{'─'*23}─┴─{'─'*23}")

    # Overfitting indicator
    print()
    print(f"  {BOLD}Overfitting Analysis:{RESET}")
    for r in results:
        gap = r.train_f1 - r.dev_f1
        if gap > 0.1:
            indicator = f"{RED}HIGH{RESET} (gap={gap:.4f})"
        elif gap > 0.05:
            indicator = f"{YELLOW}MODERATE{RESET} (gap={gap:.4f})"
        else:
            indicator = f"{GREEN}LOW{RESET} (gap={gap:.4f})"
        print(f"    Q={r.q_value:<5}  train_F1 - dev_F1 = {gap:+.4f}  → {indicator}")
    print()


def print_bar_chart(results: List[ModelResult]):
    """Print a visual bar chart of key metrics."""
    print(section("VISUAL COMPARISON"))
    print()

    max_f1 = max((r.dev_f1 for r in results), default=0.001)
    max_em = max((r.dev_em for r in results), default=0.001)

    print(f"  {BOLD}Dev F1 Score:{RESET}")
    for r in results:
        b = bar(r.dev_f1, max_f1, width=40)
        print(f"    Q={r.q_value:<5} {CYAN}{b}{RESET} {r.dev_f1:.4f}")
    print()

    print(f"  {BOLD}Dev Exact Match:{RESET}")
    for r in results:
        b = bar(r.dev_em, max_em, width=40)
        print(f"    Q={r.q_value:<5} {MAGENTA}{b}{RESET} {r.dev_em:.2%}")
    print()

    print(f"  {BOLD}Dev Perplexity (lower is better):{RESET}")
    max_ppl = max((r.dev_ppl for r in results), default=1.0)
    for r in results:
        b = bar(r.dev_ppl, max_ppl, width=40)
        print(f"    Q={r.q_value:<5} {YELLOW}{b}{RESET} {r.dev_ppl:.2f}")
    print()


def print_model_samples(r: ModelResult, top_n: int = 10):
    """Print best and worst samples for a single model."""
    dev_samples = [s for s in r.samples if s.source == "dev"]
    train_samples = [s for s in r.samples if s.source == "train"]

    print(section(f"Q={r.q_value}  |  Dev EM={r.dev_em:.2%}  F1={r.dev_f1:.4f}  PPL={r.dev_ppl:.2f}"))
    print(f"  {DIM}checkpoint: {r.checkpoint_path}{RESET}")
    print(f"  {DIM}config:     {r.config_path}{RESET}")
    print()

    for label, sample_list in [("DEV SET", dev_samples), ("TRAIN SET (peek)", train_samples)]:
        if not sample_list:
            continue

        sorted_best = sorted(sample_list, key=lambda s: s.f1, reverse=True)

        # ── Best N ──
        best = sorted_best[:top_n]
        print(f"  {GREEN}{BOLD}▲ BEST {len(best)} samples ({label}){RESET}")
        print(f"  {hline()}")
        for rank, s in enumerate(best, 1):
            _print_single_sample(rank, s)

        # ── Worst N ──
        worst = sorted_best[-top_n:]
        worst.reverse()  # worst first
        print()
        print(f"  {RED}{BOLD}▼ WORST {len(worst)} samples ({label}){RESET}")
        print(f"  {hline()}")
        for rank, s in enumerate(worst, 1):
            _print_single_sample(rank, s)

        print()


def _print_single_sample(rank: int, s: SampleResult):
    """Print one sample with formatting."""
    # Color code by quality
    if s.f1 >= 0.8:
        color = GREEN
    elif s.f1 >= 0.3:
        color = YELLOW
    else:
        color = RED

    em_icon = "✓" if s.em == 1.0 else "✗"
    em_color = GREEN if s.em == 1.0 else RED

    print(f"  {DIM}#{rank:<3}{RESET} "
          f"{em_color}{em_icon}{RESET} "
          f"EM={s.em:.0f}  {color}F1={s.f1:.4f}{RESET}  ROUGE-L={s.rouge_l:.4f}")

    # Truncate long texts for readability
    q_disp = s.question[:100] + ("..." if len(s.question) > 100 else "")
    g_disp = s.gold[:120] + ("..." if len(s.gold) > 120 else "")
    p_disp = s.prediction[:120] + ("..." if len(s.prediction) > 120 else "")
    c_disp = s.context_preview[:100] + ("..." if len(s.context_preview) > 100 else "")

    print(f"       {DIM}Context:{RESET} {c_disp}")
    print(f"       {BOLD}Q:{RESET} {q_disp}")
    print(f"       {GREEN}Gold:{RESET} {g_disp}")
    print(f"       {CYAN}Pred:{RESET} {p_disp}")
    print(f"  {DIM}{hline('·')}{RESET}")


def print_summary_ranking(results: List[ModelResult]):
    """Print a final one-line-per-model ranking."""
    print(section("FINAL RANKING (by Dev F1)"))
    print()
    ranked = sorted(results, key=lambda r: r.dev_f1, reverse=True)
    for i, r in enumerate(ranked, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"  {i}.")
        print(f"  {medal}  Q={r.q_value:<5}  "
              f"F1={r.dev_f1:.4f}  EM={r.dev_em:.2%}  "
              f"PPL={r.dev_ppl:.2f}  Loss={r.dev_loss:.4f}  "
              f"(n={r.dev_n})")
    print()


# ── CSV export ───────────────────────────────────────────────────────────────


def save_csv(results: List[ModelResult], output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "q_value", "checkpoint",
        "dev_loss", "dev_ppl", "dev_em", "dev_f1", "dev_rouge_l", "dev_n",
        "train_loss", "train_ppl", "train_em", "train_f1", "train_rouge_l", "train_n",
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
                "dev_n": r.dev_n,
                "train_loss": f"{r.train_loss:.6f}",
                "train_ppl": f"{r.train_ppl:.4f}",
                "train_em": f"{r.train_em:.6f}",
                "train_f1": f"{r.train_f1:.6f}",
                "train_rouge_l": f"{r.train_rouge_l:.6f}",
                "train_n": r.train_n,
            })
    logger.info(f"Results saved to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Detailed QA evaluation across all experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--eval_data", type=str,
                        default="data/qa_large_dev.json",
                        help="Path to held-out dev set (default: data/qa_large_dev.json)")
    parser.add_argument("--train_data", type=str,
                        default="data/qa_large_train.json",
                        help="Path to training set for peek (default: data/qa_large_train.json)")
    parser.add_argument("--max_eval_samples", type=int, default=500,
                        help="Max dev samples to evaluate (0=all, default=500)")
    parser.add_argument("--train_peek_samples", type=int, default=100,
                        help="Number of training samples to peek at (default=100)")
    parser.add_argument("--checkpoints", type=str, default=None,
                        help="Comma-separated checkpoint dirs (default: auto-discover)")
    parser.add_argument("--outputs_dir", type=str, default="outputs",
                        help="Base outputs directory (default: outputs)")
    parser.add_argument("--top_n", type=int, default=10,
                        help="Number of best/worst samples to show per model (default=10)")
    parser.add_argument("--batch_size", type=int, default=0,
                        help="Override batch size (0=use config default)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save aggregate results to CSV")
    args = parser.parse_args()

    # ── Discover checkpoints ──────────────────────────────────────────────
    if args.checkpoints:
        entries = []
        for p in args.checkpoints.split(","):
            p = p.strip()
            exp_dir = Path(p)
            # If user gave the experiment dir, find the checkpoint subdir
            if (exp_dir / "checkpoint-best" / "trainable_weights.pt").exists():
                ckpt_dir = exp_dir / "checkpoint-best"
            elif (exp_dir / "checkpoint-final" / "trainable_weights.pt").exists():
                ckpt_dir = exp_dir / "checkpoint-final"
            elif (exp_dir / "trainable_weights.pt").exists():
                ckpt_dir = exp_dir
            else:
                logger.warning(f"No checkpoint found in {p}, skipping")
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

    # ── Initialize ────────────────────────────────────────────────────────
    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    if is_main:
        print(section("DEEP COMPRESSOR — DETAILED QA EVALUATION"))
        print(f"  Checkpoints found: {len(entries)}")
        for ckpt, cfg, q in entries:
            print(f"    Q={q:<5}  {ckpt}")
        print(f"  Dev data:       {args.eval_data}")
        print(f"  Train data:     {args.train_data}")
        print(f"  Dev samples:    {args.max_eval_samples if args.max_eval_samples > 0 else 'ALL'}")
        print(f"  Train peek:     {args.train_peek_samples}")
        print(f"  Device:         {accelerator.device}")
        print(f"  Processes:      {accelerator.num_processes}")
        print()

    # ── Load tokenizer (shared) ───────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        "models/Qwen3-0.6B", trust_remote_code=True, fix_mistral_regex=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Evaluate each model ───────────────────────────────────────────────
    all_results: List[ModelResult] = []

    for ckpt_dir, config_path, q_value in entries:
        if is_main:
            print(section(f"Evaluating Q={q_value}"))

        # Load model
        model, config = load_model(ckpt_dir, config_path, accelerator)
        bs = args.batch_size if args.batch_size > 0 else config.training.batch_size
        max_ans_tokens = config.qwen.max_answer_tokens
        collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
        num_workers = 0 if accelerator.device.type == "mps" else 2
        pin_memory = accelerator.device.type == "cuda"

        mr = ModelResult(
            name=f"Q={q_value}",
            q_value=q_value,
            config_path=str(config_path),
            checkpoint_path=str(ckpt_dir),
        )

        # ── Dev set evaluation ────────────────────────────────────────
        if is_main:
            print(f"  Loading dev data: {args.eval_data}")
        dev_ds = QADataset(
            args.eval_data, tokenizer,
            max_doc_tokens=config.qwen.max_doc_tokens,
            max_question_tokens=config.qwen.max_question_tokens,
            max_answer_tokens=config.qwen.max_answer_tokens,
        )
        if args.max_eval_samples > 0 and args.max_eval_samples < len(dev_ds):
            dev_ds = Subset(dev_ds, list(range(args.max_eval_samples)))
        if is_main:
            print(f"  Dev samples: {len(dev_ds)}")

        dev_loader = DataLoader(dev_ds, batch_size=bs, shuffle=False,
                                collate_fn=collator, num_workers=num_workers,
                                pin_memory=pin_memory)
        dev_loader = accelerator.prepare(dev_loader)

        dev_metrics, dev_samples = evaluate_qa_detailed(
            model, dev_loader, tokenizer, accelerator,
            max_new_tokens=max_ans_tokens, source_label="dev")

        mr.dev_loss = dev_metrics["loss"]
        mr.dev_ppl = dev_metrics["perplexity"]
        mr.dev_em = dev_metrics["exact_match"]
        mr.dev_f1 = dev_metrics["f1"]
        mr.dev_rouge_l = dev_metrics["rouge_l"]
        mr.dev_n = dev_metrics["n_samples"]
        mr.samples.extend(dev_samples)

        if is_main:
            print(f"  {GREEN}Dev results:{RESET}  "
                  f"Loss={mr.dev_loss:.4f}  PPL={mr.dev_ppl:.2f}  "
                  f"EM={mr.dev_em:.2%}  F1={mr.dev_f1:.4f}  "
                  f"ROUGE-L={mr.dev_rouge_l:.4f}")

        # ── Train peek evaluation ─────────────────────────────────────
        if args.train_peek_samples > 0 and os.path.exists(args.train_data):
            if is_main:
                print(f"  Loading train peek: {args.train_data}")
            train_ds = QADataset(
                args.train_data, tokenizer,
                max_doc_tokens=config.qwen.max_doc_tokens,
                max_question_tokens=config.qwen.max_question_tokens,
                max_answer_tokens=config.qwen.max_answer_tokens,
            )
            if args.train_peek_samples < len(train_ds):
                train_ds = Subset(train_ds, list(range(args.train_peek_samples)))
            if is_main:
                print(f"  Train peek samples: {len(train_ds)}")

            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=False,
                                      collate_fn=collator, num_workers=num_workers,
                                      pin_memory=pin_memory)
            train_loader = accelerator.prepare(train_loader)

            train_metrics, train_samples = evaluate_qa_detailed(
                model, train_loader, tokenizer, accelerator,
                max_new_tokens=max_ans_tokens, source_label="train")

            mr.train_loss = train_metrics["loss"]
            mr.train_ppl = train_metrics["perplexity"]
            mr.train_em = train_metrics["exact_match"]
            mr.train_f1 = train_metrics["f1"]
            mr.train_rouge_l = train_metrics["rouge_l"]
            mr.train_n = train_metrics["n_samples"]
            mr.samples.extend(train_samples)

            if is_main:
                print(f"  {YELLOW}Train results:{RESET} "
                      f"Loss={mr.train_loss:.4f}  PPL={mr.train_ppl:.2f}  "
                      f"EM={mr.train_em:.2%}  F1={mr.train_f1:.4f}  "
                      f"ROUGE-L={mr.train_rouge_l:.4f}")

        all_results.append(mr)

        # Free model memory before loading next
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Print all outputs (main process only) ─────────────────────────────
    if not is_main:
        return

    print("\n" * 2)
    print("━" * W)
    print(f"{'':>{W//2-10}}{BOLD}EVALUATION COMPLETE{RESET}")
    print("━" * W)

    # 1. Comparison table
    print_comparison_table(all_results)

    # 2. Bar charts
    print_bar_chart(all_results)

    # 3. Per-model samples (best/worst)
    for r in sorted(all_results, key=lambda x: x.dev_f1, reverse=True):
        print_model_samples(r, top_n=args.top_n)

    # 4. Final ranking
    print_summary_ranking(all_results)

    # 5. CSV export
    if args.output:
        save_csv(all_results, args.output)


if __name__ == "__main__":
    main()
