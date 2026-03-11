"""Evaluation utilities for Deep Compressor.

Provides:
  - Text normalization and QA metrics (Exact Match, F1, ROUGE-L)
  - Multi-reference answer evaluation
  - QA evaluation (EM + F1 via greedy generation)
  - Latency measurement utilities
"""

import logging
import re
import string
import time
from typing import Callable, Dict, List

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def normalize_text(s: str) -> str:
    """Lowercase, remove punctuation, strip whitespace. Works for Chinese + English."""
    s = s.lower()
    # Remove punctuation (ASCII + common Chinese punctuation)
    s = re.sub(r"[%s]" % re.escape(string.punctuation), "", s)
    s = re.sub(r"[\u3000-\u303f\uff00-\uffef]", "", s)  # CJK punctuation + fullwidth forms
    s = s.strip()
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def _tokenize_for_f1(text: str):
    """Split text into tokens for F1 calculation.

    For Chinese text (characters without spaces), split per-character.
    For English text, split on whitespace.
    """
    tokens = []
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            # Each CJK character is its own token; add surrounding spaces
            tokens.append(f" {char} ")
        elif char.strip():
            tokens.append(char)
        else:
            tokens.append(" ")
    return "".join(tokens).split()


def compute_exact_match(pred: str, gold: str) -> float:
    """Return 1.0 if normalized pred == normalized gold, else 0.0."""
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0


def compute_f1(pred: str, gold: str) -> float:
    """Token-level F1 between normalized pred and gold.

    Handles both Chinese (character-level) and English (word-level) text.
    """
    pred_tokens = _tokenize_for_f1(normalize_text(pred))
    gold_tokens = _tokenize_for_f1(normalize_text(gold))

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = set(pred_tokens) & set(gold_tokens)
    # Count with multiplicity
    num_common = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common)

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list, b: list) -> int:
    """Compute length of longest common subsequence via DP."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Space-optimised: only keep two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def compute_rouge_l(pred: str, gold: str) -> float:
    """ROUGE-L F1 based on longest common subsequence.

    Reuses _tokenize_for_f1 and normalize_text for consistent tokenisation.
    """
    pred_tokens = _tokenize_for_f1(normalize_text(pred))
    gold_tokens = _tokenize_for_f1(normalize_text(gold))

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    lcs_len = _lcs_length(pred_tokens, gold_tokens)
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / len(pred_tokens)
    recall = lcs_len / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_best_metric(pred: str, golds: List[str],
                        metric_fn: Callable[[str, str], float]) -> float:
    """Return the highest metric score across multiple reference answers."""
    if not golds:
        return 0.0
    return max(metric_fn(pred, g) for g in golds)


class LatencyTimer:
    """Context manager for timing code sections.

    Uses CUDA events on CUDA devices for accurate GPU timing,
    falls back to time.perf_counter otherwise.
    """

    def __init__(self, device: torch.device = None):
        self.use_cuda = (device is not None and device.type == "cuda"
                         and torch.cuda.is_available())
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        if self.use_cuda:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.use_cuda:
            self._end_event.record()
            torch.cuda.synchronize()
            self.elapsed_ms = self._start_event.elapsed_time(self._end_event)
        else:
            self.elapsed_ms = (time.perf_counter() - self._start_time) * 1000.0


@torch.no_grad()
def evaluate_qa_multi_ref(model, eval_loader: DataLoader, tokenizer,
                          device: torch.device,
                          max_new_tokens: int = 64) -> Dict[str, float]:
    """Single-GPU QA evaluation with multi-reference answer support.

    Expects batches to contain 'all_answers' (list of list of str).
    Falls back to 'answer_text' (list of str) if 'all_answers' is absent.

    Returns:
        {"exact_match": float, "f1": float, "rouge_l": float, "n_samples": int}
    """
    model.eval()
    all_em, all_f1, all_rouge = [], [], []

    for batch in eval_loader:
        doc_ids = batch["doc_input_ids"].to(device)
        doc_mask = batch["doc_attention_mask"].to(device)
        q_ids = batch["q_input_ids"].to(device)
        q_mask = batch["q_attention_mask"].to(device)

        byte_array = model.encode_document(doc_ids, doc_mask)
        queries = model.encode_question(q_ids, q_mask)
        latent = model.compress(queries, byte_array, byte_mask=doc_mask)
        prefix = model.up_mlp(latent)

        gen_ids = model.generate_answer(
            prefix, q_ids, q_mask,
            tokenizer=tokenizer, max_new_tokens=max_new_tokens)

        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        # Multi-reference or single-reference
        if "all_answers" in batch:
            answers_list = batch["all_answers"]  # list of list of str
        else:
            answers_list = [[g] for g in batch["answer_text"]]

        for pred, golds in zip(preds, answers_list):
            all_em.append(compute_best_metric(pred, golds, compute_exact_match))
            all_f1.append(compute_best_metric(pred, golds, compute_f1))
            all_rouge.append(compute_best_metric(pred, golds, compute_rouge_l))

    n = max(len(all_em), 1)
    return {
        "exact_match": sum(all_em) / n,
        "f1": sum(all_f1) / n,
        "rouge_l": sum(all_rouge) / n,
        "n_samples": len(all_em),
    }


@torch.no_grad()
def evaluate_qa(model, eval_loader: DataLoader, tokenizer,
                accelerator: Accelerator,
                max_new_tokens: int = 64,
                show_samples: int = 5) -> Dict[str, float]:
    """Evaluate QA model on dev set, returning Exact Match, F1, loss, and perplexity.

    Args:
        model: DeepCompressor (already wrapped by accelerator)
        eval_loader: DataLoader for QA dev data (already prepared)
        tokenizer: Qwen tokenizer for decoding generated tokens
        accelerator: Accelerator instance
        max_new_tokens: max tokens to generate per answer
        show_samples: number of sample predictions to display (0 = none)
    Returns:
        {"exact_match": float, "f1": float, "loss": float, "perplexity": float}
    """
    model.eval()
    unwrapped = accelerator.unwrap_model(model)

    all_em = []
    all_f1 = []
    all_loss = []  # Track loss per sample
    sample_outputs = []  # Store samples for display

    total_batches = len(eval_loader)
    for batch_idx, batch in enumerate(eval_loader):
        # Log progress every 10 batches (only on main process)
        if accelerator.is_main_process and batch_idx % 10 == 0:
            print(f"  [EVAL] Processing batch {batch_idx}/{total_batches} "
                  f"({100*batch_idx/total_batches:.1f}%)", flush=True)
        # Encode document and compress to prefix
        byte_array = unwrapped.encode_document(
            batch["doc_input_ids"], batch["doc_attention_mask"])
        queries = unwrapped.encode_question(
            batch["q_input_ids"], batch["q_attention_mask"])
        latent = unwrapped.compress(
            queries, byte_array, byte_mask=batch["doc_attention_mask"])
        prefix_embeds = unwrapped.up_mlp(latent)

        # Compute loss (teacher-forcing with gold answers)
        suffix_ids = torch.cat([batch["q_input_ids"], batch["answer_ids"]], dim=1)
        suffix_mask = torch.cat([batch["q_attention_mask"],
                                torch.ones_like(batch["answer_ids"])], dim=1)
        q_len = batch["q_input_ids"].shape[1]
        q_labels = torch.full_like(batch["q_input_ids"], -100)
        full_labels = torch.cat([q_labels, batch["answer_labels"]], dim=1)

        outputs = unwrapped.decode(
            prefix_embeds, suffix_ids, suffix_mask, labels=full_labels)
        batch_loss = outputs.loss.detach()
        all_loss.append(batch_loss.item())

        # Generate answer tokens
        gen_ids = unwrapped.generate_answer(
            prefix_embeds, batch["q_input_ids"], batch["q_attention_mask"],
            tokenizer=tokenizer, max_new_tokens=max_new_tokens,
        )

        # Decode predictions
        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        golds = batch["answer_text"]  # list of str

        # Store samples for display (only from main process, only first few)
        if accelerator.is_main_process and len(sample_outputs) < show_samples and "question_text" in batch:
            questions = batch.get("question_text", [""] * len(preds))
            for q, pred, gold in zip(questions, preds, golds):
                if len(sample_outputs) < show_samples:
                    em = compute_exact_match(pred, gold)
                    f1 = compute_f1(pred, gold)
                    sample_outputs.append({
                        "question": q,
                        "prediction": pred,
                        "gold": gold,
                        "em": em,
                        "f1": f1,
                    })

        for pred, gold in zip(preds, golds):
            all_em.append(compute_exact_match(pred, gold))
            all_f1.append(compute_f1(pred, gold))

    # Gather across processes
    local_em = torch.tensor(all_em, device=accelerator.device) if all_em else torch.zeros(1, device=accelerator.device)
    local_f1 = torch.tensor(all_f1, device=accelerator.device) if all_f1 else torch.zeros(1, device=accelerator.device)
    local_loss = torch.tensor(all_loss, device=accelerator.device) if all_loss else torch.zeros(1, device=accelerator.device)
    local_count = torch.tensor([float(len(all_em))], device=accelerator.device)

    gathered_count = accelerator.gather(local_count)
    total_count = gathered_count.sum().item()

    if total_count == 0:
        return {"exact_match": 0.0, "f1": 0.0, "loss": 0.0, "perplexity": float('inf')}

    local_em_sum = torch.tensor([sum(all_em)], device=accelerator.device)
    local_f1_sum = torch.tensor([sum(all_f1)], device=accelerator.device)
    local_loss_sum = torch.tensor([sum(all_loss)], device=accelerator.device)
    gathered_em = accelerator.gather(local_em_sum).sum().item()
    gathered_f1 = accelerator.gather(local_f1_sum).sum().item()
    gathered_loss = accelerator.gather(local_loss_sum).sum().item()

    # FIX: gathered_loss is sum across all processes, divide by total number of batches (not just local)
    total_batches = len(all_loss) * accelerator.num_processes
    avg_loss = gathered_loss / max(total_batches, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    # Display sample predictions (only on main process)
    if accelerator.is_main_process and sample_outputs:
        print("\n" + "="*80)
        print("  Sample Predictions (first {} examples)".format(len(sample_outputs)))
        print("="*80)
        for i, sample in enumerate(sample_outputs, 1):
            print(f"\n[Sample {i}]")
            if sample["question"]:
                print(f"Question:   {sample['question'][:100]}{'...' if len(sample['question']) > 100 else ''}")
            print(f"Prediction: {sample['prediction']}")
            print(f"Gold:       {sample['gold']}")
            print(f"EM: {sample['em']:.0f}  F1: {sample['f1']:.4f}")
            if i < len(sample_outputs):
                print("-" * 80)
        print("="*80 + "\n")

    return {
        "exact_match": gathered_em / total_count,
        "f1": gathered_f1 / total_count,
        "loss": avg_loss,
        "perplexity": perplexity,
    }
