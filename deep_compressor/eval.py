"""Evaluation utilities for Deep Compressor.

Provides:
  - Text normalization and QA metrics (Exact Match, F1)
  - NTP evaluation (perplexity on validation set)
  - QA evaluation (EM + F1 via greedy generation)
"""

import re
import string
from typing import Dict

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader


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


@torch.no_grad()
def evaluate_ntp(model, eval_loader: DataLoader,
                 accelerator: Accelerator) -> Dict[str, float]:
    """Evaluate NTP model on validation set, returning perplexity and loss.

    Args:
        model: DeepCompressor (already wrapped by accelerator)
        eval_loader: DataLoader for NTP validation data (already prepared)
        accelerator: Accelerator instance
    Returns:
        {"perplexity": float, "loss": float}
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for batch in eval_loader:
        losses = model(
            mode="ntp",
            doc_input_ids=batch["doc_input_ids"],
            doc_attention_mask=batch["doc_attention_mask"],
            segment_ids=batch["segment_ids"],
            segment_attention_mask=batch["segment_attention_mask"],
            segment_labels=batch["segment_labels"],
        )
        loss_val = losses["total"].detach()
        bs = batch["doc_input_ids"].shape[0]

        total_loss += loss_val * bs
        total_samples += bs

    # Gather across processes
    stats = torch.tensor([total_loss, float(total_samples)],
                         device=accelerator.device)
    stats = accelerator.gather(stats)
    # stats is (num_processes, 2) or (2,) — sum across all processes
    if stats.dim() > 1:
        stats = stats.sum(dim=0)
    gathered_loss = stats[0].item()
    gathered_samples = stats[1].item()

    avg_loss = gathered_loss / max(gathered_samples, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {"perplexity": perplexity, "loss": avg_loss}


@torch.no_grad()
def evaluate_qa(model, eval_loader: DataLoader, tokenizer,
                accelerator: Accelerator,
                max_new_tokens: int = 64) -> Dict[str, float]:
    """Evaluate QA model on dev set, returning Exact Match and F1.

    Args:
        model: DeepCompressor (already wrapped by accelerator)
        eval_loader: DataLoader for QA dev data (already prepared)
        tokenizer: Qwen tokenizer for decoding generated tokens
        accelerator: Accelerator instance
        max_new_tokens: max tokens to generate per answer
    Returns:
        {"exact_match": float, "f1": float}
    """
    model.eval()
    unwrapped = accelerator.unwrap_model(model)

    all_em = []
    all_f1 = []

    for batch in eval_loader:
        # Encode document and compress to prefix
        byte_array = unwrapped.encode_document(
            batch["doc_input_ids"], batch["doc_attention_mask"])
        queries = unwrapped.encode_question(
            batch["q_input_ids"], batch["q_attention_mask"])
        latent = unwrapped.compress(
            queries, byte_array, byte_mask=batch["doc_attention_mask"])
        prefix_embeds = unwrapped.up_mlp(latent)

        # Generate answer tokens
        gen_ids = unwrapped.generate_answer(
            prefix_embeds, batch["q_input_ids"], batch["q_attention_mask"],
            tokenizer=tokenizer, max_new_tokens=max_new_tokens,
        )

        # Decode predictions
        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        golds = batch["answer_text"]  # list of str

        for pred, gold in zip(preds, golds):
            all_em.append(compute_exact_match(pred, gold))
            all_f1.append(compute_f1(pred, gold))

    # Gather across processes
    local_em = torch.tensor(all_em, device=accelerator.device) if all_em else torch.zeros(1, device=accelerator.device)
    local_f1 = torch.tensor(all_f1, device=accelerator.device) if all_f1 else torch.zeros(1, device=accelerator.device)
    local_count = torch.tensor([float(len(all_em))], device=accelerator.device)

    gathered_count = accelerator.gather(local_count)
    total_count = gathered_count.sum().item()

    if total_count == 0:
        return {"exact_match": 0.0, "f1": 0.0}

    local_em_sum = torch.tensor([sum(all_em)], device=accelerator.device)
    local_f1_sum = torch.tensor([sum(all_f1)], device=accelerator.device)
    gathered_em = accelerator.gather(local_em_sum).sum().item()
    gathered_f1 = accelerator.gather(local_f1_sum).sum().item()

    return {
        "exact_match": gathered_em / total_count,
        "f1": gathered_f1 / total_count,
    }
