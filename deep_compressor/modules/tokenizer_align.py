"""Tokenizer alignment: maps FinBERT token-level scores to Qwen token-level scores
via character-level span overlap (max aggregation)."""

from typing import List, Tuple

import torch


def get_char_spans(offsets: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Normalize offset pairs, filtering out special tokens with (0, 0) spans."""
    return [(s, e) for s, e in offsets if e > s]


def align_scores(
    finbert_scores: torch.Tensor,
    finbert_offsets: List[Tuple[int, int]],
    qwen_offsets: List[Tuple[int, int]],
    qwen_seq_len: int,
) -> torch.Tensor:
    """Map FinBERT per-token scores to Qwen per-token scores using character span overlap.

    Uses max aggregation: each Qwen token gets the max score among overlapping FinBERT tokens.

    Args:
        finbert_scores: (finbert_seq_len,) — per-token entity scores
        finbert_offsets: character offset pairs for each FinBERT token
        qwen_offsets: character offset pairs for each Qwen token
        qwen_seq_len: total Qwen sequence length (for output tensor)
    Returns:
        qwen_scores: (qwen_seq_len,) — aligned entity scores for Qwen tokens
    """
    device = finbert_scores.device
    qwen_scores = torch.zeros(qwen_seq_len, device=device)

    fb_spans = get_char_spans(finbert_offsets)
    qw_spans = get_char_spans(qwen_offsets)

    for qi, (qs, qe) in enumerate(qw_spans):
        if qi >= qwen_seq_len:
            break
        max_score = 0.0
        for fi, (fs, fe) in enumerate(fb_spans):
            if fi >= len(finbert_scores):
                break
            # Check overlap
            overlap_start = max(qs, fs)
            overlap_end = min(qe, fe)
            if overlap_start < overlap_end:
                max_score = max(max_score, finbert_scores[fi].item())
        qwen_scores[qi] = max_score

    return qwen_scores


def batch_align_scores(
    finbert_scores: torch.Tensor,
    finbert_offsets_batch: List[List[Tuple[int, int]]],
    qwen_offsets_batch: List[List[Tuple[int, int]]],
    qwen_seq_len: int,
) -> torch.Tensor:
    """Batch version of align_scores.

    Args:
        finbert_scores: (batch, finbert_seq_len)
        finbert_offsets_batch: list of offset pairs per sample
        qwen_offsets_batch: list of offset pairs per sample
        qwen_seq_len: output sequence length
    Returns:
        (batch, qwen_seq_len)
    """
    batch_size = finbert_scores.shape[0]
    result = torch.zeros(batch_size, qwen_seq_len, device=finbert_scores.device)
    for i in range(batch_size):
        result[i] = align_scores(
            finbert_scores[i],
            finbert_offsets_batch[i],
            qwen_offsets_batch[i],
            qwen_seq_len,
        )
    return result
