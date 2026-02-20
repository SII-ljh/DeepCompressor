"""Tests for tokenizer_align module."""

import torch

from deep_compressor.modules.tokenizer_align import align_scores, batch_align_scores


def test_exact_overlap():
    """When spans are identical, scores should transfer exactly."""
    finbert_scores = torch.tensor([0.8, 0.2, 0.9])
    finbert_offsets = [(0, 3), (3, 6), (6, 9)]
    qwen_offsets = [(0, 3), (3, 6), (6, 9)]
    result = align_scores(finbert_scores, finbert_offsets, qwen_offsets, qwen_seq_len=3)
    assert torch.allclose(result, finbert_scores)


def test_partial_overlap_max():
    """Qwen token spanning two FinBERT tokens should get the max score."""
    finbert_scores = torch.tensor([0.3, 0.9])
    finbert_offsets = [(0, 2), (2, 4)]
    qwen_offsets = [(0, 4)]  # single Qwen token covers both FinBERT tokens
    result = align_scores(finbert_scores, finbert_offsets, qwen_offsets, qwen_seq_len=1)
    assert result[0].item() == pytest.approx(0.9)


def test_no_overlap_gives_zero():
    """Qwen token with no overlapping FinBERT token should get 0."""
    finbert_scores = torch.tensor([0.5])
    finbert_offsets = [(0, 3)]
    qwen_offsets = [(5, 8)]  # no overlap
    result = align_scores(finbert_scores, finbert_offsets, qwen_offsets, qwen_seq_len=1)
    assert result[0].item() == 0.0


def test_batch_version():
    finbert_scores = torch.tensor([[0.5, 0.7], [0.1, 0.9]])
    fb_offsets = [[(0, 2), (2, 4)], [(0, 3), (3, 6)]]
    qw_offsets = [[(0, 2), (2, 4)], [(0, 3), (3, 6)]]
    result = batch_align_scores(finbert_scores, fb_offsets, qw_offsets, qwen_seq_len=2)
    assert result.shape == (2, 2)
    assert torch.allclose(result[0], finbert_scores[0])


# Need pytest for pytest.approx
import pytest
