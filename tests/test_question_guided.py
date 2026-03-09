"""Tests for question-guided compression feature."""

import json
import tempfile
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from deep_compressor.config import DeepCompressorConfig
from deep_compressor.data import NTPDataset, PaddingCollator
from deep_compressor.model import DeepCompressor


@pytest.fixture
def tokenizer():
    """Simple tokenizer for testing."""
    # Use a minimal tokenizer (same as in conftest.py)
    from unittest.mock import Mock
    tok = Mock()
    tok.pad_token_id = 0
    tok.eos_token_id = 1

    def encode(text, **kwargs):
        # Simple word-based encoding
        words = text.split()
        ids = [hash(w) % 100 + 2 for w in words]  # Map to token ids 2-101
        result = {"input_ids": ids}
        if kwargs.get("return_tensors") == "pt":
            result["input_ids"] = torch.tensor([ids])
        return result

    tok.side_effect = encode
    tok.__call__ = encode
    return tok


@pytest.fixture
def ntp_data_with_questions():
    """Create temporary NTP data file with questions."""
    data = [
        {"text": "This is document one with some financial data.", "question": "What is the revenue?"},
        {"text": "Document two contains earnings report.", "question": "What are the earnings?"},
        {"text": "Third document has balance sheet.", "question": "What is the total asset?"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink()


def test_ntp_dataset_loads_questions(ntp_data_with_questions, tokenizer):
    """Test that NTPDataset loads questions when use_questions=True."""
    # Without questions
    dataset_no_q = NTPDataset(
        ntp_data_with_questions, tokenizer,
        max_doc_tokens=128, segment_len=32,
        use_questions=False
    )

    sample = dataset_no_q[0]
    assert "doc_input_ids" in sample
    assert "segment_ids" in sample
    assert "q_input_ids" not in sample  # Should NOT have questions

    # With questions
    dataset_with_q = NTPDataset(
        ntp_data_with_questions, tokenizer,
        max_doc_tokens=128, segment_len=32,
        use_questions=True
    )

    sample = dataset_with_q[0]
    assert "doc_input_ids" in sample
    assert "segment_ids" in sample
    assert "q_input_ids" in sample  # Should HAVE questions
    assert isinstance(sample["q_input_ids"], torch.Tensor)
    assert sample["q_input_ids"].ndim == 1  # 1D tensor


def test_collator_handles_questions(ntp_data_with_questions, tokenizer):
    """Test that PaddingCollator generates attention masks for questions."""
    dataset = NTPDataset(
        ntp_data_with_questions, tokenizer,
        max_doc_tokens=128, segment_len=32,
        use_questions=True
    )

    collator = PaddingCollator(pad_token_id=0)
    batch = collator([dataset[0], dataset[1]])

    # Check that question attention mask is generated
    assert "q_input_ids" in batch
    assert "q_attention_mask" in batch
    assert batch["q_input_ids"].shape[0] == 2  # Batch size 2
    assert batch["q_attention_mask"].shape == batch["q_input_ids"].shape


def test_model_accepts_optional_questions(tiny_config):
    """Test that model.forward_ntp accepts optional question parameters."""
    import inspect

    # Check that forward_ntp signature includes optional question parameters
    sig = inspect.signature(DeepCompressor.forward_ntp)
    params = sig.parameters

    # Check required params
    assert "doc_input_ids" in params
    assert "doc_attention_mask" in params
    assert "segment_ids" in params
    assert "segment_attention_mask" in params
    assert "segment_labels" in params

    # Check optional question params
    assert "q_input_ids" in params
    assert "q_attention_mask" in params

    # Verify they are optional (have default values)
    assert params["q_input_ids"].default is not inspect.Parameter.empty
    assert params["q_attention_mask"].default is not inspect.Parameter.empty

    # The defaults should be None
    assert params["q_input_ids"].default is None
    assert params["q_attention_mask"].default is None


def test_config_ntp_use_questions_flag():
    """Test that config has ntp_use_questions flag."""
    config = DeepCompressorConfig()

    # Default should be False (backward compatible)
    assert hasattr(config.ablation, "ntp_use_questions")
    assert config.ablation.ntp_use_questions is False

    # Should be able to set to True
    config.ablation.ntp_use_questions = True
    assert config.ablation.ntp_use_questions is True


def test_backward_compatibility(ntp_data_with_questions, tokenizer):
    """Test that old data format (without questions) still works."""
    # Create data without questions
    data = [
        {"text": "Document one"},
        {"text": "Document two"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    try:
        # Should work even with use_questions=True if data has no questions
        dataset = NTPDataset(
            temp_path, tokenizer,
            max_doc_tokens=128, segment_len=32,
            use_questions=True  # Enabled but data has no questions
        )

        sample = dataset[0]
        assert "doc_input_ids" in sample
        assert "segment_ids" in sample
        assert "q_input_ids" not in sample  # No questions in data

    finally:
        Path(temp_path).unlink()
