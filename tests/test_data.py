"""Tests for data.py datasets and collator."""

import json

import torch

from deep_compressor.data import PaddingCollator, QADataset


class _FakeTokenizer:
    """Minimal tokenizer mock for testing."""

    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.eos_token_id = 2

    def __call__(self, text, truncation=True, max_length=512, return_tensors="pt", padding=False):
        # Deterministic: each char becomes a token id
        ids = [ord(c) % self.vocab_size for c in text[:max_length]]
        if not ids:
            ids = [1]
        t = torch.tensor([ids])
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}


def test_qa_dataset(tmp_path):
    data_file = tmp_path / "qa.json"
    data = [{"context": "The revenue was $10M.", "question": "What was the revenue?",
             "answer": "$10M"}]
    with open(data_file, "w") as f:
        json.dump(data, f)

    tokenizer = _FakeTokenizer()
    ds = QADataset(str(data_file), tokenizer)
    assert len(ds) == 1
    sample = ds[0]
    assert "doc_input_ids" in sample
    assert "q_input_ids" in sample
    assert "answer_ids" in sample


def test_collator():
    collator = PaddingCollator(pad_token_id=0)
    batch = [
        {"doc_input_ids": torch.tensor([1, 2, 3]), "segment_ids": torch.tensor([4, 5])},
        {"doc_input_ids": torch.tensor([6, 7]), "segment_ids": torch.tensor([8, 9, 10])},
    ]
    result = collator(batch)
    assert result["doc_input_ids"].shape == (2, 3)
    assert result["segment_ids"].shape == (2, 3)
    assert "doc_attention_mask" in result or "doc_input_attention_mask" in result
