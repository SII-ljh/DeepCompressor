"""Integration tests with real Qwen3-0.6B model. Run with --runslow."""

import pytest
import torch

from deep_compressor.config import DeepCompressorConfig
from deep_compressor.model import DeepCompressor


@pytest.mark.slow
def test_real_model_forward_ntp():
    """End-to-end forward pass with real Qwen3-0.6B."""
    config = DeepCompressorConfig()  # uses default Qwen3-0.6B
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    model = DeepCompressor(config).to(device)
    model.eval()

    B, doc_len, seg_len = 1, 64, 16
    doc_ids = torch.randint(0, config.qwen.vocab_size, (B, doc_len), device=device)
    doc_mask = torch.ones(B, doc_len, dtype=torch.long, device=device)
    seg_ids = torch.randint(0, config.qwen.vocab_size, (B, seg_len), device=device)
    seg_mask = torch.ones(B, seg_len, dtype=torch.long, device=device)
    seg_labels = torch.randint(0, config.qwen.vocab_size, (B, seg_len), device=device)

    with torch.no_grad():
        losses = model.forward_ntp(doc_ids, doc_mask, seg_ids, seg_mask, seg_labels)

    assert torch.isfinite(losses["total"])
    assert losses["total"].item() > 0
    print(f"NTP loss: {losses['total'].item():.4f}")


@pytest.mark.slow
def test_real_model_forward_qa():
    """End-to-end QA forward pass with real Qwen3-0.6B."""
    config = DeepCompressorConfig()
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    model = DeepCompressor(config).to(device)
    model.eval()

    B, doc_len, q_len, a_len = 1, 64, 16, 8
    V = config.qwen.vocab_size

    with torch.no_grad():
        losses = model.forward_qa(
            doc_input_ids=torch.randint(0, V, (B, doc_len), device=device),
            doc_attention_mask=torch.ones(B, doc_len, dtype=torch.long, device=device),
            q_input_ids=torch.randint(0, V, (B, q_len), device=device),
            q_attention_mask=torch.ones(B, q_len, dtype=torch.long, device=device),
            answer_ids=torch.randint(0, V, (B, a_len), device=device),
            answer_attention_mask=torch.ones(B, a_len, dtype=torch.long, device=device),
            answer_labels=torch.randint(0, V, (B, a_len), device=device),
        )

    assert torch.isfinite(losses["total"])
    assert "qa_ce" in losses
    print(f"QA CE loss: {losses['qa_ce'].item():.4f}")


@pytest.mark.slow
def test_real_model_backward():
    """Verify gradients flow through trainable modules with real Qwen."""
    config = DeepCompressorConfig()
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    model = DeepCompressor(config).to(device)
    model.train()

    B, doc_len, seg_len = 1, 32, 8
    V = config.qwen.vocab_size

    losses = model.forward_ntp(
        doc_input_ids=torch.randint(0, V, (B, doc_len), device=device),
        doc_attention_mask=torch.ones(B, doc_len, dtype=torch.long, device=device),
        segment_ids=torch.randint(0, V, (B, seg_len), device=device),
        segment_attention_mask=torch.ones(B, seg_len, dtype=torch.long, device=device),
        segment_labels=torch.randint(0, V, (B, seg_len), device=device),
    )
    losses["total"].backward()

    trainable_with_grad = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0:
            trainable_with_grad.append(name)

    assert len(trainable_with_grad) > 0, "No trainable parameters received gradients"
    print(f"Params with grad: {len(trainable_with_grad)}")
