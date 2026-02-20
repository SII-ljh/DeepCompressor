"""Tests for FactDecodeHead module."""

import torch

from deep_compressor.modules.fact_decode_head import FactDecodeHead


def test_output_is_scalar(tiny_config_finbert, batch_size):
    cfg = tiny_config_finbert
    module = FactDecodeHead(cfg.perceiver.perceiver_dim, cfg.finbert.hidden_size)
    processed = torch.randn(batch_size, cfg.finbert.top_k_anchors, cfg.perceiver.perceiver_dim)
    original = torch.randn(batch_size, cfg.finbert.top_k_anchors, cfg.finbert.hidden_size)
    loss = module(processed, original)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_gradient_flow(tiny_config_finbert):
    cfg = tiny_config_finbert
    module = FactDecodeHead(cfg.perceiver.perceiver_dim, cfg.finbert.hidden_size)
    processed = torch.randn(1, 4, cfg.perceiver.perceiver_dim, requires_grad=True)
    original = torch.randn(1, 4, cfg.finbert.hidden_size)
    loss = module(processed, original)
    loss.backward()
    assert processed.grad is not None
