"""Tests for AnchorAlign module."""

import torch

from deep_compressor.modules.anchor_align import AnchorAlign


def test_output_shape(tiny_config_finbert, batch_size):
    cfg = tiny_config_finbert
    module = AnchorAlign(cfg.finbert.hidden_size, cfg.perceiver.perceiver_dim,
                         cfg.finbert.anchor_align_layers)
    x = torch.randn(batch_size, cfg.finbert.top_k_anchors, cfg.finbert.hidden_size)
    out = module(x)
    assert out.shape == (batch_size, cfg.finbert.top_k_anchors, cfg.perceiver.perceiver_dim)


def test_gradient_flow(tiny_config_finbert):
    cfg = tiny_config_finbert
    module = AnchorAlign(cfg.finbert.hidden_size, cfg.perceiver.perceiver_dim)
    x = torch.randn(1, 4, cfg.finbert.hidden_size, requires_grad=True)
    out = module(x)
    out.sum().backward()
    assert x.grad is not None
