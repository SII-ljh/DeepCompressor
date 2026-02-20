"""Tests for DownProj module."""

import torch

from deep_compressor.modules.down_proj import DownProj


def test_output_shape(tiny_config, batch_size, doc_len):
    cfg = tiny_config
    module = DownProj(cfg.qwen.hidden_size, cfg.perceiver.perceiver_dim,
                      cfg.projection.down_hidden, cfg.projection.dropout)
    x = torch.randn(batch_size, doc_len, cfg.qwen.hidden_size)
    out = module(x)
    assert out.shape == (batch_size, doc_len, cfg.perceiver.perceiver_dim)


def test_gradient_flow(tiny_config):
    cfg = tiny_config
    module = DownProj(cfg.qwen.hidden_size, cfg.perceiver.perceiver_dim,
                      cfg.projection.down_hidden)
    x = torch.randn(2, 8, cfg.qwen.hidden_size, requires_grad=True)
    out = module(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
