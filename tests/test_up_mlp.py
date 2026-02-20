"""Tests for UpMLP module."""

import torch

from deep_compressor.modules.up_mlp import UpMLP


def test_output_shape(tiny_config, batch_size):
    cfg = tiny_config
    module = UpMLP(cfg.perceiver.perceiver_dim, cfg.qwen.hidden_size,
                   cfg.projection.up_hidden, cfg.projection.dropout)
    x = torch.randn(batch_size, cfg.perceiver.num_queries, cfg.perceiver.perceiver_dim)
    out = module(x)
    assert out.shape == (batch_size, cfg.perceiver.num_queries, cfg.qwen.hidden_size)


def test_gradient_flow(tiny_config):
    cfg = tiny_config
    module = UpMLP(cfg.perceiver.perceiver_dim, cfg.qwen.hidden_size,
                   cfg.projection.up_hidden)
    x = torch.randn(2, 8, cfg.perceiver.perceiver_dim, requires_grad=True)
    out = module(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
