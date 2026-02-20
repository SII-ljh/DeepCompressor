"""Tests for QueryInit module."""

import torch

from deep_compressor.modules.query_init import QueryInit


def test_output_shape(tiny_config, batch_size):
    cfg = tiny_config
    module = QueryInit(cfg.perceiver.num_queries, cfg.perceiver.perceiver_dim, cfg.qwen.hidden_size)
    q_pooled = torch.randn(batch_size, cfg.qwen.hidden_size)
    out = module(q_pooled)
    assert out.shape == (batch_size, cfg.perceiver.num_queries, cfg.perceiver.perceiver_dim)


def test_zero_question_gives_base_queries(tiny_config):
    cfg = tiny_config
    module = QueryInit(cfg.perceiver.num_queries, cfg.perceiver.perceiver_dim, cfg.qwen.hidden_size)
    zero = torch.zeros(1, cfg.qwen.hidden_size)
    out = module(zero)
    # With zero input, bias is just Linear(0) which is the bias term of the linear layer
    # Result should be base_queries + bias_of_linear_layer (broadcast)
    assert out.shape == (1, cfg.perceiver.num_queries, cfg.perceiver.perceiver_dim)


def test_different_questions_give_different_queries(tiny_config):
    cfg = tiny_config
    module = QueryInit(cfg.perceiver.num_queries, cfg.perceiver.perceiver_dim, cfg.qwen.hidden_size)
    q1 = torch.randn(1, cfg.qwen.hidden_size)
    q2 = torch.randn(1, cfg.qwen.hidden_size)
    out1 = module(q1)
    out2 = module(q2)
    assert not torch.allclose(out1, out2)
