"""Tests for NERHead module."""

import torch

from deep_compressor.modules.ner_head import NERHead


def test_output_shape(tiny_config_finbert, batch_size, doc_len):
    cfg = tiny_config_finbert
    module = NERHead(cfg.finbert.hidden_size, cfg.finbert.num_ner_labels)
    x = torch.randn(batch_size, doc_len, cfg.finbert.hidden_size)
    scores = module(x)
    assert scores.shape == (batch_size, doc_len)


def test_output_range(tiny_config_finbert):
    cfg = tiny_config_finbert
    module = NERHead(cfg.finbert.hidden_size, cfg.finbert.num_ner_labels)
    x = torch.randn(2, 10, cfg.finbert.hidden_size)
    scores = module(x)
    assert (scores >= 0).all()
    assert (scores <= 1).all()


def test_gradient_flow(tiny_config_finbert):
    cfg = tiny_config_finbert
    module = NERHead(cfg.finbert.hidden_size, cfg.finbert.num_ner_labels)
    x = torch.randn(1, 8, cfg.finbert.hidden_size, requires_grad=True)
    scores = module(x)
    scores.sum().backward()
    assert x.grad is not None
