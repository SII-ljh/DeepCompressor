"""Tests for GuidedPerceiver module."""

import torch

from deep_compressor.modules.perceiver import GuidedPerceiver


def _make_perceiver(cfg):
    pcfg = cfg.perceiver
    return GuidedPerceiver(
        dim=pcfg.perceiver_dim, num_heads=pcfg.num_heads, head_dim=pcfg.head_dim,
        ff_mult=pcfg.ff_mult, dropout=pcfg.dropout,
        stage_a_cross_layers=pcfg.stage_a_cross_layers,
        stage_a_self_layers=pcfg.stage_a_self_layers,
        stage_b_layers=pcfg.stage_b_layers,
        stage_c_cross_layers=pcfg.stage_c_cross_layers,
        stage_c_self_layers=pcfg.stage_c_self_layers,
    )


def test_output_shape(tiny_config, batch_size, doc_len):
    cfg = tiny_config
    perceiver = _make_perceiver(cfg)
    queries = torch.randn(batch_size, cfg.perceiver.num_queries, cfg.perceiver.perceiver_dim)
    byte_array = torch.randn(batch_size, doc_len, cfg.perceiver.perceiver_dim)
    byte_mask = torch.ones(batch_size, doc_len, dtype=torch.bool)

    out = perceiver(queries, byte_array, byte_mask=byte_mask)
    assert out.shape == (batch_size, cfg.perceiver.num_queries, cfg.perceiver.perceiver_dim)


def test_gradient_flow(tiny_config, batch_size, doc_len):
    cfg = tiny_config
    perceiver = _make_perceiver(cfg)
    queries = torch.randn(batch_size, cfg.perceiver.num_queries, cfg.perceiver.perceiver_dim, requires_grad=True)
    byte_array = torch.randn(batch_size, doc_len, cfg.perceiver.perceiver_dim)
    out = perceiver(queries, byte_array)
    out.sum().backward()
    assert queries.grad is not None


def test_variable_length_mask(tiny_config):
    cfg = tiny_config
    perceiver = _make_perceiver(cfg)
    B = 2
    queries = torch.randn(B, cfg.perceiver.num_queries, cfg.perceiver.perceiver_dim)

    # Different sequence lengths
    byte_array = torch.randn(B, 20, cfg.perceiver.perceiver_dim)
    mask = torch.ones(B, 20, dtype=torch.bool)
    mask[1, 10:] = False  # second sample shorter

    out = perceiver(queries, byte_array, byte_mask=mask)
    assert out.shape == (B, cfg.perceiver.num_queries, cfg.perceiver.perceiver_dim)
