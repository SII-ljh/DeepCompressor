"""Tests for config.py."""

import pytest

from deep_compressor.config import (
    DeepCompressorConfig,
    FinBERTConfig,
    LossConfig,
    PerceiverConfig,
    ProjectionConfig,
    QwenConfig,
    TrainingConfig,
)


def test_default_config():
    cfg = DeepCompressorConfig()
    assert cfg.qwen.hidden_size == 1024
    assert cfg.perceiver.perceiver_dim == 1024
    assert cfg.finbert.enabled is False
    assert cfg.training.stage == 1


def test_tiny_config(tiny_config):
    assert tiny_config.qwen.hidden_size == 64
    assert tiny_config.perceiver.num_heads * tiny_config.perceiver.head_dim == tiny_config.perceiver.perceiver_dim


def test_dim_consistency_check():
    with pytest.raises(ValueError, match="must equal perceiver_dim"):
        DeepCompressorConfig(
            perceiver=PerceiverConfig(perceiver_dim=512, num_heads=8, head_dim=32),
        )


def test_hidden_layer_validation():
    with pytest.raises(ValueError, match="hidden_distill_layer"):
        DeepCompressorConfig(
            qwen=QwenConfig(num_hidden_layers=4),
            loss=LossConfig(hidden_distill_layers=[5]),
        )


def test_nested_configs():
    cfg = DeepCompressorConfig()
    assert isinstance(cfg.qwen, QwenConfig)
    assert isinstance(cfg.finbert, FinBERTConfig)
    assert isinstance(cfg.perceiver, PerceiverConfig)
    assert isinstance(cfg.projection, ProjectionConfig)
    assert isinstance(cfg.loss, LossConfig)
    assert isinstance(cfg.training, TrainingConfig)


def test_from_yaml(tmp_path):
    yaml_content = """
qwen:
  hidden_size: 128
  num_hidden_layers: 4
perceiver:
  perceiver_dim: 64
  num_heads: 8
  head_dim: 8
loss:
  hidden_distill_layers: [1, 3]
"""
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text(yaml_content)
    cfg = DeepCompressorConfig.from_yaml(str(yaml_file))
    assert cfg.qwen.hidden_size == 128
    assert cfg.perceiver.perceiver_dim == 64
