"""Tests for config.py."""

import pytest

from deep_compressor.config import (
    DeepCompressorConfig,
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


def test_tiny_config(tiny_config):
    assert tiny_config.qwen.hidden_size == 64
    assert tiny_config.perceiver.num_heads * tiny_config.perceiver.head_dim == tiny_config.perceiver.perceiver_dim


def test_dim_consistency_check():
    """Non-registry model: num_heads * head_dim must equal perceiver_dim."""
    with pytest.raises(ValueError, match="must equal perceiver_dim"):
        DeepCompressorConfig(
            qwen=QwenConfig(model_name_or_path="tiny", hidden_size=64, num_hidden_layers=4),
            perceiver=PerceiverConfig(perceiver_dim=512, num_heads=8, head_dim=32),
        )


def test_hidden_layer_validation():
    with pytest.raises(ValueError, match="hidden_distill_layer"):
        DeepCompressorConfig(
            qwen=QwenConfig(model_name_or_path="tiny", hidden_size=64, num_hidden_layers=4),
            perceiver=PerceiverConfig(perceiver_dim=64, num_heads=4, head_dim=16),
            loss=LossConfig(hidden_distill_layers=[5]),
        )


def test_nested_configs():
    cfg = DeepCompressorConfig()
    assert isinstance(cfg.qwen, QwenConfig)
    assert isinstance(cfg.perceiver, PerceiverConfig)
    assert isinstance(cfg.projection, ProjectionConfig)
    assert isinstance(cfg.loss, LossConfig)
    assert isinstance(cfg.training, TrainingConfig)


def test_qwen3_registry_auto_sync():
    """Registry models auto-resolve hidden_size and sync perceiver_dim."""
    # Qwen3-4B: hidden_size=2560, num_hidden_layers=36
    cfg = DeepCompressorConfig(
        qwen=QwenConfig(model_name_or_path="Qwen/Qwen3-4B"),
        perceiver=PerceiverConfig(num_heads=16),
    )
    assert cfg.qwen.hidden_size == 2560
    assert cfg.qwen.num_hidden_layers == 36
    assert cfg.perceiver.perceiver_dim == 2560
    assert cfg.perceiver.head_dim == 160  # 2560 / 16

    # Local path should also match
    cfg2 = DeepCompressorConfig(
        qwen=QwenConfig(model_name_or_path="models/Qwen3-8B"),
        perceiver=PerceiverConfig(num_heads=16),
    )
    assert cfg2.qwen.hidden_size == 4096
    assert cfg2.perceiver.perceiver_dim == 4096
    assert cfg2.perceiver.head_dim == 256  # 4096 / 16


def test_qwen3_registry_bad_num_heads():
    """Registry model with indivisible num_heads should raise."""
    with pytest.raises(ValueError, match="must be divisible"):
        DeepCompressorConfig(
            qwen=QwenConfig(model_name_or_path="Qwen/Qwen3-4B"),
            perceiver=PerceiverConfig(num_heads=7),  # 2560 % 7 != 0
        )


def test_training_config_new_defaults():
    """New training config fields have correct defaults."""
    cfg = DeepCompressorConfig()
    assert cfg.training.epochs == 0
    assert cfg.training.auto_batch_size is False
    assert cfg.training.target_effective_batch_size == 256


def test_epochs_validation():
    """epochs must be >= 0."""
    with pytest.raises(ValueError, match="epochs must be >= 0"):
        DeepCompressorConfig(
            qwen=QwenConfig(model_name_or_path="tiny", hidden_size=64, num_hidden_layers=4),
            perceiver=PerceiverConfig(perceiver_dim=64, num_heads=4, head_dim=16),
            loss=LossConfig(hidden_distill_layers=[1, 3]),
            training=TrainingConfig(epochs=-1),
        )


def test_target_effective_batch_size_validation():
    """target_effective_batch_size must be >= 1."""
    with pytest.raises(ValueError, match="target_effective_batch_size must be >= 1"):
        DeepCompressorConfig(
            qwen=QwenConfig(model_name_or_path="tiny", hidden_size=64, num_hidden_layers=4),
            perceiver=PerceiverConfig(perceiver_dim=64, num_heads=4, head_dim=16),
            loss=LossConfig(hidden_distill_layers=[1, 3]),
            training=TrainingConfig(target_effective_batch_size=0),
        )


def test_from_yaml(tmp_path):
    yaml_content = """
qwen:
  model_name_or_path: "tiny"
  hidden_size: 128
  num_hidden_layers: 4
perceiver:
  perceiver_dim: 128
  num_heads: 8
  head_dim: 16
loss:
  hidden_distill_layers: [1, 3]
"""
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text(yaml_content)
    cfg = DeepCompressorConfig.from_yaml(str(yaml_file))
    assert cfg.qwen.hidden_size == 128
    assert cfg.perceiver.perceiver_dim == 128
