"""Tests for wandb naming, tagging, and config building helpers."""

import argparse
import re
from types import SimpleNamespace

import pytest

from deep_compressor.config import (
    AblationConfig,
    DeepCompressorConfig,
    LossConfig,
    PerceiverConfig,
    ProjectionConfig,
    QwenConfig,
    TrainingConfig,
)
from deep_compressor.train import (
    _short_model_name,
    build_wandb_conf,
    generate_auto_tags,
    generate_run_name,
)


# ── fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def default_config():
    """Default config with known values for predictable test output."""
    return DeepCompressorConfig(
        qwen=QwenConfig(
            model_name_or_path="tiny",
            hidden_size=64,
            num_hidden_layers=4,
            vocab_size=100,
            max_doc_tokens=8192,
        ),
        perceiver=PerceiverConfig(
            perceiver_dim=64, num_queries=64, num_heads=4, head_dim=16,
            stage_a_cross_layers=1, stage_a_self_layers=1,
            stage_b_layers=1,
            stage_c_cross_layers=1, stage_c_self_layers=1,
            ff_mult=2, dropout=0.0,
        ),
        projection=ProjectionConfig(down_hidden=48, up_hidden=48, dropout=0.0),
        loss=LossConfig(hidden_distill_layers=[1, 3]),
        training=TrainingConfig(
            learning_rate=1e-4,
            batch_size=4,
            gradient_accumulation_steps=4,
        ),
    )


# ── _short_model_name ─────────────────────────────────────────────────

class TestShortModelName:
    def test_huggingface_name(self):
        assert _short_model_name("Qwen/Qwen3-0.6B") == "3-0.6b"

    def test_local_path(self):
        assert _short_model_name("models/Qwen3-0.6B") == "3-0.6b"

    def test_local_path_trailing_slash(self):
        assert _short_model_name("models/Qwen3-0.6B/") == "3-0.6b"

    def test_non_qwen_model(self):
        assert _short_model_name("facebook/llama-7b") == "llama-7b"

    def test_tiny(self):
        assert _short_model_name("tiny") == "tiny"

    def test_empty_fallback(self):
        assert _short_model_name("Qwen/Qwen") == "unknown"


# ── generate_run_name ─────────────────────────────────────────────────

class TestGenerateRunName:
    def test_format_with_timestamp(self, default_config):
        name = generate_run_name(default_config, timestamp="20260311_143052")
        assert name == "q64_lr1e-04_tiny_20260311_143052"

    def test_auto_timestamp(self, default_config):
        name = generate_run_name(default_config)
        # Should contain a timestamp-like pattern
        assert re.search(r"\d{8}_\d{6}$", name)

    def test_different_q_values(self, default_config):
        default_config.ablation.override_num_queries = 128
        name = generate_run_name(default_config, timestamp="20260101_000000")
        assert "q128" in name

    def test_different_lr(self, default_config):
        default_config.training.learning_rate = 5e-5
        name = generate_run_name(default_config, timestamp="20260101_000000")
        assert "lr5e-05" in name


# ── generate_auto_tags ────────────────────────────────────────────────

class TestGenerateAutoTags:
    def test_contains_model_tag(self, default_config):
        tags = generate_auto_tags(default_config)
        assert "model:tiny" in tags

    def test_contains_q_tag(self, default_config):
        tags = generate_auto_tags(default_config)
        assert "q64" in tags

    def test_contains_doc_length_tag(self, default_config):
        tags = generate_auto_tags(default_config)
        assert "doc8192" in tags

    def test_contains_lr_tag(self, default_config):
        tags = generate_auto_tags(default_config)
        assert "lr:1e-04" in tags

    def test_contains_ebs_tag(self, default_config):
        tags = generate_auto_tags(default_config)
        # effective_batch_size = 4 * 4 = 16
        assert "ebs16" in tags

    def test_no_proj_tags_for_identity(self, default_config):
        tags = generate_auto_tags(default_config)
        assert not any(t.startswith("down:") for t in tags)
        assert not any(t.startswith("up:") for t in tags)

    def test_proj_tags_for_mlp(self, default_config):
        default_config.ablation.down_proj_mode = "mlp"
        default_config.ablation.up_proj_mode = "linear"
        tags = generate_auto_tags(default_config)
        assert "down:mlp" in tags
        assert "up:linear" in tags

    def test_disabled_stage_tags(self, default_config):
        default_config.ablation.enable_stage_a = False
        default_config.ablation.enable_stage_c = False
        tags = generate_auto_tags(default_config)
        assert "no-stage-a" in tags
        assert "no-stage-c" in tags
        assert "no-stage-b" not in tags

    def test_override_num_queries(self, default_config):
        default_config.ablation.override_num_queries = 256
        tags = generate_auto_tags(default_config)
        assert "q256" in tags
        assert "q64" not in tags


# ── build_wandb_conf ──────────────────────────────────────────────────

class TestBuildWandbConf:
    def _make_args(self, **overrides):
        """Build a minimal argparse-like namespace for build_wandb_conf."""
        defaults = dict(
            wandb=True,
            config="configs/tiny_subset.yaml",
            wandb_project="test-project",
            wandb_entity=None,
            wandb_run_name=None,
            wandb_tags=None,
            wandb_group=None,
            wandb_notes=None,
            wandb_offline=False,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_returns_none_when_disabled(self):
        args = self._make_args(wandb=False)
        assert build_wandb_conf(args) is None

    def test_auto_generates_run_name(self):
        args = self._make_args()
        conf = build_wandb_conf(args)
        assert conf is not None
        assert conf.run_name is not None
        assert len(conf.run_name) > 0
        # Should contain q value and timestamp
        assert re.search(r"q\d+", conf.run_name)

    def test_explicit_run_name_preserved(self):
        args = self._make_args(wandb_run_name="my-custom-run")
        conf = build_wandb_conf(args)
        assert conf.run_name == "my-custom-run"

    def test_auto_tags_generated(self):
        args = self._make_args()
        conf = build_wandb_conf(args)
        assert len(conf.tags) > 0
        # Should have auto-generated model tag
        assert any(t.startswith("model:") for t in conf.tags)

    def test_explicit_tags_merged(self):
        args = self._make_args(wandb_tags="exp1,baseline")
        conf = build_wandb_conf(args)
        assert "exp1" in conf.tags
        assert "baseline" in conf.tags
        # Auto tags should also be present
        assert any(t.startswith("model:") for t in conf.tags)

    def test_explicit_tags_come_first(self):
        args = self._make_args(wandb_tags="my-first-tag")
        conf = build_wandb_conf(args)
        assert conf.tags[0] == "my-first-tag"

    def test_no_duplicate_tags(self):
        args = self._make_args(wandb_tags="q64")
        conf = build_wandb_conf(args)
        # q64 appears in explicit and auto — should not be duplicated
        assert conf.tags.count("q64") == 1

    def test_group_and_notes(self):
        args = self._make_args(wandb_group="ablation-v2",
                               wandb_notes="Testing stage C removal")
        conf = build_wandb_conf(args)
        assert conf.group == "ablation-v2"
        assert conf.notes == "Testing stage C removal"

    def test_project_name(self):
        args = self._make_args(wandb_project="my-project")
        conf = build_wandb_conf(args)
        assert conf.project == "my-project"

    def test_offline_mode(self):
        args = self._make_args(wandb_offline=True)
        conf = build_wandb_conf(args)
        assert conf.offline is True
