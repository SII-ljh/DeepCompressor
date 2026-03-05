"""Shared test fixtures with tiny model configs."""

import pytest
import torch
import torch.nn as nn

from deep_compressor.config import (
    DeepCompressorConfig,
    FinBERTConfig,
    LossConfig,
    PerceiverConfig,
    ProjectionConfig,
    QwenConfig,
    TrainingConfig,
)


@pytest.fixture
def tiny_config():
    """Tiny config for fast unit tests (no real model loading).

    perceiver_dim = qwen.hidden_size = 64 (sequence-length-only compression).
    """
    return DeepCompressorConfig(
        qwen=QwenConfig(
            model_name_or_path="tiny",
            hidden_size=64,
            num_hidden_layers=4,
            vocab_size=100,
            max_doc_tokens=128,
            max_question_tokens=32,
            max_answer_tokens=64,
        ),
        finbert=FinBERTConfig(enabled=False, hidden_size=48, num_ner_labels=15,
                              top_k_anchors=4, anchor_align_layers=2),
        perceiver=PerceiverConfig(
            perceiver_dim=64, num_queries=8, num_heads=4, head_dim=16,
            stage_a_cross_layers=1, stage_a_self_layers=1,
            stage_b_layers=1,
            stage_c_cross_layers=1, stage_c_self_layers=1,
            ff_mult=2, dropout=0.0,
        ),
        projection=ProjectionConfig(down_hidden=48, up_hidden=48, dropout=0.0),
        loss=LossConfig(hidden_distill_layers=[1, 3]),
        training=TrainingConfig(stage=1, batch_size=2),
    )


@pytest.fixture
def tiny_config_finbert():
    """Tiny config with FinBERT enabled.

    perceiver_dim = qwen.hidden_size = 64 (sequence-length-only compression).
    """
    return DeepCompressorConfig(
        qwen=QwenConfig(
            model_name_or_path="tiny",
            hidden_size=64,
            num_hidden_layers=4,
            vocab_size=100,
        ),
        finbert=FinBERTConfig(enabled=True, hidden_size=48, num_ner_labels=15,
                              top_k_anchors=4, anchor_align_layers=2),
        perceiver=PerceiverConfig(
            perceiver_dim=64, num_queries=8, num_heads=4, head_dim=16,
            stage_a_cross_layers=1, stage_a_self_layers=1,
            stage_b_layers=1,
            stage_c_cross_layers=1, stage_c_self_layers=1,
            ff_mult=2, dropout=0.0,
        ),
        projection=ProjectionConfig(down_hidden=48, up_hidden=48, dropout=0.0),
        loss=LossConfig(hidden_distill_layers=[1, 3]),
    )


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def doc_len():
    return 16


@pytest.fixture
def q_len():
    return 8
