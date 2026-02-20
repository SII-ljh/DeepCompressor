"""Hydra structured configuration for Deep Compressor.

Maps 1-to-1 with the dataclasses in ``config.py`` so that Hydra CLI
overrides are validated at parse time, then converted to the canonical
``DeepCompressorConfig`` via :func:`to_deep_compressor_config`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from deep_compressor.config import (
    AblationConfig,
    DeepCompressorConfig,
    FinBERTConfig,
    LossConfig,
    PerceiverConfig,
    ProjectionConfig,
    QwenConfig,
    TrainingConfig,
)


# ── Hydra sub-configs (mirror config.py) ──────────────────────────────

@dataclass
class QwenHydraConf:
    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    hidden_size: int = 1024
    num_hidden_layers: int = 28
    vocab_size: int = 151936
    max_doc_tokens: int = 8192
    max_question_tokens: int = 256
    max_answer_tokens: int = 512


@dataclass
class FinBERTHydraConf:
    enabled: bool = False
    model_name_or_path: str = "valuesimplex/FinBERT2"
    hidden_size: int = 768
    num_ner_labels: int = 15
    top_k_anchors: int = 32
    anchor_align_layers: int = 3


@dataclass
class PerceiverHydraConf:
    perceiver_dim: int = 1024
    num_queries: int = 64
    num_heads: int = 16
    head_dim: int = 64
    stage_a_cross_layers: int = 2
    stage_a_self_layers: int = 2
    stage_b_layers: int = 2
    stage_c_cross_layers: int = 2
    stage_c_self_layers: int = 4
    ff_mult: int = 4
    anchor_score_scale_init: float = 1.0
    dropout: float = 0.1


@dataclass
class ProjectionHydraConf:
    down_hidden: int = 768
    up_hidden: int = 768
    dropout: float = 0.1


@dataclass
class LossHydraConf:
    kl_temperature: float = 2.0
    hidden_distill_ramp_steps: int = 2000
    hidden_distill_layers: List[int] = field(
        default_factory=lambda: [7, 14, 21, 27]
    )
    qa_ce_weight: float = 1.0
    kl_weight: float = 1.0
    hidden_mse_weight: float = 1.0
    anchor_recon_weight: float = 0.5


@dataclass
class TrainingHydraConf:
    stage: int = 1
    learning_rate: float = 1e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_steps: int = 50000
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    scheduler: str = "cosine"
    seed: int = 42
    log_every: int = 50
    eval_every: int = 500
    save_every: int = 1000
    output_dir: str = "outputs"
    ntp_segment_len: int = 256
    gradient_checkpointing: bool = False
    mixed_precision: str = "no"


@dataclass
class AblationHydraConf:
    down_proj_mode: str = "mlp"
    up_proj_mode: str = "mlp"
    query_condition_on_question: bool = True
    enable_stage_a: bool = True
    enable_stage_b: bool = True
    enable_stage_c: bool = True
    override_stage_a_cross_layers: int = 0
    override_stage_a_self_layers: int = 0
    override_stage_b_layers: int = 0
    override_stage_c_cross_layers: int = 0
    override_stage_c_self_layers: int = 0
    enable_kl_distillation: bool = True
    enable_hidden_mse_distillation: bool = True
    override_num_queries: int = 0


@dataclass
class WandbConf:
    enabled: bool = True
    project: str = "deep-compressor"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    offline: bool = False


# ── top-level config ──────────────────────────────────────────────────

@dataclass
class RunConf:
    qwen: QwenHydraConf = field(default_factory=QwenHydraConf)
    finbert: FinBERTHydraConf = field(default_factory=FinBERTHydraConf)
    perceiver: PerceiverHydraConf = field(default_factory=PerceiverHydraConf)
    projection: ProjectionHydraConf = field(default_factory=ProjectionHydraConf)
    loss: LossHydraConf = field(default_factory=LossHydraConf)
    training: TrainingHydraConf = field(default_factory=TrainingHydraConf)
    ablation: AblationHydraConf = field(default_factory=AblationHydraConf)

    # runtime parameters (not part of DeepCompressorConfig)
    data_path: str = MISSING
    eval_data_path: Optional[str] = None
    resume_from: Optional[str] = None
    max_eval_samples: int = 0
    max_train_samples: int = 0

    # wandb
    wandb: WandbConf = field(default_factory=WandbConf)


# ── conversion ────────────────────────────────────────────────────────

def to_deep_compressor_config(cfg: DictConfig) -> DeepCompressorConfig:
    """Convert a Hydra ``DictConfig`` to a ``DeepCompressorConfig``.

    Triggers ``__post_init__`` validation in the canonical dataclass.
    """
    return DeepCompressorConfig(
        qwen=QwenConfig(**OmegaConf.to_container(cfg.qwen, resolve=True)),
        finbert=FinBERTConfig(**OmegaConf.to_container(cfg.finbert, resolve=True)),
        perceiver=PerceiverConfig(**OmegaConf.to_container(cfg.perceiver, resolve=True)),
        projection=ProjectionConfig(**OmegaConf.to_container(cfg.projection, resolve=True)),
        loss=LossConfig(**OmegaConf.to_container(cfg.loss, resolve=True)),
        training=TrainingConfig(**OmegaConf.to_container(cfg.training, resolve=True)),
        ablation=AblationConfig(**OmegaConf.to_container(cfg.ablation, resolve=True)),
    )


# ── register with ConfigStore ─────────────────────────────────────────

cs = ConfigStore.instance()
cs.store(name="base_config", node=RunConf)
