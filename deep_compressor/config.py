"""Configuration dataclasses for Deep Compressor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class QwenConfig:
    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    hidden_size: int = 1024
    num_hidden_layers: int = 28
    vocab_size: int = 151936
    max_doc_tokens: int = 8192
    max_question_tokens: int = 256
    max_answer_tokens: int = 512


@dataclass
class FinBERTConfig:
    enabled: bool = False
    model_name_or_path: str = "valuesimplex/FinBERT2"
    hidden_size: int = 768
    num_ner_labels: int = 15  # BIO tags for 7 entity types + O
    top_k_anchors: int = 32
    anchor_align_layers: int = 3


@dataclass
class PerceiverConfig:
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
class ProjectionConfig:
    down_hidden: int = 768
    up_hidden: int = 768
    dropout: float = 0.1


@dataclass
class LossConfig:
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
class TrainingConfig:
    stage: int = 1  # 1 = NTP pretraining, 2 = QA + distillation
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
    mixed_precision: str = "no"  # "no", "fp16", "bf16"


@dataclass
class DeepCompressorConfig:
    qwen: QwenConfig = field(default_factory=QwenConfig)
    finbert: FinBERTConfig = field(default_factory=FinBERTConfig)
    perceiver: PerceiverConfig = field(default_factory=PerceiverConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        if self.perceiver.num_heads * self.perceiver.head_dim != self.perceiver.perceiver_dim:
            raise ValueError(
                f"num_heads ({self.perceiver.num_heads}) * head_dim ({self.perceiver.head_dim}) "
                f"must equal perceiver_dim ({self.perceiver.perceiver_dim})"
            )
        for layer_idx in self.loss.hidden_distill_layers:
            if layer_idx >= self.qwen.num_hidden_layers:
                raise ValueError(
                    f"hidden_distill_layer {layer_idx} >= num_hidden_layers {self.qwen.num_hidden_layers}"
                )

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_yaml(cls, path: str) -> "DeepCompressorConfig":
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            qwen=QwenConfig(**raw.get("qwen", {})),
            finbert=FinBERTConfig(**raw.get("finbert", {})),
            perceiver=PerceiverConfig(**raw.get("perceiver", {})),
            projection=ProjectionConfig(**raw.get("projection", {})),
            loss=LossConfig(**raw.get("loss", {})),
            training=TrainingConfig(**raw.get("training", {})),
        )
