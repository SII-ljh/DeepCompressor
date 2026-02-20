# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Deep Compressor** -- a system for compressing ultra-long financial texts (8K-64K tokens) into fixed-length latent vector sequences that serve as prefixes for a frozen Qwen decoder to answer detail-level financial questions. Teacher distillation ensures the compressed prefix carries information equivalent to the full document. FinBERT entity guidance is an optional enhancement module (default OFF).

The architectural plan lives in `plan.md`. The `FinBERT/` subdirectory is a cloned reference repo ([valuesimplex/FinBERT2](https://github.com/valuesimplex/FinBERT2)) used for NER and entity-level features.

## Environment Setup

```bash
conda create --name FinBERT python=3.11
conda activate FinBERT
pip install -r requirements.txt
```

The conda env is at `/opt/homebrew/anaconda3/envs/FinBERT`. Use its Python directly when needed:
```bash
/opt/homebrew/anaconda3/envs/FinBERT/bin/python <script>
```

## Key Dependencies

torch >=2.0, transformers >=4.40, datasets >=2.20, sentence-transformers >=3.0, accelerate >=0.20, sentencepiece, jieba, bertopic, scikit-learn, pandas, numpy, optuna, wandb (optional).

## Project Structure

```
deep_compressor/              # Core library
  config.py                   # Dataclass configs (QwenConfig, PerceiverConfig, AblationConfig, etc.)
  hydra_conf.py               # Hydra structured config (mirrors config.py for CLI overrides)
  data.py                     # NTPDataset, QADataset, PaddingCollator
  model.py                    # DeepCompressor main model
  train.py                    # Training loop (_run_training)
  loss.py                     # DistillationLoss, compute_total_loss
  eval.py                     # Evaluation metrics (EM, F1)
  modules/
    down_proj.py              # DownProj, IdentityProj, LinearProj, build_down_proj()
    up_mlp.py                 # UpMLP, build_up_proj()
    query_init.py             # QueryInit (learnable base queries + question bias)
    perceiver.py              # GuidedPerceiver (Stage A/B/C with enable flags)
    anchor_align.py           # AnchorAlign (FinBERT optional)
    ner_head.py               # NERHead (FinBERT optional)
    fact_decode_head.py        # FactDecodeHead (FinBERT optional)
    tokenizer_align.py         # FinBERT-Qwen tokenizer alignment

configs/                      # YAML configuration files
  default.yaml                # Full training config
  tiny_subset.yaml            # Quick iteration config (tiny data)
  macbook_debug.yaml          # Local debug config (CPU/MPS)
  ablation_base.yaml          # Ablation experiments base config
  benchmark.yaml              # Benchmark evaluation config
  hp_search.yaml              # Hyperparameter search config

scripts/
  prepare_data.py             # Data download & subset generation
  diagnostic.py               # 5 diagnostic experiments (overfit, bottleneck, gradient flow, attention, fidelity)
  hp_search.py                # Optuna HP search with wandb integration
  ablation.py                 # Ablation experiment runner (17 registered experiments)
  benchmark.py                # Baseline comparison (direct_qwen, random_prefix, pool baselines, deep_compressor)
  visualize_architecture.py   # Architecture diagram generation

tests/                        # Unit tests (pytest)
```

## Architecture

The data flow has five stages:

1. **Document Encoding** -- Frozen Qwen encoder -> hidden states -> `DownProj` -> `byte_array` in Perceiver dimension.
2. **Entity Guidance (optional)** -- Frozen FinBERT + NER Head -> `anchor_embs` (via `AnchorAlign`) + `anchor_scores`. Default OFF.
3. **Query Encoding** -- Frozen Qwen encoder -> mean-pooled question vector -> `QueryInit` (learnable base queries + question additive bias) -> initial query vectors.
4. **Compression** -- `GuidedPerceiver` with three stages:
   - Stage A: global cross-attention from `byte_array` (with optional `anchor_scores` bias) + self-attention.
   - Stage B: anchor refinement cross-attention (FinBERT ON) or extra self-attention (FinBERT OFF).
   - Stage C: deep reasoning cross-attention back to `byte_array` + multi-layer self-attention -> fixed-length `latent_array`.
5. **Generation** -- `UpMLP` maps `latent_array` back to Qwen dimension -> prefix + question tokens -> frozen Qwen decoder -> answer.

### Ablation Infrastructure

The `AblationConfig` dataclass controls experimental switches:
- **Projection modes**: `down_proj_mode` / `up_proj_mode` -- "mlp" (default), "linear", or "identity"
- **Query conditioning**: `query_condition_on_question` -- toggle question-conditioned query bias
- **Perceiver stage enables**: `enable_stage_a/b/c` -- disable individual Perceiver stages
- **Layer count overrides**: `override_stage_*_layers` -- override default layer counts (0 = use defaults)
- **Distillation toggles**: `enable_kl_distillation` / `enable_hidden_mse_distillation`
- **Query count override**: `override_num_queries` -- override default num_queries (0 = use defaults)

Factory functions `build_down_proj()` and `build_up_proj()` create projection modules based on the ablation mode. `DeepCompressorConfig` provides computed properties (`effective_num_queries`, `effective_stage_*_layers`) that resolve overrides.

### Distillation

Teacher is the same frozen Qwen reading the full uncompressed document. Two distillation losses:
- **Output distribution**: KL divergence on answer token logits (temperature-scaled).
- **Hidden state alignment**: MSE on shared question+answer token positions across selected decoder layers.

### Training Strategy

- **Stage 1 (NTP Pretraining)**: compress document -> prefix -> next-token prediction on random document segments. No labels needed. Solves cold-start problem.
- **Stage 2 (QA Fine-tuning + Distillation)**: joint training with QA cross-entropy + KL distillation + hidden-state MSE (weight ramped from 0). Uses SQuAD, CMRC2018, DuReader, TriviaQA, DRCD.

## Data Subsets

Generated by `scripts/prepare_data.py`:

```
data/
  ntp_train.jsonl              # Full NTP training data
  qa_train.json                # Full QA training data
  qa_dev.json                  # Full QA dev data
  ntp_tiny.jsonl               # 50 samples (pipeline smoke test)
  qa_tiny_train.json           # 50 samples
  qa_tiny_dev.json             # 20 samples
  ntp_dev.jsonl                # 2000 samples, stratified by doc length
  qa_dev_hp.json               # 500 samples, stratified by source
  ablation/
    ntp_ablation.jsonl         # 1000 samples (seed=123)
    qa_ablation_train.json     # 1000 samples (seed=123)
    qa_ablation_dev.json       # 200 samples (seed=123)
```

CLI flags:
```bash
python scripts/prepare_data.py                    # full download
python scripts/prepare_data.py --test             # small test subset
python scripts/prepare_data.py --make-tiny        # ~50 samples
python scripts/prepare_data.py --make-dev         # ~2000 NTP + 500 QA
python scripts/prepare_data.py --make-ablation    # ablation subsets
python scripts/prepare_data.py --make-all-subsets # generate all subsets
```

## Running Experiments

### Diagnostics (5 experiments)
```bash
python scripts/diagnostic.py \
    --config configs/macbook_debug.yaml \
    --data_path data/ntp_tiny.jsonl \
    --steps 300 --experiments 1,2,3,4,5
```

### Hyperparameter Search
```bash
python scripts/hp_search.py --n_trials 50 --stage 1 \
    --config configs/hp_search.yaml \
    --data_path data/ntp_tiny.jsonl \
    --wandb --wandb_project dc-hp-search
```

### Ablation Experiments
```bash
# List all available ablations
python scripts/ablation.py --list

# Run specific ablations
python scripts/ablation.py --stage 1 \
    --config configs/ablation_base.yaml \
    --data_path data/ablation/ntp_ablation.jsonl \
    --ablation full_pipeline,no_stage_c,down_proj_identity

# Run all ablations with wandb
python scripts/ablation.py --stage 1 \
    --data_path data/ablation/ntp_ablation.jsonl \
    --wandb --wandb_project dc-ablation
```

Available ablation experiments: `down_proj_identity`, `down_proj_linear`, `up_proj_identity`, `up_proj_linear`, `query_no_question`, `no_stage_a`, `no_stage_b`, `no_stage_c`, `shallow_perceiver`, `deep_perceiver`, `no_kl_distill`, `no_mse_distill`, `no_distillation`, `queries_16`, `queries_32`, `queries_128`, `full_pipeline`.

### Benchmark Comparison
```bash
python scripts/benchmark.py \
    --config configs/benchmark.yaml \
    --checkpoint outputs/checkpoint-final/trainable_weights.pt \
    --eval_data data/qa_dev.json \
    --wandb --wandb_project dc-benchmark
```

Compares: `direct_qwen` (upper bound), `random_prefix` (lower bound), `mean_pool_linear`, `mean_pool_mlp`, `deep_compressor` (requires checkpoint).

## Testing

```bash
# Run all tests
python -m pytest tests/ -x -q

# Run specific test module
python -m pytest tests/test_model.py -v
```

64 tests covering all modules, config validation, data loading, loss computation, eval metrics, and integration.

## Configuration System

Two configuration paths:
1. **YAML + `from_yaml()`**: Used by scripts (`train.py`, `ablation.py`, `benchmark.py`, etc.)
2. **Hydra + `RunConf`**: Used for CLI override workflows via `hydra_conf.py`

Both paths produce a validated `DeepCompressorConfig` with `__post_init__` checks.

## Implementation Notes

- All frozen models (Qwen encoder/decoder, FinBERT) must have `requires_grad=False`. Only trainable modules: DownProj, QueryInit, GuidedPerceiver, UpMLP, and optionally AnchorAlign/NER Head/FactDecodeHead.
- `encode_document()` and `encode_question()` use `torch.no_grad()` context managers around the frozen Qwen forward pass, then call trainable modules outside the context to preserve gradient flow.
- Teacher path is inference-only (no gradients). For 64K inputs, pre-compute and cache Teacher logits/hidden states to manage GPU memory.
- Hidden-state distillation weight should ramp gradually from 0 to avoid early training instability.
- Tokenizer alignment between FinBERT and Qwen uses character-level span overlap (max aggregation).
- Evaluation metrics: exact match for numeric questions, F1 for overall QA quality.

## FinBERT Reference Repo

```
FinBERT/
  Fin-NER/              # NER fine-tuning & inference (BIO tags)
  Fin-labeler/          # Sentiment classification
  Fin-retriever/        # Contrastive learning retrieval
  Fin-Topicmodel/       # BERTopic-based topic modeling
  FinBERT2/pretrain/    # MLM pretraining
```

Entity types: ORG, PER, METRIC, VALUE, DATE, EVENT, PRODUCT.

FinBERT toggle is controlled by `finbert.enabled` config flag. When OFF: no FinBERT/NER/AnchorAlign modules, Stage B degrades to self-attention, no anchor_scores bias, no auxiliary anchor reconstruction loss. System is fully functional without FinBERT.
