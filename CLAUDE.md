# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Deep Compressor** — compresses ultra-long financial texts (8K-64K tokens) into fixed-length latent vector sequences that serve as prefixes for a frozen Qwen decoder to answer detail-level financial questions. Teacher distillation ensures the compressed prefix carries information equivalent to the full document. FinBERT entity guidance is an optional enhancement module (default OFF).

The architectural plan lives in `plan.md` (Chinese). The `FinBERT/` subdirectory is a cloned reference repo ([valuesimplex/FinBERT2](https://github.com/valuesimplex/FinBERT2)) used for NER and entity-level features.

## Environment Setup

```bash
conda create --name FinBERT python=3.11
conda activate FinBERT
pip install -r FinBERT/requirements.txt
pip install -r requirements.txt
```

The conda env is at `/opt/homebrew/anaconda3/envs/FinBERT`. Use its Python directly when needed:
```bash
/opt/homebrew/anaconda3/envs/FinBERT/bin/python <script>
```

## Testing

```bash
# Run all tests (skips slow integration tests)
python -m pytest tests/ -x -q

# Include slow integration tests (require Qwen3-0.6B download)
python -m pytest tests/ --runslow

# Run specific test module
python -m pytest tests/test_model.py -v
```

67 tests across 14 files. Integration tests in `test_integration.py` are marked `@pytest.mark.slow` and skipped by default — they require the real Qwen model.

**Test fixtures** (`tests/conftest.py`): `tiny_config` provides a `DeepCompressorConfig` with minimal dimensions (Qwen 64D/4 layers, Perceiver 32D/8 queries, FinBERT off) for fast unit tests. `tiny_config_finbert` is the same with FinBERT enabled. Tests use a `_MockQwenModel` in `test_model.py` that mimics the Qwen3 interface to avoid loading the real 600M-parameter model.

## Training

```bash
# Stage 1: NTP Pretraining (compress doc -> prefix -> next-token prediction)
python -m deep_compressor.train \
    --config configs/<config>.yaml \
    --data_path data/ntp_train.jsonl --stage 1

# Stage 2: QA Fine-tuning + Distillation (requires Stage 1 checkpoint)
python -m deep_compressor.train \
    --config configs/default.yaml \
    --data_path data/qa_train.json --eval_data_path data/qa_dev.json \
    --stage 2 --resume_from outputs/checkpoint-final

# Multi-GPU with Accelerate
accelerate launch --multi_gpu --num_processes 4 \
    -m deep_compressor.train --config configs/default.yaml \
    --data_path data/ntp_train.jsonl --stage 1

# With wandb tracking (append to any training command)
--wandb --wandb_project deep-compressor

# Hydra CLI overrides (alternative config path)
python -m deep_compressor.train \
    --config-path ../configs --config-name default \
    training.learning_rate=1e-3 perceiver.num_queries=128
```

**Config files** — choose based on scenario:
| Config | Use Case | Doc Tokens | Steps |
|--------|----------|-----------|-------|
| `default.yaml` | Full production training | 8192 | 50K |
| `tiny_subset.yaml` | Quick iteration / HP tuning | 256 | 200 |
| `macbook_debug.yaml` | Local MacBook validation (MPS) | 256 | 300 |
| `ablation_base.yaml` | Ablation experiments | 512 | 500 |
| `hp_search.yaml` | Hyperparameter search | 256 | 200 |
| `benchmark.yaml` | Evaluation-only | 8192 | — |

## Data Preparation

```bash
python scripts/prepare_data.py                    # full download (~1-2 hours)
python scripts/prepare_data.py --test             # small test subset
python scripts/prepare_data.py --make-tiny        # ~50 samples for pipeline smoke test
python scripts/prepare_data.py --make-all-subsets # all subsets (tiny, dev, ablation)
```

Downloads Qwen3-0.6B to `models/Qwen3-0.6B/` and builds NTP/QA datasets into `data/`. Key subsets: `ntp_tiny.jsonl` (50 samples), `qa_tiny_*.json` (50/20 samples), `ablation/` (1000/200 samples).

## Running Experiments

### Diagnostics (9 experiments, 3 phases)

```bash
# Pre-training (Exp 1-3): overfit, gradient flow, bottleneck
python scripts/diagnostics/pre_training.py \
    --config configs/macbook_debug.yaml \
    --data_path data/ntp_tiny.jsonl --steps 300 --experiments 1,2,3

# Mid-training (Exp 4-5): query diversity, stagewise info gain
python scripts/diagnostics/mid_training.py \
    --config configs/macbook_debug.yaml \
    --data_path data/ntp_tiny.jsonl --experiments 4,5

# Mid-training as callback during training
python -m deep_compressor.train --config configs/tiny_subset.yaml \
    --data_path data/ntp_tiny.jsonl --stage 1 \
    --diagnostic_every 50 --diagnostic_experiments 4,5

# Post-training (Exp 6-9): attention, fidelity, length scaling, distillation
python scripts/diagnostics/post_training.py \
    --config configs/benchmark.yaml \
    --checkpoint outputs/checkpoint-final/trainable_weights.pt \
    --eval_data data/qa_dev.json --experiments 6,7,8,9
```

Legacy `scripts/diagnostic.py` still works but emits a DeprecationWarning.

### Ablation, HP Search, Benchmark

```bash
# List all 17 ablation experiments
python scripts/ablation.py --list

# Run specific ablations
python scripts/ablation.py --stage 1 \
    --config configs/ablation_base.yaml \
    --data_path data/ablation/ntp_ablation.jsonl \
    --ablation full_pipeline,no_stage_c,down_proj_identity

# Hyperparameter search
python scripts/hp_search.py --n_trials 50 --stage 1 \
    --config configs/hp_search.yaml \
    --data_path data/ntp_tiny.jsonl

# Benchmark comparison (direct_qwen, random_prefix, pool baselines, deep_compressor)
python scripts/benchmark.py \
    --config configs/benchmark.yaml \
    --checkpoint outputs/checkpoint-final/trainable_weights.pt \
    --eval_data data/qa_dev.json
```

## Architecture

### Data Flow (5 stages)

1. **Document Encoding** — Frozen Qwen encoder -> hidden states -> `DownProj` -> `byte_array` in Perceiver dimension.
2. **Entity Guidance (optional, default OFF)** — Frozen FinBERT + NER Head -> `anchor_embs` (via `AnchorAlign`) + `anchor_scores`.
3. **Query Encoding** — Frozen Qwen encoder -> mean-pooled question vector -> `QueryInit` (learnable base queries + question additive bias) -> initial query vectors.
4. **Compression** — `GuidedPerceiver` with three stages:
   - Stage A: global cross-attention from `byte_array` (with optional `anchor_scores` bias) + self-attention.
   - Stage B: anchor refinement cross-attention (FinBERT ON) or extra self-attention (FinBERT OFF).
   - Stage C: deep reasoning cross-attention back to `byte_array` + multi-layer self-attention -> fixed-length `latent_array`.
5. **Generation** — `UpMLP` maps `latent_array` back to Qwen dimension -> prefix + question tokens -> frozen Qwen decoder -> answer.

### Gradient Flow

Frozen Qwen forward passes run inside `torch.no_grad()` context managers. Trainable modules (`DownProj`, `QueryInit`, `GuidedPerceiver`, `UpMLP`) are called **outside** the no-grad context to preserve gradient flow. The teacher path (same frozen Qwen reading full uncompressed document) is inference-only with no gradients.

### Distillation

Two distillation losses (Stage 2 only):
- **Output distribution**: KL divergence on answer token logits (temperature-scaled).
- **Hidden state alignment**: MSE on shared question+answer token positions across selected decoder layers. Weight ramps linearly from 0 over `hidden_distill_ramp_steps` to avoid early training instability.

### Configuration System

Two paths producing the same validated `DeepCompressorConfig`:
1. **YAML + `DeepCompressorConfig.from_yaml(path)`** — used by all scripts.
2. **Hydra + `RunConf` in `hydra_conf.py`** — used for CLI override workflows.

Config hierarchy: `DeepCompressorConfig` wraps `QwenConfig`, `FinBERTConfig`, `PerceiverConfig`, `ProjectionConfig`, `LossConfig`, `TrainingConfig`, `AblationConfig`. `__post_init__` validates constraints like `num_heads * head_dim == perceiver_dim`.

### Ablation Infrastructure

`AblationConfig` controls experimental switches: projection modes (`down_proj_mode`/`up_proj_mode`: mlp/linear/identity), query conditioning, Perceiver stage enables (`enable_stage_a/b/c`), layer count overrides, distillation toggles, query count override. `DeepCompressorConfig` provides `effective_*` computed properties that resolve overrides.

Factory functions `build_down_proj()` and `build_up_proj()` create projection modules based on ablation mode.

## Implementation Notes

- **Frozen vs trainable**: All Qwen/FinBERT params have `requires_grad=False`. Trainable: DownProj, QueryInit, GuidedPerceiver, UpMLP; optionally AnchorAlign, NERHead, FactDecodeHead.
- **Checkpoint saving**: Only trainable weights are saved (skips `qwen.*` parameters).
- **Lazy loading**: `NTPDataset` uses byte offsets into JSONL files to avoid loading multi-GB data into memory.
- **Distributed training**: Uses HuggingFace `accelerate` for automatic DDP + device placement.
- **Teacher caching**: For 64K inputs, pre-compute and cache teacher logits/hidden states to manage GPU memory.
- **Tokenizer alignment**: FinBERT-Qwen alignment in `tokenizer_align.py` uses character-level span overlap with max aggregation.
- **Evaluation metrics**: Exact match for numeric questions, F1 for overall QA quality. Chinese text normalization handled via jieba.

## FinBERT Toggle

Controlled by `finbert.enabled` config flag. When OFF (default): no FinBERT/NER/AnchorAlign modules loaded, Stage B degrades to self-attention, no anchor_scores bias, no auxiliary anchor reconstruction loss. System is fully functional without FinBERT.

Entity types when enabled: ORG, PER, METRIC, VALUE, DATE, EVENT, PRODUCT.
