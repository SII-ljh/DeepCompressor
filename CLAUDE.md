# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Deep Compressor** — compresses ultra-long financial texts (8K-64K tokens) into fixed-length latent vector sequences that serve as prefixes for a frozen Qwen decoder to answer detail-level financial questions. Teacher distillation ensures the compressed prefix carries information equivalent to the full document.

The architectural plan lives in `plan.md` (Chinese).

## Environment Setup

```bash
conda create --name deep_compressor python=3.11
conda activate deep_compressor
pip install -r requirements.txt
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

69 tests across 14 files. Integration tests in `test_integration.py` are marked `@pytest.mark.slow` and skipped by default — they require the real Qwen model.

**Test fixtures** (`tests/conftest.py`): `tiny_config` provides a `DeepCompressorConfig` with minimal dimensions (Qwen 64D/4 layers, Perceiver 32D/8 queries) for fast unit tests. Tests use a `_MockQwenModel` in `test_model.py` that mimics the Qwen3 interface to avoid loading the real 600M-parameter model.

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
python scripts/prepare_data.py --make-ablation    # 5% of total for ablation experiments
python scripts/prepare_data.py --make-all-subsets # all subsets (tiny, dev, ablation)
```

Downloads Qwen3-0.6B to `models/Qwen3-0.6B/` and builds NTP/QA datasets into `data/`. Key subsets: `ntp_tiny.jsonl` (50 samples), `qa_tiny_*.json` (50/20 samples), `ablation/` (5% of total data, stratified by source).

## Stage 1 Varying Query (Q) Experiments

**NEW**: Automated workflow for training multiple Stage 1 models with different `num_queries` values to find the optimal compression ratio.

### Quick Start

```bash
# One-command full pipeline: data prep + training + evaluation
bash scripts/run_full_experiment.sh

# Or manual steps:
# 1. Filter data (keep docs < 512 tokens)
python scripts/filter_ntp_data.py \
    --input data/ntp_train.jsonl \
    --output data/ntp_train_512.jsonl \
    --max_length 512

# 2. Train all Q values (16, 32, 64, 128, 256)
python scripts/train_stage1_varying_q.py

# 3. Evaluate all checkpoints
python scripts/evaluate_all_checkpoints.py \
    --eval_data data/ntp_train_512.jsonl \
    --stage 1 \
    --output results/stage1_results.csv
```

### Files Created

- `configs/stage1_q{16,32,64,128,256}.yaml` — configs for each Q value
- `scripts/filter_ntp_data.py` — filter training data by length
- `scripts/train_stage1_varying_q.py` — batch training for all Q values
- `scripts/evaluate_all_checkpoints.py` — auto-discover and eval all checkpoints
- `scripts/analyze_data_distribution.py` — analyze document length distribution
- `scripts/run_full_experiment.sh` — one-click full pipeline
- `scripts/STAGE1_VARYING_Q_EXPERIMENTS.md` — detailed documentation
- `scripts/QUICKSTART_STAGE1_Q_EXPERIMENTS.md` — quick reference

### Training Output

Each Q value saves to separate directory: `outputs/stage1_q{16,32,64,128,256}/checkpoint-final/`.

### Evaluation Output

Produces comparison table with **direct_qwen baseline** (Qwen reading full document without compression) and CSV:
```
Model                     | perplexity | loss    | Retention
----------------------------------------------------------------
Direct Qwen (baseline)    |    18.23   |  2.90   |     —
Q=16                      |    45.68   |  3.82   |   76.0%
Q=32                      |    32.12   |  3.47   |   83.7%
Q=64                      |    24.57   |  3.20   |   90.7%
Q=128                     |    20.99   |  3.04   |   95.4%
Q=256                     |    19.46   |  2.97   |   97.8%
```

**Retention** = (baseline_loss / model_loss) × 100% — measures how much quality is preserved after compression.

### Sample Display During Eval

**NEW**: Both NTP and QA evaluation now display sample predictions during training:
- NTP: Shows compressed prefix → generated continuation vs gold
- QA: Shows question → prediction vs gold answer (with EM/F1)

Modified functions:
- `evaluate_ntp(..., tokenizer, show_sample=True)` — now displays sample predictions
- `evaluate_qa(..., show_samples=5)` — already had this feature, now more prominent

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

### Progressive Overfitting Tests

`scripts/overfitting/` contains a 3-step progressive validation suite:
1. `step1_single_sample.py` — overfit on 1 sample (loss should → 0)
2. `step2_memorize_tiny.py` — memorize ~50 samples
3. `step3_ablation_full.py` — full ablation on real data

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

### Key Design Constraint: Sequence-Length-Only Compression

For known Qwen3 models, `__post_init__` forces `perceiver_dim = qwen.hidden_size` (e.g., 1024 for Qwen3-0.6B). This means DownProj/UpProj default to **identity** mode (no parameters) — compression happens purely via sequence length reduction (doc_len → num_queries), not dimensionality reduction. The MLP/linear projection modes exist for ablation experiments only.

Multi-model support: `QWEN3_REGISTRY` in `config.py` maps Qwen3-0.6B through 8B, auto-resolving hidden_size/num_layers/vocab_size and recomputing `perceiver_dim` and `head_dim`.

### Data Flow (4 stages)

1. **Document Encoding** — Frozen Qwen encoder (`self.qwen.model(...)`) -> hidden states -> `DownProj` (identity by default) -> `byte_array` in Perceiver dimension.
2. **Query Encoding** — Frozen Qwen encoder -> mean-pooled question vector -> `QueryInit` (learnable base queries + question additive bias) -> initial query vectors. For NTP (Stage 1), question vector is zero → pure base queries.
3. **Compression** — `GuidedPerceiver` with three stages:
   - Stage A: global cross-attention from `byte_array` + self-attention.
   - Stage B: self-attention.
   - Stage C: deep reasoning cross-attention back to `byte_array` + multi-layer self-attention -> fixed-length `latent_array`.
4. **Generation** — `UpMLP` (identity by default) maps `latent_array` back to Qwen dimension -> prefix + question tokens -> frozen Qwen decoder -> answer.

### Gradient Flow

Frozen Qwen forward passes run inside `torch.no_grad()` context managers. Trainable modules (`DownProj`, `QueryInit`, `GuidedPerceiver`, `UpMLP`) are called **outside** the no-grad context to preserve gradient flow. The teacher path (same frozen Qwen reading full uncompressed document) is inference-only with no gradients.

**Critical**: The encoder calls `self.qwen.model(...)` (the inner transformer), NOT `self.qwen(...)` (which also computes the LM head logits). Using `self.qwen(...)` for encoding wastes ~15GB on unused logits and causes OOM.

### Distillation

Two distillation losses (Stage 2 only):
- **Output distribution**: KL divergence on answer token logits (temperature-scaled).
- **Hidden state alignment**: MSE on shared question+answer token positions across selected decoder layers. Weight ramps linearly from 0 over `hidden_distill_ramp_steps` to avoid early training instability.

### Configuration System

Two paths producing the same validated `DeepCompressorConfig`:
1. **YAML + `DeepCompressorConfig.from_yaml(path)`** — used by all scripts.
2. **Hydra + `RunConf` in `hydra_conf.py`** — used for CLI override workflows.

Config hierarchy: `DeepCompressorConfig` wraps `QwenConfig`, `PerceiverConfig`, `ProjectionConfig`, `LossConfig`, `TrainingConfig`, `AblationConfig`. `__post_init__` validates constraints (`num_heads * head_dim == perceiver_dim`), auto-resolves Qwen3 specs from `QWEN3_REGISTRY`, and forces `perceiver_dim = qwen.hidden_size`.

### Ablation Infrastructure

`AblationConfig` controls experimental switches: projection modes (`down_proj_mode`/`up_proj_mode`: mlp/linear/identity), query conditioning, Perceiver stage enables (`enable_stage_a/b/c`), layer count overrides, distillation toggles, query count override. `DeepCompressorConfig` provides `effective_*` computed properties that resolve overrides.

Factory functions `build_down_proj()` and `build_up_proj()` create projection modules based on ablation mode.

## Implementation Notes

- **Frozen vs trainable**: All Qwen params have `requires_grad=False`. Trainable: QueryInit, GuidedPerceiver; optionally DownProj, UpMLP (only when not identity mode).
- **Checkpoint saving**: Only trainable weights are saved (skips `qwen.*` parameters).
- **Lazy loading**: `NTPDataset` uses byte offsets into JSONL files to avoid loading multi-GB data into memory.
- **Distributed training**: Uses HuggingFace `accelerate` for automatic DDP + device placement.
- **Teacher distillation**: Teacher sees full uncompressed doc+question+answer, student sees compressed prefix+question+answer. Teacher logits/hidden sliced to Q+A region only for loss computation.
- **Evaluation metrics**: Exact match for numeric questions, F1 for overall QA quality. Chinese text normalization splits CJK per-character.

## Known Pitfalls (from past bugs)

- **Encoder must use `self.qwen.model(...)` not `self.qwen(...)`** — the LM head computes 15GB of unused logits, causing OOM.
- **NTP labels: do NOT manually shift** — Qwen's `forward(labels=...)` shifts internally. Double-shifting causes PPL plateau at ~510.
- **Gradient checkpointing requires `use_reentrant=False`** — call `gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})`. Without this kwarg, gradients silently break.
- **MLP projection bottleneck** — `down_proj_mode="mlp"` with `down_hidden=768` creates a 1024→768 bottleneck that blocks convergence. Default is now `identity`.
- **MPS num_workers** — macOS MPS doesn't support multi-process data loading reliably; `num_workers` is forced to 0 on MPS devices.
