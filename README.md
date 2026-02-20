# DeepCompressor

Compress ultra-long financial documents (8K-64K tokens) into fixed-length latent vector sequences that serve as prefixes for a frozen Qwen decoder to answer detail-level financial questions. Teacher distillation ensures the compressed prefix carries information equivalent to the full document.

## Architecture

```
Document ──► Frozen Qwen ──► DownProj ──► byte_array
                                              │
Question ──► Frozen Qwen ──► QueryInit ──► queries
                                              │
                              GuidedPerceiver(queries, byte_array)
                                              │
                                          latent_array
                                              │
                                            UpMLP
                                              │
                                        prefix_embeds
                                              │
                              Frozen Qwen Decoder(prefix + question) ──► Answer
```

**Training strategy:**
- **Stage 1 (NTP Pretraining):** Compress document, predict next tokens on random segments. No labels needed.
- **Stage 2 (QA Fine-tuning + Distillation):** Joint training with QA cross-entropy + KL distillation + hidden-state MSE alignment.

**Optional FinBERT enhancement:** Entity-guided attention with FinBERT NER anchors (default OFF).

## Project Structure

```
DeepCompressor/
├── deep_compressor/          # Core package
│   ├── config.py             # Configuration dataclasses
│   ├── hydra_conf.py         # Hydra structured config + conversion
│   ├── model.py              # DeepCompressor main model
│   ├── train.py              # Training script (argparse + Hydra + wandb)
│   ├── data.py               # NTP and QA datasets + collator
│   ├── eval.py               # Evaluation (EM, F1, perplexity)
│   ├── loss.py               # Distillation + total loss
│   └── modules/              # Trainable modules
│       ├── down_proj.py      # Qwen dim → Perceiver dim
│       ├── up_mlp.py         # Perceiver dim → Qwen dim
│       ├── query_init.py     # Learnable queries + question bias
│       ├── perceiver.py      # 3-stage guided Perceiver
│       ├── anchor_align.py   # FinBERT → Perceiver alignment
│       ├── ner_head.py       # NER classification head
│       ├── fact_decode_head.py
│       └── tokenizer_align.py
├── configs/
│   ├── default.yaml          # Full training config
│   ├── macbook_debug.yaml    # MacBook quick validation
│   └── tiny_subset.yaml      # HP tuning on small data
├── scripts/
│   ├── prepare_data.py       # Download model + datasets
│   ├── hp_search.py          # Optuna hyperparameter search
│   ├── diagnostic.py         # Overfitting + bottleneck diagnostics
│   └── visualize_architecture.py
├── tests/                    # Unit + integration tests
├── FinBERT/                  # Cloned reference repo (valuesimplex/FinBERT2)
├── plan.md                   # Detailed technical plan (Chinese)
├── CLAUDE.md                 # AI assistant instructions
└── requirements.txt          # New dependencies (wandb, hydra, optuna)
```

## Setup

### 1. Create conda environment

```bash
conda create --name FinBERT python=3.11
conda activate FinBERT
pip install -r FinBERT/requirements.txt
pip install -r requirements.txt
```

### 2. Download model and data

```bash
# Full download (model + all datasets, ~1-2 hours)
python scripts/prepare_data.py

# Quick test subset only
python scripts/prepare_data.py --test

# Extract tiny subset from existing data (for HP tuning)
python scripts/prepare_data.py --make-tiny

# Skip model download (if already have Qwen3-0.6B locally)
python scripts/prepare_data.py --skip-model
```

This downloads:
- **Model:** Qwen3-0.6B → `models/Qwen3-0.6B/`
- **NTP data (Stage 1):** WikiText-103, SQuAD contexts, C4, CLUECorpusSmall → `data/ntp_train.jsonl`
- **QA data (Stage 2):** SQuAD, CMRC2018, DuReader, TriviaQA, DRCD → `data/qa_train.json`, `data/qa_dev.json`

## Testing

```bash
# Run all unit tests (fast, no model loading)
pytest tests/ -v

# Run including slow integration tests
pytest tests/ -v --runslow

# Run diagnostic experiments (overfitting + information bottleneck)
python scripts/diagnostic.py \
    --config configs/tiny_subset.yaml \
    --data_path data/ntp_train.jsonl \
    --steps 300
```

## Training

### Stage 1: NTP Pretraining

```bash
# Single GPU / MPS / CPU (auto-detected)
python -m deep_compressor.train \
    --config configs/default.yaml \
    --data_path data/ntp_train.jsonl \
    --stage 1

# MacBook quick validation
python -m deep_compressor.train \
    --config configs/macbook_debug.yaml \
    --data_path data/ntp_train.jsonl \
    --stage 1

# Multi-GPU with Accelerate
accelerate launch --multi_gpu --num_processes 4 \
    -m deep_compressor.train \
    --config configs/default.yaml \
    --data_path data/ntp_train.jsonl \
    --stage 1
```

### Stage 2: QA Fine-tuning + Distillation

```bash
python -m deep_compressor.train \
    --config configs/default.yaml \
    --data_path data/qa_train.json \
    --eval_data_path data/qa_dev.json \
    --stage 2 \
    --resume_from outputs/checkpoint-final
```

### With wandb experiment tracking

```bash
# Append --wandb flags to any training command
python -m deep_compressor.train \
    --config configs/default.yaml \
    --data_path data/ntp_train.jsonl \
    --stage 1 \
    --wandb --wandb_project deep-compressor

# Offline mode (no internet required)
python -m deep_compressor.train \
    --config configs/tiny_subset.yaml \
    --data_path data/ntp_train.jsonl \
    --stage 1 \
    --wandb --wandb_offline
```

### With Hydra configuration overrides

```bash
python -m deep_compressor.train \
    --config-path ../configs --config-name default \
    data_path=data/ntp_train.jsonl \
    training.learning_rate=1e-3 \
    training.max_steps=100 \
    perceiver.dropout=0.2 \
    wandb.enabled=true wandb.run_name=experiment-1
```

## Hyperparameter Search

```bash
# Stage 1 (NTP) — minimize loss
python scripts/hp_search.py --n_trials 50 --stage 1 \
    --data_path data/ntp_train.jsonl

# Stage 2 (QA) — maximize F1
python scripts/hp_search.py --n_trials 30 --stage 2 \
    --data_path data/qa_train.json \
    --eval_data_path data/qa_dev.json

# With persistent storage (resumable)
python scripts/hp_search.py --n_trials 50 --stage 1 \
    --data_path data/ntp_train.jsonl \
    --storage sqlite:///outputs/hp_search/study.db
```

Search space: learning_rate, warmup_steps, weight_decay, perceiver_dropout, projection_dropout (+ kl_temperature, kl_weight, hidden_mse_weight for Stage 2).

## Verification

Quick smoke tests to verify everything works:

```bash
# 1. Legacy mode (backward compatible)
python -m deep_compressor.train \
    --config configs/tiny_subset.yaml \
    --data_path data/ntp_train.jsonl \
    --stage 1 --max_train_samples 10

# 2. wandb offline mode
python -m deep_compressor.train \
    --config configs/tiny_subset.yaml \
    --data_path data/ntp_train.jsonl \
    --stage 1 --wandb --wandb_offline --max_train_samples 10

# 3. Optuna (3 trials, fast)
python scripts/hp_search.py --n_trials 3 --stage 1 \
    --data_path data/ntp_train.jsonl \
    --config configs/tiny_subset.yaml
```

## Key Dependencies

- Python 3.11
- PyTorch >= 2.0
- Transformers >= 4.40
- Accelerate >= 0.20
- wandb >= 0.16
- Hydra >= 1.3
- Optuna >= 3.5

See `FinBERT/requirements.txt` for core ML dependencies and `requirements.txt` for experiment tracking / config management.

## License

This project is for research purposes.
