# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Deep Compressor** — a system for compressing ultra-long financial texts (8K–64K tokens) into fixed-length latent vector sequences that serve as prefixes for a frozen Qwen decoder to answer detail-level financial questions. Teacher distillation ensures the compressed prefix carries information equivalent to the full document. FinBERT entity guidance is an optional enhancement module (default OFF).

The architectural plan lives in `plan.md`. The `FinBERT/` subdirectory is a cloned reference repo ([valuesimplex/FinBERT2](https://github.com/valuesimplex/FinBERT2)) used for NER and entity-level features.

## Environment Setup

```bash
conda create --name FinBERT python=3.11
conda activate FinBERT
pip install -r FinBERT/requirements.txt
```

The conda env is at `/opt/homebrew/anaconda3/envs/FinBERT`. Use its Python directly when needed:
```bash
/opt/homebrew/anaconda3/envs/FinBERT/bin/python <script>
```

## Key Dependencies

torch ≥2.0, transformers ≥4.40, datasets ≥2.20, sentence-transformers ≥3.0, accelerate ≥0.20, sentencepiece, jieba, bertopic, scikit-learn, pandas, numpy.

## Architecture (from plan.md)

The data flow has five stages, all of which need to be implemented:

1. **Document Encoding** — Frozen Qwen encoder → hidden states → `DownProj` (2-layer MLP) → `byte_array` in Perceiver dimension.
2. **Entity Guidance (optional)** — Frozen FinBERT + NER Head → `anchor_embs` (via `AnchorAlign`) + `anchor_scores` (per-token entity probability). Requires character-level span overlap for FinBERT↔Qwen tokenizer alignment.
3. **Query Encoding** — Frozen Qwen encoder → mean-pooled question vector → `QueryInit` (learnable base queries + question additive bias) → initial query vectors.
4. **Compression** — `GuidedPerceiver` with three stages:
   - Stage A: global cross-attention from `byte_array` (with optional `anchor_scores` bias) + self-attention.
   - Stage B: anchor refinement cross-attention (FinBERT ON) or extra self-attention (FinBERT OFF).
   - Stage C: deep reasoning cross-attention back to `byte_array` + multi-layer self-attention → fixed-length `latent_array`.
5. **Generation** — `UpMLP` (2-layer MLP) maps `latent_array` back to Qwen dimension → prefix + question tokens → frozen Qwen decoder → answer.

### Distillation

Teacher is the same frozen Qwen reading the full uncompressed document. Two distillation losses:
- **Output distribution**: KL divergence on answer token logits (temperature-scaled).
- **Hidden state alignment**: MSE on shared question+answer token positions across selected decoder layers.

### Training Strategy

- **Stage 1 (NTP Pretraining)**: compress document → prefix → next-token prediction on random document segments. No labels needed. Solves cold-start problem.
- **Stage 2 (QA Fine-tuning + Distillation)**: joint training with QA cross-entropy + KL distillation + hidden-state MSE (weight ramped from 0). Uses SQuAD, CMRC2018, DuReader, etc.

### FinBERT Optional Toggle

Controlled by a config flag. When OFF: no FinBERT/NER/AnchorAlign modules, Stage B degrades to self-attention, no anchor_scores bias, no auxiliary anchor reconstruction loss. System is fully functional without FinBERT.

## FinBERT Reference Repo Structure

```
FinBERT/
├── Fin-NER/              # NER fine-tuning & inference (relevant for anchor extraction)
│   ├── finetune_ner.py   # Training with entity-level metrics (BIO tags)
│   ├── ner_dataset.py    # Span→BIO conversion, character-level subword alignment
│   └── ner_inference.py  # Entity extraction pipeline
├── Fin-labeler/          # Sentiment classification fine-tuning
├── Fin-retriever/        # Contrastive learning retrieval (uses FlagEmbedding)
├── Fin-Topicmodel/       # BERTopic-based topic modeling
└── FinBERT2/pretrain/    # MLM pretraining (torchrun, bf16, whole-word masking)
```

Entity types used in Fin-NER: ORG, PER, METRIC, VALUE, DATE, EVENT, PRODUCT.

## Running FinBERT Tasks

```bash
# MLM pretraining
cd FinBERT/FinBERT2/pretrain && sh run_mlm.sh

# NER fine-tuning
python FinBERT/Fin-NER/finetune_ner.py

# Sentiment classification
cd FinBERT/Fin-labeler && sh runclassify.sh

# NER inference
python FinBERT/Fin-NER/ner_inference.py
```

## Implementation Notes

- All frozen models (Qwen encoder/decoder, FinBERT) must have `requires_grad=False`. Only trainable modules: DownProj, QueryInit, GuidedPerceiver, UpMLP, and optionally AnchorAlign/NER Head/FactDecodeHead.
- Teacher path is inference-only (no gradients). For 64K inputs, pre-compute and cache Teacher logits/hidden states to manage GPU memory.
- Hidden-state distillation weight should ramp gradually from 0 to avoid early training instability.
- Tokenizer alignment between FinBERT and Qwen uses character-level span overlap (max aggregation) — see `ner_dataset.py` for the BIO-to-subword alignment pattern.
- Evaluation metrics: exact match for numeric questions, F1 for overall QA quality.
