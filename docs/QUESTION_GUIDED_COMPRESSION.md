# Question-Guided Compression for Stage 1 NTP Training

## Overview

This feature enables **question-guided compression** during Stage 1 NTP pretraining, allowing the model to learn what information to preserve during compression by providing question context.

### Problem Addressed

**Original approach**: Stage 1 trained with zero question vector → model learns generic compression but loses key information (numbers, entities, dates) → Stage 2 struggles to recover this information.

**New approach**: Stage 1 trained with questions → model learns task-aware compression → Stage 2 achieves better performance with the same architecture.

### Evidence of Need

- Loss converges normally (3.15) ✅
- But generated samples are fluent yet irrelevant ❌
- Only 10-15% of samples contain factual information
- Example: Document about "打更制度" → Model generates "台湾茶文化"

---

## Quick Start

### 1. Augment NTP Data with Questions

Generate pseudo-questions for your NTP training data:

```bash
python scripts/augment_ntp_with_questions.py \
    --input data/ntp_train.jsonl \
    --output data/ntp_train_guided.jsonl \
    --qa_source data/qa_train.json \
    --strategy mixed
```

**Question generation strategies**:
- **70% sampled**: Randomly select questions from QA dataset (most relevant)
- **20% extracted**: Use TF-IDF to extract key sentences, convert to questions
- **10% generic**: Fallback templates like "请总结这篇文章"

### 2. Train with Question-Guided Compression

```bash
python -m deep_compressor.train \
    --config configs/stage1_guided_q128.yaml \
    --data_path data/ntp_train_guided.jsonl \
    --stage 1 \
    --wandb --wandb_project deep-compressor-refactor
```

### 3. Compare with Baseline

```bash
python scripts/compare_guided_vs_unguided.py \
    --unguided outputs/stage1_q128/checkpoint-final \
    --guided outputs/stage1_guided_q128/checkpoint-final \
    --eval_data data/ntp_train_512.jsonl \
    --qa_data data/qa_dev.json \
    --num_samples 50
```

---

## Architecture Changes

### Model (`deep_compressor/model.py`)

**Modified `forward_ntp` signature**:
```python
def forward_ntp(
    self,
    doc_input_ids: torch.Tensor,
    doc_attention_mask: torch.Tensor,
    segment_ids: torch.Tensor,
    segment_attention_mask: torch.Tensor,
    segment_labels: torch.Tensor,
    q_input_ids: Optional[torch.Tensor] = None,      # NEW
    q_attention_mask: Optional[torch.Tensor] = None   # NEW
) -> Dict[str, torch.Tensor]:
```

**Behavior**:
- If `q_input_ids` provided → use `encode_question()` for query initialization
- If not provided → fall back to zero vector (backward compatible)

### Data Loading (`deep_compressor/data.py`)

**Extended `NTPDataset`**:
```python
def __init__(
    self,
    data_path: str,
    tokenizer,
    max_doc_tokens: int = 8192,
    segment_len: int = 256,
    seed: int = 42,
    use_questions: bool = False  # NEW
):
```

**Data format**:
```json
{"text": "...", "question": "What is the revenue in Q3 2023?"}
```

If `use_questions=True` and `"question"` field exists, it's tokenized and added to the batch as `q_input_ids`.

### Training Integration (`deep_compressor/train.py`)

**Modified `_build_forward_kwargs`**:
- Checks if `q_input_ids` exists in batch
- Passes it to `forward_ntp` if available

**Modified dataset creation**:
```python
dataset = NTPDataset(
    data_path, tokenizer,
    max_doc_tokens=config.qwen.max_doc_tokens,
    segment_len=tcfg.ntp_segment_len,
    use_questions=config.ablation.ntp_use_questions  # NEW
)
```

### Configuration (`deep_compressor/config.py`)

**Added to `AblationConfig`**:
```python
ntp_use_questions: bool = False
```

---

## Configuration Files

### `configs/stage1_guided_q128.yaml`

Full-scale guided training config:
- `ablation.ntp_use_questions: true`
- `training.output_dir: outputs/stage1_guided_q128`
- `wandb.run_name: stage1_guided_q128`
- `wandb.project: deep-compressor-refactor`

### `configs/stage1_comparison.yaml`

Quick comparison config for A/B testing:
- Smaller scale: `max_steps: 10000`, `eval_every: 500`
- Override `ntp_use_questions` via CLI for guided experiments

---

## Training Workflow

### Experiment 1: Baseline (unguided)

```bash
python -m deep_compressor.train \
    --config configs/stage1_q128.yaml \
    --data_path data/ntp_train.jsonl \
    --stage 1 \
    --wandb --wandb_project deep-compressor-refactor
```

### Experiment 2: Question-guided

```bash
# Step 1: Augment data
python scripts/augment_ntp_with_questions.py \
    --input data/ntp_train.jsonl \
    --output data/ntp_train_guided.jsonl \
    --qa_source data/qa_train.json

# Step 2: Train with questions
python -m deep_compressor.train \
    --config configs/stage1_guided_q128.yaml \
    --data_path data/ntp_train_guided.jsonl \
    --stage 1 \
    --wandb --wandb_project deep-compressor-refactor
```

### Experiment 3: Stage 2 QA Fine-tuning

Use the guided Stage 1 checkpoint for Stage 2:

```bash
python -m deep_compressor.train \
    --config configs/default.yaml \
    --data_path data/qa_train.json \
    --eval_data_path data/qa_dev.json \
    --stage 2 \
    --resume_from outputs/stage1_guided_q128/checkpoint-final \
    --wandb --wandb_project deep-compressor-refactor
```

---

## Evaluation Metrics

### Stage 1 Improvements (Expected)

- **Perplexity**: Guided PPL should be 5-15% lower when question provided
- **Sample quality**: 40-50% samples contain factual information (vs 10-15% baseline)
- **Attention patterns**: Cross-attention focuses on question-relevant regions

### Stage 2 Improvements (Expected)

- **Convergence speed**: Reach target EM/F1 in 30-50% fewer steps
- **Final metrics**: EM/F1 absolute improvement of 3-8 points
- **Distillation**: KL/MSE loss lower by 10-20%

---

## Success Criteria

### Minimum Viable Success

- NTP perplexity improvement > 5%
- Sample factual quality improvement > 10 points
- Stage 2 EM/F1 improvement > 2 points

### Strong Success

- NTP perplexity improvement > 10%
- Sample factual quality improvement > 20 points
- Stage 2 EM/F1 improvement > 5 points

---

## Backward Compatibility

✅ **All changes are backward compatible**:
- `q_input_ids` parameters are optional
- `ntp_use_questions` defaults to `False`
- Existing configs work unchanged
- Old checkpoints still load

---

## Files Modified

### Core Implementation
1. `deep_compressor/model.py` - Added optional question params to `forward_ntp`
2. `deep_compressor/data.py` - Extended `NTPDataset` to load questions
3. `deep_compressor/train.py` - Pass questions in forward kwargs
4. `deep_compressor/config.py` - Added `ntp_use_questions` flag

### New Files
1. `scripts/augment_ntp_with_questions.py` - Question generation script
2. `scripts/compare_guided_vs_unguided.py` - Evaluation comparison script
3. `configs/stage1_guided_q128.yaml` - Config for guided training
4. `configs/stage1_comparison.yaml` - Config for quick experiments
5. `docs/QUESTION_GUIDED_COMPRESSION.md` - This documentation

---

## Troubleshooting

### Data augmentation script fails

**Error**: `No module named 'sklearn'`

**Solution**: Install scikit-learn:
```bash
pip install scikit-learn
```

### Training doesn't use questions

**Check**:
1. Config has `ablation.ntp_use_questions: true`
2. Data file has `"question"` field in each JSONL line
3. No errors in data loading logs

### Perplexity doesn't improve

**Possible causes**:
1. Question quality too generic → increase `qa_source` sampling ratio
2. Training steps too few → increase `max_steps`
3. Model not converged → check loss curves in wandb

---

## Next Steps

### If This Succeeds (EM/F1 improvement > 3 points)

1. **Scale up**: Train on 8K-32K token documents (currently using 512)
2. **Optimize Q value**: May need Q=256-512 for longer documents
3. **Multi-task learning**: Joint NTP+QA loss from Stage 1
4. **FinBERT integration**: Add entity-level guidance
5. **Publish findings**: EMNLP/ACL on question-guided compression

### If This Fails (no improvement after 2 weeks)

**Fallback to Solution B**: Skip Stage 1, direct QA training
- Train compression directly on QA task
- Risk: Losing 2.7M unlabeled NTP data

**Fallback to Solution C**: Curriculum learning
- Start with short docs (256 tokens, Q=256)
- Gradually increase length and compression ratio
- More complex, 2+ weeks implementation

---

## Related Documentation

- `CLAUDE.md` - Main project documentation
- `plan.md` - Original architectural plan (Chinese)
- `docs/EVALUATION_IMPROVEMENTS.md` - Evaluation metrics documentation
- `scripts/STAGE1_VARYING_Q_EXPERIMENTS.md` - Stage 1 Q-value experiments

---

## Citation

If you use this approach in research, please cite:

```bibtex
@misc{deep-compressor-guided-compression,
  title={Question-Guided Compression for Ultra-Long Financial Documents},
  author={Deep Compressor Team},
  year={2026},
  note={Technical Report}
}
```
