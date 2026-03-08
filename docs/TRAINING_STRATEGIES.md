# Stage 2 Training Strategies Comparison

## TL;DR

**Recommended: Single-stage long-sequence training** (`train_h200_stage2_long_optimized.sh`)

Perceiver is designed to handle long sequences efficiently. The three-stage curriculum (short→medium→long) is unnecessary overhead.

---

## Strategy Comparison

### ❌ Original: Three-Stage Curriculum (Not Recommended)

```bash
Stage 2a (short):  512 tokens  → 5K steps → outputs/h200_stage2a/
Stage 2b (medium): 2048 tokens → 3K steps → outputs/h200_stage2b/
Stage 2c (mixed):  0-2048 tokens → 5K steps → outputs/h200_stage2c/
```

**Problems:**
- ❌ **Complexity**: Manage 3 checkpoints, 3 training runs
- ❌ **Time overhead**: Checkpoint loading between stages
- ❌ **Limited coverage**: 2048 tokens only covers 62% of data
- ❌ **Unnecessary**: Perceiver handles long sequences naturally

**Why it seemed reasonable:**
- Historical practice from Transformer curriculum learning
- But Perceiver's cross-attention is **linear** in sequence length!

---

### ✅ Recommended: Single-Stage Long-Sequence

```bash
Stage 2 (long): 4096 tokens → 10K steps → outputs/h200_stage2_long/
```

**Config:** `configs/h200_stage2_long_optimized.yaml`
**Script:** `scripts/train_h200_stage2_long_optimized.sh`

**Advantages:**
- ✅ **Simple**: One training run, one checkpoint
- ✅ **Fast**: No stage transitions, no reload overhead
- ✅ **High coverage**: 4096 tokens covers ~92% of QA data
- ✅ **Natural**: Perceiver is designed for this
- ✅ **Proven**: Perceiver paper used similar approach

**Memory feasibility:**
```
Qwen encoder:  O(doc_len²)  ~35GB for 4096 tokens (Flash Attention)
Perceiver:     O(Q × L)     ~5GB (64 queries × 4096 tokens)
Gradients:     ~20GB
Total:         ~60GB per H200 80GB ✅ Feasible
```

**Training time:**
- 10K steps × 8 sec/step ÷ 3600 = **~22 hours** on 8×H200
- vs. three-stage: 13K steps × 8 sec/step = **~29 hours** + overhead

---

### 🚀 Advanced: Ultra-Long Sequence (8192 tokens)

```bash
Stage 2 (ultra): 8192 tokens → 8K steps
```

**Config:** `configs/h200_stage2_ultralong.yaml`

**For experts who:**
- Have verified 4096-token training works stably
- Want maximum coverage (99%+ of data)
- Have H200/A100 80GB with headroom

**Memory warning:**
- ~73GB per GPU (very tight on H200 80GB)
- Requires `batch_size=1`, high gradient accumulation
- If OOM: drop to 6144 tokens (still covers 97%)

---

## Why Perceiver Handles Long Sequences

### Memory Complexity

| Component | Complexity | 512 tokens | 4096 tokens | 8192 tokens |
|-----------|-----------|-----------|-------------|-------------|
| **Qwen encoder** | O(L²) | 0.26M | 16.8M | 67M |
| **Perceiver cross-attn** | O(Q×L) | 32K | 256K | 512K |
| **Perceiver self-attn** | O(Q²) | 4K | 4K | 4K |

**Key insight**: Perceiver self-attention is **independent of document length**!

### Perceiver Design

```python
# Input: Any length
byte_array: (batch, doc_len, dim)  # Can be 512, 4096, or 64K

# Fixed compression
queries: (batch, 64, dim)  # Always 64 queries

# Cross-attention: Linear in doc_len
latent = cross_attn(queries, byte_array)  # O(64 × doc_len)

# Self-attention: Constant
latent = self_attn(latent)  # O(64²) - doesn't grow!

# Output: Fixed length
output: (batch, 64, dim)
```

**The entire point of Perceiver** is to compress arbitrary-length inputs to fixed-length representations efficiently.

---

## Empirical Data Coverage

| max_doc_tokens | QA data coverage | Effective batch | Steps for 3 epochs |
|----------------|------------------|-----------------|-------------------|
| 512 (original) | 42% | 64 | 30K |
| 2048 (2b) | 62% | 32 | 25K |
| 4096 (recommended) | 92% | 16 | 15K |
| 8192 (advanced) | 99% | 8 | 12K |

**Insight**: Longer documents = higher coverage = fewer samples needed = fewer steps!

---

## Migration Guide

### If you already started three-stage training:

**Option 1**: Abandon and restart with single-stage
```bash
# Use existing Stage 1 checkpoint
bash scripts/train_h200_stage2_long_optimized.sh
```

**Option 2**: Continue three-stage but simplify
```bash
# Skip 2a, start directly from 2b with longer docs
bash scripts/train_h200_stage2b_long.sh  # Already updated with max_eval_samples
```

### Fresh start (recommended):

```bash
# Stage 1: NTP pretraining (already done)
bash scripts/train_h200_stage1.sh

# Stage 2: Single-stage long-sequence QA
bash scripts/train_h200_stage2_long_optimized.sh
```

---

## Troubleshooting

### OOM on 4096 tokens?

1. **Check your GPU memory:**
   ```bash
   nvidia-smi  # Should show 80GB available per GPU
   ```

2. **Reduce batch size:**
   ```yaml
   training:
     batch_size: 1  # Down from 2
     gradient_accumulation_steps: 8  # Up from 4
   ```

3. **Verify gradient checkpointing:**
   ```yaml
   training:
     gradient_checkpointing: true  # Must be enabled
   ```

4. **Try shorter docs first:**
   - Start with 2048 tokens (h200_stage2b_long.yaml)
   - Verify stability
   - Gradually increase to 4096

### Training unstable?

- Check warmup is sufficient (500 steps)
- Verify learning rate isn't too high (5e-5 is safe)
- Monitor gradient norms (should be < 5.0)

---

## Conclusion

**For most users**: Use `train_h200_stage2_long_optimized.sh` with 4096 tokens.

**Key takeaway**: Perceiver's architecture makes it naturally suited for long sequences. Multi-stage curriculum learning is unnecessary complexity carried over from standard Transformer practices.

The three-stage approach (2a/2b/2c) was a conservative design choice, but in practice, Perceiver can jump directly to long documents efficiently.
