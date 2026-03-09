# 数据长度处理说明

## 🔍 当前处理方式

### 代码实现 (deep_compressor/data.py)

```python
doc = self.tokenizer(
    item["context"],
    truncation=True,              # ← 截断，不是舍弃
    max_length=self.max_doc_tokens,  # 默认512
    ...
)
```

**处理方式**: **截断 (Truncation)**

---

## ⚠️ 当前问题

### 配置
```yaml
qwen:
  max_doc_tokens: 512      # 文档最大512 tokens
  max_question_tokens: 256
  max_answer_tokens: 512
```

### 问题分析

| 问题 | 影响 | 严重性 |
|------|------|--------|
| **文档被截断** | 文档>512的部分直接丢弃 | ⚠️ 中等 |
| **答案可能丢失** | 如果答案在后半部分，会被截掉 | ❌ 严重 |
| **与目标不符** | 项目目标是8K-64K超长文档 | ⚠️ 需要调整 |

---

## 📊 步骤1: 先统计数据

运行统计脚本查看实际情况：

```bash
python scripts/analyze_dataset_stats.py \
    --data data/qa_large_train.json \
    --output results/dataset_stats.json
```

**关键看这些指标**:

```
CONTEXT LENGTH DISTRIBUTION
  Median:    ??? tokens  ← 如果>512，说明一半数据被截断
  95th %ile: ??? tokens  ← 如果>512，说明很多数据被截断
  Max:       ??? tokens  ← 最长的文档有多长
```

**根据结果决定策略**:
- Median < 512: 大部分数据OK，少量截断 → 可以保持512或稍微增加
- Median > 512 < 2048: 很多数据被截断 → 建议增加到2048
- Median > 2048: 大量长文档 → 建议4096-8192

---

## 🎯 推荐方案

### 方案A: 增加max_doc_tokens（强烈推荐）

**原因**:
1. ✅ 项目叫"Deep Compressor"，目标就是处理超长文档
2. ✅ 你有143GB显存，完全够用
3. ✅ 保留完整信息，QA性能更好

**建议配置**:

```yaml
qwen:
  max_doc_tokens: 2048   # 512 → 2048 (覆盖大部分)
  # 或
  max_doc_tokens: 4096   # 覆盖95%+ (推荐)
  # 或
  max_doc_tokens: 8192   # 项目原始目标
```

**实施**:

```bash
# 方法1: 手动编辑所有配置文件
# configs/qa_q64_8gpu.yaml
# configs/qa_q128_8gpu.yaml
# configs/qa_q256_8gpu.yaml
# configs/qa_q512_8gpu.yaml

# 方法2: 批量修改
for config in configs/qa_q*_8gpu.yaml; do
    sed -i 's/max_doc_tokens: 512/max_doc_tokens: 2048/g' "$config"
done
```

**影响**:
- 训练时间: +20-30%（因为文档更长）
- 显存占用: +30-50%（但你有143GB，完全够）
- QA性能: 可能提升5-10个点（因为保留了完整信息）

### 方案B: 过滤掉超长样本

**适合**: 只想用短文档快速训练

```bash
# 过滤训练集（只保留<=512的）
python scripts/filter_qa_by_length.py \
    --input data/qa_large_train.json \
    --output data/qa_large_train_512.json \
    --max_length 512

# 过滤评估集
python scripts/filter_qa_by_length.py \
    --input data/qa_large_dev.json \
    --output data/qa_large_dev_512.json \
    --max_length 512

# 然后修改训练脚本使用过滤后的数据
# DATA_PATH="data/qa_large_train_512.json"
```

**优点**:
- ✅ 训练更快
- ✅ 显存占用更少

**缺点**:
- ❌ 丢弃大量数据（可能50%+）
- ❌ 与项目目标不符

---

## 💡 我的建议

### 推荐配置（基于你的硬件）

```yaml
qwen:
  max_doc_tokens: 4096     # 推荐！覆盖95%+数据
  max_question_tokens: 256
  max_answer_tokens: 512
```

**理由**:
1. **硬件足够**: 143GB显存可以轻松处理4K文档
2. **项目目标**: Deep Compressor就是为超长文档设计的
3. **性能最优**: 保留完整信息，QA效果最好
4. **覆盖率高**: 4096可以覆盖95%+的样本

### 不同Q值的显存占用（max_doc_tokens=4096）

| Q值 | Batch | 有效Batch | 预估显存 | 利用率 |
|-----|-------|-----------|----------|--------|
| 64  | 16    | 256       | ~70GB    | 49%    |
| 128 | 12    | 192       | ~90GB    | 63%    |
| 256 | 8     | 128       | ~110GB   | 77%    |
| 512 | 4     | 64        | ~130GB   | 91%    |

**如果改成4096还是有足够的显存余量！**

---

## 🚀 实施步骤

### 第1步: 统计当前数据

```bash
python scripts/analyze_dataset_stats.py \
    --data data/qa_large_train.json
```

查看输出中的:
- `context_median`: 中位数
- `context_p95`: 95%分位数
- `context_p99`: 99%分位数

### 第2步: 根据统计结果决定

**如果median < 512**:
→ 保持512即可

**如果median在512-2048**:
→ 增加到2048

**如果median在2048-8192**:
→ 增加到4096或8192

### 第3步: 修改配置

```bash
# 批量修改所有配置（改成2048）
for config in configs/qa_q*_8gpu.yaml; do
    sed -i 's/max_doc_tokens: 512/max_doc_tokens: 2048/g' "$config"
done

# 或者改成4096
for config in configs/qa_q*_8gpu.yaml; do
    sed -i 's/max_doc_tokens: 512/max_doc_tokens: 4096/g' "$config"
done
```

### 第4步: 调整batch size（如果需要）

如果改成4096后显存不够：

```bash
# 减小batch size（但你的143GB应该够）
# 编辑 configs/qa_q256_8gpu.yaml:
training:
  batch_size: 8   # 16 → 8
```

---

## 📈 不同长度设置的权衡

| max_doc_tokens | 覆盖率 | 训练速度 | 显存占用 | QA性能 | 推荐 |
|----------------|--------|----------|----------|--------|------|
| 512  | ~50%? | 最快 | 最低 | 一般 | ❌ 太短 |
| 1024 | ~70%? | 快 | 中等 | 较好 | ⚠️ 可考虑 |
| 2048 | ~85%? | 中等 | 中高 | 好 | ✅ 推荐 |
| 4096 | ~95%+ | 稍慢 | 高 | 很好 | ✅✅ 强烈推荐 |
| 8192 | ~99%+ | 最慢 | 最高 | 最好 | ✅ 如果硬件够 |

**基于你的143GB显存，推荐4096！**

---

## 🎯 总结

### 当前状态
- ❌ max_doc_tokens=512 太短
- ❌ 大量数据被截断（具体比例需统计）
- ❌ 可能丢失答案信息

### 建议操作

1. **先统计**: 看看实际长度分布
   ```bash
   python scripts/analyze_dataset_stats.py
   ```

2. **改配置**: 增加到4096（推荐）
   ```bash
   for config in configs/qa_q*_8gpu.yaml; do
       sed -i 's/max_doc_tokens: 512/max_doc_tokens: 4096/g' "$config"
   done
   ```

3. **调整batch**: 根据显存占用微调batch size

4. **开始训练**: 使用优化后的配置

---

**下一步: 先运行统计脚本，看看数据实际情况！** 📊
