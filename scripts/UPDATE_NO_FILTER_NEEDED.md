# 更新说明：无需预先过滤数据

## ✅ 重要变更

**不再需要预先过滤数据！**

数据加载时会根据配置文件中的 `max_doc_tokens` 自动截断。

## 📝 变更内容

### 1. 配置文件保持不变

```yaml
# configs/stage1_q64.yaml
qwen:
  max_doc_tokens: 512  # 自动截断到512 tokens
```

### 2. 直接使用原始数据

```bash
# ❌ 旧方式：需要预先过滤
python scripts/filter_ntp_data.py \
    --input data/ntp_train.jsonl \
    --output data/ntp_train_512.jsonl \
    --max_length 512

# ✅ 新方式：直接使用原始数据
python -m deep_compressor.train \
    --config configs/stage1_q64.yaml \
    --data_path data/ntp_train.jsonl \
    --stage 1
```

### 3. 自动截断机制

NTPDataset 在 `__getitem__` 方法中会自动：
- 使用 tokenizer 的 `truncation=True`
- 截断到 `max_doc_tokens + segment_len` (512 + 256 = 768 tokens)
- 然后分割成文档部分（≤512）和段落部分（256）

## 🚀 更新后的启动方式

### 最简单方式

```bash
# 1. 准备数据（如果还没有）
python scripts/prepare_data.py

# 2. 8卡并行训练（最快）
bash scripts/train_parallel_8gpu.sh

# 3. 评估
python scripts/evaluate_all_checkpoints.py \
    --eval_data data/ntp_train.jsonl \
    --stage 1
```

### 单卡训练

```bash
python -m deep_compressor.train \
    --config configs/stage1_q64.yaml \
    --data_path data/ntp_train.jsonl \
    --stage 1
```

### 8卡DDP训练单个Q值

```bash
accelerate launch \
    --multi_gpu --num_processes 8 \
    -m deep_compressor.train \
    --config configs/stage1_q64.yaml \
    --data_path data/ntp_train.jsonl \
    --stage 1
```

### 批量训练所有Q值

```bash
python scripts/train_stage1_varying_q.py
```

## 💡 优势

1. **更简单** - 无需额外的数据过滤步骤
2. **更灵活** - 修改配置文件即可改变截断长度
3. **节省空间** - 不需要保存多个过滤后的数据副本
4. **更快开始** - 准备数据后立即开始训练

## 🔧 如何修改文档长度

只需修改配置文件：

```yaml
# configs/stage1_q64.yaml
qwen:
  max_doc_tokens: 1024  # 改为1024 tokens
```

无需重新过滤数据！

## 📊 数据分析（可选）

如果想了解数据分布，仍可使用：

```bash
python scripts/analyze_data_distribution.py \
    --data_path data/ntp_train.jsonl
```

## ⚠️ 注意事项

1. **filter_ntp_data.py 脚本仍然保留**，以备特殊需求
2. **配置文件中的 max_doc_tokens 很重要**，决定了截断长度
3. **所有文档已更新**，反映新的使用方式

## 📚 相关文件更新

已更新的文件：
- `scripts/train_stage1_varying_q.py` - 默认使用 ntp_train.jsonl
- `scripts/train_parallel_8gpu.sh` - 检查 ntp_train.jsonl
- `scripts/run_full_experiment.sh` - 移除过滤步骤
- `scripts/evaluate_all_checkpoints.py` - 使用 ntp_train.jsonl

文档说明：
- `scripts/TRAINING_GUIDE.md` - 训练指南（部分已更新）
- `scripts/QUICKSTART_STAGE1_Q_EXPERIMENTS.md` - 快速开始（部分已更新）
- 其他文档会随后更新

## ✅ 立即开始

```bash
# 检查数据
ls -lh data/ntp_train.jsonl

# 启动训练
bash scripts/train_parallel_8gpu.sh

# 或单卡训练
python scripts/train_stage1_varying_q.py
```

**数据会自动截断到512 tokens，无需任何预处理！**
