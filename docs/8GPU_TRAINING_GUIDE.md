# 8-GPU Training Guide

**完整的8卡训练指南** - 训练Q=64, 128, 256, 512四个模型

---

## 📋 准备工作

### 1. 下载数据集

```bash
# 下载大规模QA数据集（~800K样本，1-2小时）
python scripts/prepare_large_qa_data.py
```

**数据集包括**:
- SQuAD v1+v2 (217K)
- TriviaQA (95K)
- Natural Questions (150K)
- HotpotQA (90K)
- CMRC2018, DuReader, DRCD, WebQA (中文)

### 2. 验证环境

```bash
# 检查GPU
nvidia-smi

# 检查accelerate配置
accelerate env

# 快速过拟合测试（确保代码正常）
python scripts/quick_overfit_qa.py --synthetic
```

---

## 🚀 训练方式

### 方式1: 单独训练（推荐，灵活）

训练单个Q值的模型：

```bash
# 训练Q=64（推荐先跑这个测试）
bash scripts/train_qa_q64_8gpu.sh

# 训练Q=128（平衡性能和速度）
bash scripts/train_qa_q128_8gpu.sh

# 训练Q=256（更好的性能）
bash scripts/train_qa_q256_8gpu.sh

# 训练Q=512（最佳性能，需要更多内存）
bash scripts/train_qa_q512_8gpu.sh
```

### 方式2: 批量训练（自动化）

一次性训练所有Q值（按顺序执行）：

```bash
# 训练所有Q值（总计~26小时）
bash scripts/train_all_q_values_8gpu.sh
```

**注意**: 这会**顺序**执行Q=64→128→256→512，不是并行。

---

## ⚙️ 配置详情

### 训练参数对比

| Q值 | 每卡batch | 梯度累积 | 有效batch | GPU内存 | 预计时间 |
|-----|-----------|----------|-----------|---------|----------|
| 64  | 8         | 2        | 128       | ~20GB   | ~5h      |
| 128 | 6         | 3        | 144       | ~30GB   | ~6h      |
| 256 | 4         | 4        | 128       | ~40GB   | ~7h      |
| 512 | 2         | 8        | 128       | ~60GB   | ~8h      |

**有效batch size** = 8 GPUs × 每卡batch × 梯度累积

### 共同参数

- **学习率**: 1e-4
- **Warmup**: 2000步
- **总步数**: 50000步
- **优化器**: AdamW (weight_decay=0.01)
- **调度器**: Cosine with warmup
- **混合精度**: BF16
- **评估频率**: 每1000步
- **保存频率**: 每5000步

---

## 📊 监控训练

### WandB实时监控

所有训练会自动上传到WandB：

- **Project**: `deep-compressor-qa`
- **Run名称**: `qa_q{64,128,256,512}_8gpu_full`

**关键指标**:
- `qa/loss` - 训练loss（越低越好）
- `qa/ppl` - 困惑度（越低越好）
- `eval/exact_match` - 精确匹配率（越高越好）
- `eval/f1` - F1分数（越高越好）

### 本地日志

每个训练都会生成日志文件：

```bash
# 查看训练日志
tail -f outputs/qa_q128_8gpu_training.log

# 查看最近的训练状态
tail -100 outputs/qa_q128_8gpu_training.log
```

---

## 📂 输出结构

```
outputs/
├── qa_q64_8gpu/
│   ├── checkpoint-5000/
│   ├── checkpoint-10000/
│   ├── ...
│   └── checkpoint-final/
│       └── trainable_weights.pt
├── qa_q128_8gpu/
├── qa_q256_8gpu/
└── qa_q512_8gpu/

# 训练日志
outputs/qa_q64_8gpu_training.log
outputs/qa_q128_8gpu_training.log
outputs/qa_q256_8gpu_training.log
outputs/qa_q512_8gpu_training.log
```

---

## 🔍 检查训练状态

### 查看正在运行的训练

```bash
# 查看GPU使用情况
watch -n 1 nvidia-smi

# 查看训练进程
ps aux | grep "deep_compressor.train"

# 查看最新日志
tail -f outputs/*_training.log
```

### 估算剩余时间

```bash
# 查看当前步数（从日志）
grep "step" outputs/qa_q128_8gpu_training.log | tail -1

# 计算: 剩余步数 = 50000 - 当前步数
# 预计时间 = (剩余步数 / 当前步数) × 已用时间
```

---

## 🛠️ 常见问题

### 1. 内存不足 (OOM)

**Q=512特别容易OOM**

**解决方案**:

```bash
# 方法1: 启用gradient checkpointing（已在Q=512配置中启用）
# 在configs/qa_q512_8gpu.yaml中确认:
# training:
#   gradient_checkpointing: true

# 方法2: 减小batch size（如果还是OOM）
# 编辑 scripts/train_qa_q512_8gpu.sh:
BATCH_SIZE=1           # 降到1
GRAD_ACCUM=16          # 增加累积步数保持有效batch=128
```

### 2. 训练中断恢复

训练会自动保存checkpoint，但默认不支持自动恢复。如需恢复：

```bash
# 找到最新checkpoint
ls -lt outputs/qa_q128_8gpu/

# 手动恢复训练（修改脚本，添加--resume_from）
# 暂不支持自动恢复，需要重新训练
```

### 3. WandB上传失败

```bash
# 离线模式
export WANDB_MODE=offline

# 或在脚本中移除--wandb参数
```

### 4. 检查数据是否正确

```bash
# 检查数据文件
ls -lh data/qa_large_*.json

# 查看样本数量
python -c "import json; print(len(json.load(open('data/qa_large_train.json'))))"
```

---

## 📈 预期结果

基于~800K训练样本的预期性能（在评估集上）：

| Q值 | 压缩比 | EM（预期） | F1（预期） |
|-----|--------|------------|------------|
| 64  | ~8:1   | 40-50%     | 0.50-0.60  |
| 128 | ~4:1   | 50-60%     | 0.60-0.70  |
| 256 | ~2:1   | 60-70%     | 0.70-0.80  |
| 512 | ~1:1   | 65-75%     | 0.75-0.85  |

**注**: 实际结果会因数据质量、训练时长等因素波动。

---

## 🎯 训练完成后

### 1. 评估所有模型

```bash
python scripts/evaluate_all_checkpoints.py \
    --eval_data data/qa_large_dev.json \
    --stage 2 \
    --output results/qa_8gpu_results.csv
```

### 2. 分析结果

```bash
# 查看结果CSV
cat results/qa_8gpu_results.csv

# 或用pandas分析
python -c "
import pandas as pd
df = pd.read_csv('results/qa_8gpu_results.csv')
print(df.sort_values('exact_match', ascending=False))
"
```

### 3. 选择最优模型

根据EM/F1和压缩比权衡选择：

- **高性能**: Q=256 或 Q=512
- **平衡**: Q=128
- **快速推理**: Q=64

---

## 💡 优化建议

### 提高训练速度

1. **使用更大的有效batch size**
   ```bash
   # 编辑配置文件
   # training.batch_size: 16  # 如果GPU内存足够
   # training.gradient_accumulation_steps: 1
   ```

2. **减少评估频率**
   ```bash
   # training.eval_every: 2000  # 改为2000步评估一次
   ```

3. **使用更少的保存点**
   ```bash
   # training.save_every: 10000  # 减少IO开销
   ```

### 提高模型性能

1. **增加训练步数**
   ```bash
   # training.max_steps: 100000  # 训练更久
   ```

2. **调整学习率**
   ```bash
   # training.learning_rate: 2e-4  # 尝试更大的学习率
   ```

3. **使用更长的warmup**
   ```bash
   # training.warmup_steps: 5000  # 更平滑的学习
   ```

---

## 🚦 训练状态指示

### 正常训练的标志

✅ Loss持续下降
✅ PPL持续下降
✅ eval/exact_match持续上升
✅ GPU利用率>80%

### 可能有问题的标志

⚠️ Loss不下降或震荡
⚠️ eval/exact_match不增长
⚠️ GPU利用率<50%
❌ 训练崩溃/OOM

---

## 📞 支持

遇到问题？

1. **检查日志**: `tail -100 outputs/*_training.log`
2. **检查GPU**: `nvidia-smi`
3. **检查数据**: 确保`data/qa_large_*.json`存在
4. **重新过拟合测试**: `python scripts/quick_overfit_qa.py --synthetic`

---

## 📚 相关文档

- **纯QA训练**: [`QUICKSTART_QA_ONLY_TRAINING.md`](./QUICKSTART_QA_ONLY_TRAINING.md)
- **项目文档**: [`CLAUDE.md`](../CLAUDE.md)
- **评估指南**: [`EVALUATION_IMPROVEMENTS.md`](./EVALUATION_IMPROVEMENTS.md)

---

**开始训练！** 🚀

```bash
# 推荐流程
bash scripts/train_qa_q128_8gpu.sh
```
