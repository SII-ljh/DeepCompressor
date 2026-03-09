# 训练问题诊断和解决方案

## 🚨 问题描述

### 观察到的现象

1. **Overfitting实验**: PPL可以降到1 ✅
2. **全量训练**: 1 epoch后PPL稳定在30+，不下降 ❌
3. **样本预测**: 生成重复文本 "以'打更'为名，以'打更'为名..." ❌

### 分析

- ✅ **模型容量足够**：能在小数据上过拟合到PPL=1
- ❌ **没有充分学习**：全量数据上PPL高，生成质量差
- ❌ **重复生成**：说明模型陷入了重复模式

## 🔍 根本原因

### 1. 训练严重不足 ⭐⭐⭐

**当前**: 50k steps = 1.17 epoch
**问题**: 272万样本，1个epoch远远不够

**证据**:
- Overfitting能到PPL=1 → 模型能力OK
- 全量只训练1 epoch → 每个样本只见过1次
- 大规模预训练通常需要5-20 epoch

### 2. 压缩比可能过大 ⭐⭐

**当前**: 512 tokens → 64 queries (8倍压缩)
**问题**: 对于中文文本，512 tokens压缩到64可能信息损失大

**建议**: 先试Q=128或Q=256（压缩比4倍或2倍）

### 3. 学习率可能偏小 ⭐

**当前**: 1e-4
**对比**: 预训练通常用1e-4到5e-4

## 💡 解决方案（按优先级）

### 🥇 方案1：大幅增加训练步数（强烈推荐）

```bash
# 修改配置文件
vim configs/stage1_q64.yaml

# 改为:
training:
  max_steps: 400000  # 从50k改为400k (~10 epoch)
  # 或更激进
  max_steps: 800000  # ~20 epoch
```

**预期效果**:
- 50k steps (1 epoch): PPL ~30-35
- 200k steps (5 epoch): PPL ~15-20
- 400k steps (10 epoch): PPL ~10-15
- 800k steps (20 epoch): PPL ~8-12

**时间成本**:
- 400k steps: 约8-16小时 (8卡DDP)
- 800k steps: 约16-32小时

### 🥈 方案2：先用大Q值验证

```bash
# 只修改Q=128和Q=256的配置
for q in 128 256; do
  sed -i 's/max_steps: 50000/max_steps: 200000/g' configs/stage1_q${q}.yaml
done

# 训练Q=128（压缩比更小，应该更容易收敛）
bash scripts/train_q128_8gpu.sh
```

如果Q=128能降到PPL~15，说明是压缩比问题。

### 🥉 方案3：调整学习率

```bash
# 修改所有配置
for q in 16 32 64 128 256; do
  # 备份
  cp configs/stage1_q${q}.yaml configs/stage1_q${q}.yaml.bak
done

# 手动编辑（或用sed）
vim configs/stage1_q64.yaml
# 改为：
#   learning_rate: 2.0e-4  # 加倍
#   warmup_steps: 2000     # 延长warmup
```

### 🏅 方案4：组合拳（推荐用于正式实验）

```bash
# 1. 增加训练量 + 2. 调整学习率
vim configs/stage1_q64.yaml

修改为:
training:
  learning_rate: 2.0e-4      # 从1e-4改为2e-4
  max_steps: 400000          # 从50k改为400k
  warmup_steps: 2000         # 从1k改为2k
  eval_every: 2000           # 从500改为2000（减少eval开销）
  save_every: 10000          # 从1000改为10000（减少IO）
```

## 🔬 诊断步骤

### Step 1: 检查当前训练是否还在下降

```bash
# 查看最近的loss
grep "NTP\]" logs/q64_*.log | tail -50

# 看loss趋势
grep "step.*loss=" logs/q64_*.log | awk '{print $4, $5}' | tail -100
```

如果loss还在下降 → 继续训练
如果loss已经平稳 → 可能需要调整其他参数

### Step 2: 对比不同Q值的表现

```bash
# 看看是否所有Q值都卡在30+
grep "EVAL.*perplexity" logs/q*_*.log | tail -20
```

如果所有Q值都卡住 → 训练不足或学习率问题
如果只有小Q值卡住 → 压缩比问题

### Step 3: 检查数据质量

```bash
# 看看数据是否有问题
head -10 data/ntp_train.jsonl
wc -l data/ntp_train.jsonl
```

## 📊 预期的训练曲线

正常的NTP预训练应该是：

```
Steps    PPL     状态
─────────────────────────
1k       500+    初始随机
5k       100-200 开始学习
10k      50-80   快速下降
25k      30-40   你现在在这里 ✗
50k      20-30   应该到这里 ✓
100k     15-20
200k     10-15   充分训练 ✓✓
400k     8-12    接近收敛 ✓✓✓
```

你现在50k步PPL=33，说明**学习速度正常**，但**训练量不够**。

## 🎯 立即行动建议

### 方案A：快速验证（推荐先做这个）

```bash
# 1. 继续训练Q=64到100k步
# 手动编辑配置
vim configs/stage1_q64.yaml
# 改 max_steps: 100000

# 2. 从checkpoint继续
bash scripts/train_q64_8gpu.sh
# 会自动从50k步继续训练到100k步

# 3. 看PPL是否下降
# 如果100k步能降到PPL~20，说明只是训练不够
```

### 方案B：完整重训（如果时间充足）

```bash
# 批量修改所有配置为400k steps
for q in 16 32 64 128 256; do
  sed -i 's/max_steps: 50000/max_steps: 400000/g' configs/stage1_q${q}.yaml
  sed -i 's/eval_every: 500/eval_every: 2000/g' configs/stage1_q${q}.yaml
  sed -i 's/save_every: 1000/save_every: 10000/g' configs/stage1_q${q}.yaml
done

# 删除旧checkpoint，重新训练
rm -rf outputs/stage1_q*/
bash scripts/train_all_q_sequential.sh
```

## 📝 我的建议

**优先级排序**:

1. **立即**: 修改Q=64配置为200k steps，继续训练 → 验证假设
2. **如果有效**: 批量修改所有Q值为200k-400k steps
3. **同时**: 试试Q=128和Q=256（压缩比更小）
4. **观察**: 看PPL曲线，如果还在下降就继续训练

需要我帮你批量修改配置文件吗？
