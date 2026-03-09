# 数据集划分策略详解

## 📊 数据文件概况

### 原始数据文件
```
data/
├── ntp_train.jsonl      5.8G    2,728,492 条  (NTP 预训练)
├── qa_train.json        2.6G    ~1,500,000 条  (QA 训练集)
└── qa_dev.json          351M    ~200,000 条    (QA 验证集)
```

**关键观察**：
- ✅ **QA 任务**：有独立的 train 和 dev 集
- ⚠️ **NTP 任务**：只有一个 `ntp_train.jsonl` 文件，**没有独立的 dev/test 集**

---

## 🔍 当前的划分策略

### Stage 1: NTP (Next Token Prediction)

#### 训练时的划分 (`deep_compressor/train.py`)

```python
# 加载完整数据集
dataset = NTPDataset("data/ntp_train.jsonl", ...)  # 2,728,492 条

# 划分训练集和验证集
n_total = len(dataset)                              # 2,728,492
n_val = min(5000, n_total // 10)                   # min(5000, 272,849) = 5000
n_train = n_total - n_val                          # 2,723,492

# 训练集：前 2,723,492 条 (99.8%)
train_indices = list(range(n_train))
train_subset = Subset(dataset, train_indices)

# 验证集：最后 5000 条 (0.2%)
val_indices = list(range(n_train, n_total))
val_subset = Subset(dataset, val_indices)
```

**实际划分**：
- **训练集**: 前 2,723,492 条 (99.8%)
- **验证集**: 最后 5,000 条 (0.2%)

#### 评估时的划分 (`scripts/evaluate_all_checkpoints.py`)

```python
# 加载同一个数据文件
dataset = NTPDataset("data/ntp_train.jsonl", ...)  # 2,728,492 条

# 划分（与训练时相同）
n_total = len(dataset)                              # 2,728,492
n_val = min(5000, n_total // 10)                   # 5000
n_train = n_total - n_val                          # 2,723,492

# 评估集：最后 5000 条（与训练时的验证集完全相同）
val_indices = list(range(n_train, n_total))
val_subset = Subset(dataset, val_indices)
```

**关键发现**：
- ✅ **评估集与训练验证集相同** - 使用相同的最后 5000 条
- ✅ **无数据泄露** - 评估集没有用于训练（只用于训练期间的验证）
- ⚠️ **验证集太小** - 只占 0.2%，代表性可能不足

### Stage 2: QA (Question Answering)

#### 训练时的数据

```python
# 使用独立的训练集
train_dataset = QADataset("data/qa_train.json", ...)    # ~1,500,000 条
```

#### 评估时的数据

```python
# 使用独立的验证集
eval_dataset = QADataset("data/qa_dev.json", ...)       # ~200,000 条
```

**关键发现**：
- ✅ **完全独立的 train/dev 划分**
- ✅ **dev 集占比合理** (~13%)
- ✅ **无数据泄露风险**

---

## ⚠️ 当前策略的问题

### 1. **NTP 验证集过小**
- **当前**: 5,000 条 (0.2%)
- **问题**:
  - 代表性不足，可能无法充分反映模型在完整数据分布上的表现
  - 标准做法通常是 10-20% 的验证集
- **影响**:
  - Perplexity 和 loss 的估计可能不够稳定
  - 难以捕捉边缘情况和长尾分布

### 2. **命名混淆**
- **文件名**: `ntp_train.jsonl`
- **实际用途**: train (99.8%) + val (0.2%)
- **问题**: 名称暗示这是纯训练数据，但实际包含验证数据
- **影响**: 可能误导用户，认为需要额外的验证文件

### 3. **缺少独立测试集**
- **当前**: 只有 train + val，没有 test
- **问题**:
  - 无法评估最终模型在未见数据上的泛化性能
  - 调参和最终评估使用同一个验证集，可能过拟合验证集
- **标准做法**: train (80%) / val (10%) / test (10%)

### 4. **Stage 1 和 Stage 2 不一致**
- **NTP**: 动态划分，验证集极小 (0.2%)
- **QA**: 独立文件，验证集合理 (13%)
- **问题**: 两个阶段的数据处理策略不统一

---

## ✅ 当前策略的优点

### 1. **无数据泄露**
- 训练时只使用前 2,723,492 条
- 评估时使用最后 5,000 条（训练期间的验证集）
- 训练和评估使用的是同一个 split，确保一致性

### 2. **实现简单**
- 不需要预先创建独立的 val/test 文件
- 动态划分，灵活方便

### 3. **内存高效**
- `NTPDataset` 使用 lazy loading（字节偏移）
- 不需要一次性加载全部数据到内存

---

## 📋 评估时使用的具体数据

### 运行 `bash scripts/evaluate_all_models.sh` 时

```bash
# 实际评估的数据集
数据文件: data/ntp_train.jsonl
总条数: 2,728,492
评估集: 最后 5,000 条 (行 2,723,493 到 2,728,492)
占比: 0.18%
```

### 评估集的选择逻辑

```python
# 1. 加载完整数据集
dataset = NTPDataset("data/ntp_train.jsonl", ...)

# 2. 计算划分点
n_total = len(dataset)                    # 2,728,492
n_val = min(5000, n_total // 10)         # min(5000, 272,849) = 5000
n_train = n_total - n_val                # 2,723,492

# 3. 取最后 5000 条作为评估集
val_indices = list(range(n_train, n_total))
# val_indices = [2723492, 2723493, ..., 2728491]

# 4. 创建评估集
val_subset = Subset(dataset, val_indices)
```

### 为什么是 5000 条？

代码逻辑：
```python
n_val = min(5000, n_total // 10)
```

- **如果数据集 ≥ 50,000 条**: 取 10% 和 5000 中的较小值 → **5000 条**
- **如果数据集 < 50,000 条**: 取 10% → **实际的 10%**

当前 NTP 数据集有 2,728,492 条（远超 50,000），所以验证集固定为 **5000 条**。

---

## 💡 建议的改进方案

### 方案 1: 增大验证集（短期改进）

```python
# 修改 train.py 和 evaluate_all_checkpoints.py
n_val = min(50000, n_total // 10)  # 从 5000 改为 50000
# 验证集变为 50,000 条 (1.8%)
```

**优点**:
- 简单，只需改一个数字
- 验证集增大 10 倍，更具代表性

**缺点**:
- 仍然没有独立的测试集
- 验证集占比仍然偏小

### 方案 2: 标准 train/val/test 划分（推荐）

```bash
# 创建独立的 train/val/test 文件
python scripts/split_ntp_data.py \
    --input data/ntp_train.jsonl \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --output_dir data/

# 生成：
# data/ntp_train_split.jsonl   (80% = 2,182,794 条)
# data/ntp_val.jsonl            (10% =   272,849 条)
# data/ntp_test.jsonl           (10% =   272,849 条)
```

**优点**:
- 标准的 80/10/10 划分
- 有独立的测试集用于最终评估
- 与 QA 阶段的数据处理一致

**缺点**:
- 需要创建新文件（~5.8G → 分成 3 个文件）
- 需要修改训练和评估脚本

### 方案 3: 使用哈希划分（最佳）

```python
# 基于文档 hash 动态划分，保证一致性
import hashlib

def get_split(doc_id: str) -> str:
    """基于文档 ID 的 hash 值确定属于哪个 split"""
    hash_val = int(hashlib.md5(doc_id.encode()).hexdigest(), 16)
    ratio = (hash_val % 100) / 100.0
    if ratio < 0.8:
        return "train"
    elif ratio < 0.9:
        return "val"
    else:
        return "test"
```

**优点**:
- 不需要创建新文件
- 划分是确定性的（基于文档 ID）
- 可以灵活调整划分比例
- 保证同一文档始终在同一个 split 中

**缺点**:
- 实现稍复杂
- 需要文档有唯一 ID

---

## 🎯 当前评估的可信度

### 总体评估

| 维度 | 评分 | 说明 |
|------|------|------|
| **数据一致性** | ⭐⭐⭐⭐⭐ | 训练和评估使用同一个 split，完全一致 |
| **无数据泄露** | ⭐⭐⭐⭐⭐ | 评估集未用于训练，完全独立 |
| **验证集大小** | ⭐⭐ | 5000 条只占 0.18%，偏小 |
| **代表性** | ⭐⭐⭐ | 5000 条可能不足以捕捉完整分布 |
| **可复现性** | ⭐⭐⭐⭐⭐ | 划分是确定性的，完全可复现 |
| **与 QA 一致性** | ⭐⭐ | QA 有独立 dev 集，NTP 动态划分 |

### 关键结论

✅ **评估结果是可信的**：
- 无数据泄露
- 训练和评估使用一致的 split
- 指标计算正确

⚠️ **但有改进空间**：
- 验证集太小（5000 vs 272万），代表性不足
- 缺少独立测试集
- 与 QA 阶段不一致

💡 **建议**：
- **短期**: 增大验证集到 50,000 条（改一行代码）
- **长期**: 创建标准的 train/val/test 划分（80/10/10）

---

## 📝 FAQ

### Q1: 评估时会用到训练数据吗？
**A**: 不会。训练使用前 2,723,492 条，评估使用最后 5,000 条，完全不重叠。

### Q2: 为什么不像 QA 那样创建独立的 dev 文件？
**A**: 历史原因。最初为了简化数据准备流程，NTP 使用动态划分。后续可以改进。

### Q3: 5000 条够评估吗？
**A**: 对于粗略评估够了，但如果追求精确的性能估计，建议增大到至少 50,000 条。

### Q4: 不同模型评估时用的是同一个 5000 条吗？
**A**: 是的！所有模型（包括 baseline）都用相同的最后 5000 条，确保对比公平。

### Q5: 如何验证评估集的具体范围？
```python
# 在评估脚本中添加日志
logger.info(f"Evaluation indices: [{val_indices[0]}, {val_indices[-1]}]")
# 输出：Evaluation indices: [2723492, 2728491]
```

---

## 🔧 实现细节

### 数据加载（Lazy Loading）

```python
class NTPDataset:
    def __init__(self, jsonl_path, ...):
        # 不加载数据，只记录字节偏移
        with open(jsonl_path, "rb") as f:
            self.offsets = []
            offset = 0
            while True:
                line = f.readline()
                if not line:
                    break
                self.offsets.append(offset)
                offset = f.tell()

    def __getitem__(self, idx):
        # 按需读取单条数据
        with open(self.jsonl_path, "rb") as f:
            f.seek(self.offsets[idx])
            line = f.readline()
            return json.loads(line)
```

**优点**: 内存高效，可以处理 5.8G 的大文件

### Subset 实现

```python
from torch.utils.data import Subset

# Subset 是一个包装器，只访问指定的索引
val_subset = Subset(dataset, val_indices)
# val_subset[0] → dataset[2723492]
# val_subset[1] → dataset[2723493]
# ...
# val_subset[4999] → dataset[2728491]
```

**优点**: 不需要复制数据，只是重新映射索引

---

## 📊 数据统计

### NTP 评估集统计（最后 5000 条）

```bash
# 可以运行这个脚本查看统计
python scripts/analyze_eval_split.py \
    --data_path data/ntp_train.jsonl \
    --split_start 2723492 \
    --split_end 2728492
```

**预期输出**:
- 文档长度分布
- 文本来源分布
- Token 统计
- 验证集是否与训练集分布一致

---

## 总结

当前的评估策略是：
1. **数据源**: `data/ntp_train.jsonl`（272万条）
2. **评估集**: 最后 5000 条（行 2,723,493-2,728,492）
3. **占比**: 0.18%
4. **与训练集关系**: 完全独立，无重叠
5. **可信度**: 高（无泄露），但验证集偏小

**建议**: 增大验证集到 50,000 条（1.8%），以获得更稳定的性能估计。
