# Deep Compressor 技术报告

> 日期：2026-02-19

---

## 一、项目概述

Deep Compressor 是一个面向超长金融文本（8K–64K tokens）的压缩问答系统。其核心思路是：将完整文档通过三阶段 Perceiver 压缩为固定长度的隐向量序列（latent array），作为冻结 Qwen 解码器的前缀，使解码器能基于压缩表示回答细节级别的金融问题。训练时引入 Teacher 蒸馏监督，确保压缩前缀携带的信息等效于完整原文。

**基座模型**：Qwen3-0.6B（hidden_size=1024，28 层 Transformer，751,632,384 参数）

**可训练参数**：44,930,560（约 45M），仅占基座模型的 6.0%

**FinBERT 增强**：默认关闭。系统在纯 Qwen 路径下完整可用，FinBERT 作为可选增强模块提供金融实体级别的注意力引导。

---

## 二、系统架构

### 2.1 数据流

```
金融文本 ──→ 冻结 Qwen Encoder ──→ DownProj ──→ byte_array (doc_len × 512)
                                                      │
用户问题 ──→ 冻结 Qwen Encoder ──→ 均值池化 ──→ QueryInit ──→ queries (64 × 512)
                                                      │
                          ┌───────────────────────────┘
                          ▼
                   GuidedPerceiver
                  ┌─────────────┐
                  │  Stage A    │  全局压缩: queries ← byte_array
                  │  Stage B    │  自注意力（或锚点精读）
                  │  Stage C    │  深层推理: queries ← byte_array
                  └──────┬──────┘
                         ▼
              latent_array (64 × 512)
                         │
                       UpMLP
                         ▼
              prefix_embeds (64 × 1024)
                         │
           prefix + 问题 tokens + 答案 tokens
                         │
                   冻结 Qwen Decoder ──→ 生成回答
```

### 2.2 模块清单

| 模块 | 文件 | 参数量 | 功能 |
|------|------|--------|------|
| DownProj | `modules/down_proj.py` | 1,181,952 | Qwen 维度 (1024) → Perceiver 维度 (512) |
| QueryInit | `modules/query_init.py` | 557,568 | 可学习基础查询 + 问题偏置 |
| GuidedPerceiver | `modules/perceiver.py` | 42,007,552 | 三阶段注意力压缩（核心模块） |
| UpMLP | `modules/up_mlp.py` | 1,183,488 | Perceiver 维度 (512) → Qwen 维度 (1024) |
| **合计（FinBERT OFF）** | | **44,930,560** | |

**可选 FinBERT 模块**（启用后额外增加约 1.6M 参数）：

| 模块 | 文件 | 功能 |
|------|------|------|
| AnchorAlign | `modules/anchor_align.py` | FinBERT → Perceiver 空间对齐（3 层残差 MLP） |
| NERHead | `modules/ner_head.py` | 逐 token 实体概率分数 |
| FactDecodeHead | `modules/fact_decode_head.py` | 锚点还原辅助损失 |
| TokenizerAlign | `modules/tokenizer_align.py` | 字符级 span 重叠跨 tokenizer 分数映射 |

### 2.3 GuidedPerceiver 层结构（默认配置）

| 阶段 | Cross-Attn 层 | Self-Attn 层 | 功能 |
|------|--------------|-------------|------|
| Stage A | 2 | 2 | 全局粗粒度压缩，queries 从 byte_array 提取信息 |
| Stage B | 0（FinBERT OFF） | 2 | 自注意力精炼 / 锚点精读（FinBERT ON） |
| Stage C | 2 | 4 | 深层推理，re-read byte_array + 多层自注意力 |
| **合计** | **4** | **8** | **12 个注意力块** |

每个注意力块包含：Pre-LayerNorm → Multi-Head Attention (8 头 × 64 维) → 残差连接 → Feed-Forward (512 → 2048 → 512)。

---

## 三、训练策略

### 3.1 两阶段训练

| | Stage 1: NTP 预训练 | Stage 2: QA 微调 + 蒸馏 |
|---|---|---|
| **目标** | 解决冷启动，让前缀有意义 | 对齐 Teacher 推理过程 |
| **输入** | 文档 → 压缩前缀 → 续写文档片段 | 文档 + 问题 → 压缩前缀 → 生成答案 |
| **损失** | 标准 NTP | QA CE + KL 蒸馏 + 隐藏状态 MSE |
| **数据** | 任意文本（无需标注） | QA 数据集（SQuAD / CMRC2018 / DuReader） |
| **Teacher** | 无 | 同一个冻结 Qwen 读完整原文 |

### 3.2 蒸馏损失设计

**总损失**：$\mathcal{L} = w_1 \cdot \mathcal{L}_{\text{QA-CE}} + w_2 \cdot \mathcal{L}_{\text{KL}} + w_3 \cdot \mathcal{L}_{\text{MSE}} + w_4 \cdot \mathcal{L}_{\text{anchor}}$

| 损失分量 | 权重 | 说明 |
|----------|------|------|
| QA 交叉熵 | 1.0 | 标准答案生成损失 |
| KL 散度 | 1.0 | 温度缩放 (T=2.0) 下的输出分布对齐，仅 answer token 位置 |
| 隐藏状态 MSE | 1.0 | 选定层 [7, 14, 21, 27] 的 question+answer 共享位置对齐 |
| 锚点还原 | 0.5 | 仅 FinBERT 启用时生效 |

**关键设计**：隐藏状态 MSE 权重从 0 线性增长到 1.0（ramp_steps=2000），避免训练初期梯度混乱。

### 3.3 默认超参数

```yaml
learning_rate:       1e-4
batch_size:          4 (× gradient_accumulation=4 = 有效批次 16)
max_steps:           50,000
warmup_steps:        1,000
weight_decay:        0.01
max_grad_norm:       1.0
scheduler:           cosine
压缩查询数:           64
Perceiver 工作维度:    512
```

---

## 四、代码实现

### 4.1 项目结构

```
IMLC/
├── deep_compressor/          # 源代码：1,371 行
│   ├── config.py             # 6 个 dataclass 配置 (119 行)
│   ├── model.py              # DeepCompressor 主模型 (292 行)
│   ├── modules/              # 8 个子模块
│   │   ├── perceiver.py      # GuidedPerceiver 三阶段压缩 (255 行)
│   │   ├── tokenizer_align.py# 跨 tokenizer 分数映射 (80 行)
│   │   ├── anchor_align.py   # FinBERT → Perceiver 对齐 (45 行)
│   │   ├── ner_head.py       # NER 分类头 (31 行)
│   │   ├── down_proj.py      # 下投影 MLP (25 行)
│   │   ├── up_mlp.py         # 上投影 MLP (25 行)
│   │   ├── query_init.py     # 查询初始化 (22 行)
│   │   └── fact_decode_head.py# 锚点还原头 (22 行)
│   ├── loss.py               # 蒸馏损失 (105 行)
│   ├── data.py               # 数据集与 collator (137 行)
│   └── train.py              # 训练脚本 (207 行)
├── tests/                    # 测试代码：853 行
│   ├── test_model.py         # 模型单元测试 (150 行)
│   ├── test_integration.py   # 真实模型集成测试 (92 行)
│   ├── test_perceiver.py     # Perceiver 测试 (80 行)
│   └── ... (11 个测试文件)
├── configs/
│   └── default.yaml          # 默认配置文件
└── conftest.py               # 根 conftest (26 行)
```

**代码总计**：2,224 行（源码 1,371 + 测试 853）

### 4.2 关键设计决策

**1. 冻结模型与可训练模块的分离**

Qwen3-0.6B 全部参数设为 `requires_grad=False`。编码器的输出通过 `.detach()` 切断计算图后，才进入可训练的 DownProj 模块。这确保梯度只流经 45M 可训练参数，不会意外修改 751M 的基座模型。

**2. Stage B 的优雅降级**

当 FinBERT 关闭时，Stage B 的 cross-attention 块自动退化为等量的 self-attention 块，保持总层数不变。系统始终维护一组 self-attention 回退块，即使 FinBERT 开启但 anchor_embs 为空时也能正确运行。

**3. 灵活的注意力偏置机制**

`PerceiverCrossAttention` 支持可选的 `anchor_scores` bias：一个可学习的缩放因子乘以 per-token 实体概率，加到注意力 logits 上。未启用 FinBERT 时此路径零开销。

**4. 前缀式解码**

解码时通过 `inputs_embeds` 拼接 prefix（压缩表示）和 suffix（问题+答案 token embeddings），labels 在 prefix 位置填 -100 以排除损失计算。这避免了修改 Qwen 模型的任何内部结构。

---

## 五、测试验证

### 5.1 测试结果汇总

```
47 passed in 5.84s
```

| 类别 | 测试数 | 耗时 | 说明 |
|------|-------|------|------|
| 单元测试 | 44 | 0.07s | 使用 tiny mock 模型，无需加载真实权重 |
| 集成测试 | 3 | ~5s | 加载真实 Qwen3-0.6B，端到端前向/反向传播 |
| **总计** | **47** | **5.84s** | **全部通过** |

### 5.2 单元测试覆盖

| 模块 | 测试数 | 测试要点 |
|------|-------|---------|
| config | 6 | 默认值、维度一致性校验、YAML 加载 |
| down_proj | 2 | 输出形状、梯度传播 |
| up_mlp | 2 | 输出形状、梯度传播 |
| query_init | 3 | 输出形状、零问题→纯基础查询、不同问题→不同查询 |
| perceiver | 5 | FinBERT ON/OFF 形状、梯度传播、变长 mask、anchor bias 生效 |
| anchor_align | 2 | 输出形状、梯度传播 |
| ner_head | 3 | 输出形状、值域 [0,1]、梯度传播 |
| fact_decode_head | 2 | 标量输出、梯度传播 |
| tokenizer_align | 4 | 精确重叠、部分重叠取最大值、无重叠为零、批量版本 |
| loss | 6 | KL=0 当 student==teacher、ramp 机制、mask 生效、组合损失 |
| model | 6 | encode/compress/decode 形状、冻结参数、NTP/QA 前向有限损失、反向传播 |
| data | 3 | NTP 数据集、QA 数据集、collator padding |

### 5.3 集成测试详情

使用真实 Qwen3-0.6B 在 Apple Silicon (MPS) 上进行：

| 测试 | 内容 | 结果 |
|------|------|------|
| `test_real_model_forward_ntp` | 文档压缩 → NTP 前向传播 | NTP loss 有限且 > 0 |
| `test_real_model_forward_qa` | 文档+问题压缩 → QA 前向传播 | QA CE loss 有限 |
| `test_real_model_backward` | NTP 前向 + 反向传播 | 可训练参数均收到梯度 |

### 5.4 验证的关键属性

1. **冻结正确性**：Qwen 的所有 751M 参数 `requires_grad=False`，不会被训练修改
2. **梯度连通性**：损失梯度成功回传到 DownProj、QueryInit、Perceiver、UpMLP 的所有可训练参数
3. **形状一致性**：从文档编码到最终解码的全链路张量维度正确
4. **FinBERT 可选性**：FinBERT ON/OFF 两条路径均正常工作
5. **数值稳定性**：所有损失值有限且大于零，无 NaN/Inf

---

## 六、参数效率分析

| 组件 | 参数量 | 占比 |
|------|--------|------|
| Qwen3-0.6B（冻结） | 751,632,384 | 94.4% |
| GuidedPerceiver | 42,007,552 | 5.3% |
| UpMLP | 1,183,488 | 0.15% |
| DownProj | 1,181,952 | 0.15% |
| QueryInit | 557,568 | 0.07% |
| **可训练总计** | **44,930,560** | **5.6%** |
| **系统总计** | **796,562,944** | **100%** |

仅需训练 5.6% 的参数即可实现文档压缩与问答，大幅降低了训练的计算成本和显存需求。

---

## 七、压缩比分析

| 输入文档长度 | 压缩后序列长度 | 压缩比 | 维度变化 |
|-------------|--------------|--------|---------|
| 8,192 tokens | 64 vectors | 128:1 | 1024 → 512 → 1024 |
| 16,384 tokens | 64 vectors | 256:1 | 1024 → 512 → 1024 |
| 32,768 tokens | 64 vectors | 512:1 | 1024 → 512 → 1024 |
| 64,000 tokens | 64 vectors | 1000:1 | 1024 → 512 → 1024 |

无论输入文档多长，输出始终是 64 个 1024 维向量（作为 Qwen 解码器的前缀）。这使得推理时的解码成本与文档长度无关。

---

## 八、后续工作

1. **Stage 1 NTP 预训练**：在大规模中文金融语料上训练，验证 NTP 损失收敛
2. **Stage 2 QA 微调**：引入 SQuAD/CMRC2018 QA 数据集，联合 KL 蒸馏与隐藏状态对齐
3. **长文本渐进训练**：从 8K 逐步扩展到 64K token 输入
4. **FinBERT 增强实验**：启用实体引导模块，验证金融实体注意力偏置的效果
5. **评测**：数字类问题 Exact Match + 整体 QA F1，对比有无蒸馏、有无 FinBERT 的消融实验

---

## 附录：运行指南

```bash
# 环境
conda activate qwen3

# 快速单元测试（0.07 秒）
python -m pytest tests/ -v --ignore=tests/test_integration.py

# 集成测试（需要 Qwen3-0.6B，约 6 秒）
python -m pytest tests/test_integration.py -v --runslow

# 全部测试
python -m pytest tests/ -v --runslow

# 训练 Stage 1 NTP
python -m deep_compressor.train --config configs/default.yaml --data_path <jsonl_path> --stage 1

# 训练 Stage 2 QA + 蒸馏
python -m deep_compressor.train --config configs/default.yaml --data_path <json_path> --stage 2
```
