# DeepCompressor

针对超长金融文本（8K-64K tokens）的压缩系统，将原文压缩为固定长度的隐向量序列（latent array），作为冻结 Qwen 解码器的前缀来回答细节级别的金融问题。训练时通过 Teacher 蒸馏确保压缩前缀携带的信息等效于完整原文。

## 架构概览

```
文档 ──► 冻结 Qwen 编码 ──► DownProj ──► byte_array
                                              │
问题 ──► 冻结 Qwen 编码 ──► QueryInit ──► queries
                                              │
                              GuidedPerceiver(queries, byte_array)
                                              │
                                          latent_array
                                              │
                                            UpMLP
                                              │
                                        prefix_embeds
                                              │
                              冻结 Qwen 解码器(prefix + question) ──► 答案
```

**训练策略：**
- **第一阶段（NTP 预训练）**：压缩文档，在随机段上做 next-token prediction。不需要标注数据，解决冷启动问题。
- **第二阶段（QA 微调 + 蒸馏）**：联合训练 QA 交叉熵 + KL 蒸馏 + 隐藏状态 MSE 对齐。

## 项目结构

```
DeepCompressor/
├── deep_compressor/          # 核心代码包
│   ├── config.py             # 配置数据类（QwenConfig, PerceiverConfig, AblationConfig 等）
│   ├── hydra_conf.py         # Hydra 结构化配置（CLI 覆写用）
│   ├── model.py              # DeepCompressor 主模型
│   ├── train.py              # 训练脚本（argparse + Hydra + wandb）
│   ├── data.py               # NTP 和 QA 数据集 + 填充拼接器
│   ├── eval.py               # 评估指标（EM, F1, perplexity）
│   ├── loss.py               # 蒸馏损失 + 总损失计算
│   └── modules/              # 可训练模块
│       ├── down_proj.py      # Qwen 维度 → Perceiver 维度
│       ├── up_mlp.py         # Perceiver 维度 → Qwen 维度
│       ├── query_init.py     # 可学习查询 + 问题偏置
│       └── perceiver.py      # 三阶段引导式 Perceiver
├── configs/                  # YAML 配置文件
│   ├── default.yaml          # 完整训练配置
│   ├── tiny_subset.yaml      # 快速迭代配置（小数据集）
│   ├── macbook_debug.yaml    # MacBook 本地调试配置（CPU/MPS）
│   ├── ablation_base.yaml    # 消融实验基础配置
│   ├── hp_search.yaml        # 超参数搜索配置
│   └── benchmark.yaml        # 基线对比评估配置
├── scripts/                  # 实验脚本
│   ├── prepare_data.py       # 数据下载与子集生成
│   ├── hp_search.py          # Optuna 超参数搜索
│   ├── ablation.py           # 消融实验运行器（17 个实验）
│   ├── benchmark.py          # 基线对比（5 种方法）
│   ├── diagnostic.py         # （已废弃）旧版诊断实验
│   ├── visualize_architecture.py  # 架构图生成
│   └── diagnostics/          # 重构后的诊断实验（9 个）
│       ├── common.py         # 共享工具（设备检测、统计、模型加载、wandb）
│       ├── pre_training.py   # 实验 1-3：过拟合、梯度流、信息瓶颈
│       ├── mid_training.py   # 实验 4-5：查询多样性、逐阶段信息增益
│       └── post_training.py  # 实验 6-9：注意力模式、压缩保真度、长度缩放、蒸馏
├── tests/                    # 单元测试 + 集成测试（pytest）
├── plan.md                   # 详细技术计划书（中文）
├── CLAUDE.md                 # AI 辅助开发指引
└── requirements.txt          # 项目依赖
```

---

## 完整使用流程

### 第一步：搭建环境

#### 1.1 创建 conda 环境

```bash
conda create --name deep_compressor python=3.11
conda activate deep_compressor
```

#### 1.2 安装依赖

```bash
pip install -r requirements.txt
```

**主要依赖**：PyTorch >= 2.0, Transformers >= 4.40, Accelerate >= 0.20, wandb >= 0.16, Hydra >= 1.3, Optuna >= 3.5

### 第二步：准备数据

```bash
# 完整下载（模型 + 全部数据集，约 1-2 小时）
python scripts/prepare_data.py

# 仅下载小测试子集（快速验证管线）
python scripts/prepare_data.py --test

# 从已有数据中提取 tiny 子集（约 50 样本，用于超参搜索）
python scripts/prepare_data.py --make-tiny

# 跳过模型下载（如已有本地 Qwen3-0.6B）
python scripts/prepare_data.py --skip-model

# 生成消融实验子集（总数据的 5%）
python scripts/prepare_data.py --make-ablation

# 生成所有子集（tiny + dev + ablation）
python scripts/prepare_data.py --make-all-subsets
```

下载内容：
- **模型**：Qwen3-0.6B → `models/Qwen3-0.6B/`（~3 GB float32）
- **NTP 数据（第一阶段）**→ `data/ntp_train.jsonl`
- **QA 数据（第二阶段）**→ `data/qa_train.json`, `data/qa_dev.json`

#### 训练数据集介绍

**NTP 预训练数据（第一阶段）**：用于 next-token prediction 预训练，目标约 1.5B tokens。来源于四个公开语料库：

| 数据集 | 语言 | 规模 | 说明 |
|--------|------|------|------|
| WikiText-103 | 英文 | ~130M tokens, ~28K 篇 | 维基百科长文，结构化百科知识 |
| SQuAD v1.1 上下文 | 英文 | ~2M tokens, ~442 篇 | 阅读理解段落，按文章标题合并为长文档 |
| C4 realnewslike | 英文 | ~700M tokens | Common Crawl 新闻子集，流式加载 |
| CLUECorpusSmall | 中文 | ~500M tokens | 中文通用语料（CLUECorpusSmall），流式加载；如不可用则回退到 mc4 中文子集 |

每条 NTP 样本是一篇文档的纯文本（`{"text": "..."}`），最短 200 字符。训练时从文档中随机截取一段作为续写目标。

**QA 微调数据（第二阶段）**：用于问答微调和蒸馏训练。来源于五个开源抽取式问答数据集，覆盖中英文：

| 数据集 | 语言 | 训练集 | 验证集 | 说明 |
|--------|------|--------|--------|------|
| SQuAD v1.1 | 英文 | 87,599 | 10,570 | 斯坦福问答数据集，基于维基百科段落 |
| CMRC2018 | 中文 | 10,142 | 3,219 | 中文机器阅读理解，类 SQuAD 格式 |
| DuReader-robust | 中文 | 14,520 | 1,417 | 百度鲁棒阅读理解数据集 |
| TriviaQA RC | 英文 | ~95K | ~12K | 基于维基百科/网页证据的问答，上下文较长 |
| DRCD | 繁体中文 | ~27K | ~4K | 台湾繁体中文阅读理解数据集 |

每条 QA 样本包含四个字段：`context`（文档）、`question`（问题）、`answer`（答案）、`source`（来源数据集）。

#### 数据子集

| 文件 | 用途 | 样本数 |
|------|------|--------|
| `data/ntp_tiny.jsonl` | 管线冒烟测试 | 50 |
| `data/qa_tiny_train.json` / `qa_tiny_dev.json` | 快速迭代 | 50 / 20 |
| `data/ntp_dev.jsonl` | 开发集验证（按文档长度分层） | 2000 |
| `data/qa_dev_hp.json` | 超参搜索验证（按来源分层） | 500 |
| `data/ablation/` | 消融实验（总数据的 5%，按来源分层） | 自动计算 |

### 第三步：运行测试

```bash
# 运行全部单元测试（快速，不加载真实模型）
python -m pytest tests/ -x -q

# 运行特定测试模块
python -m pytest tests/test_model.py -v

# 包含慢速集成测试（需要已下载 Qwen3-0.6B）
python -m pytest tests/ --runslow

# 带覆盖率运行
python -m pytest tests/ --cov=deep_compressor
```

共 67 个测试，覆盖所有模块。集成测试标记为 `@pytest.mark.slow`，默认跳过（需要加载真实 Qwen 模型）。

### 第四步：快速冒烟验证

在正式训练前，用以下命令验证整个管线是否正常工作：

```bash
# 验证 1：基本训练管线
python -m deep_compressor.train \
    --config configs/tiny_subset.yaml \
    --data_path data/ntp_tiny.jsonl \
    --stage 1 --max_train_samples 10

# 验证 2：wandb 离线模式
python -m deep_compressor.train \
    --config configs/tiny_subset.yaml \
    --data_path data/ntp_tiny.jsonl \
    --stage 1 --wandb --wandb_offline --max_train_samples 10

# 验证 3：Optuna 超参搜索（3 次试验）
python scripts/hp_search.py --n_trials 3 --stage 1 \
    --data_path data/ntp_tiny.jsonl \
    --config configs/tiny_subset.yaml
```

### 第五步：训练模型

#### 5.1 第一阶段 — NTP 预训练

压缩文档 → 生成前缀 → 在随机段上做 next-token prediction。解决冷启动问题，不需要标注数据。

```bash
# 单 GPU / MPS / CPU（自动检测）
python -m deep_compressor.train \
    --config configs/default.yaml \
    --data_path data/ntp_train.jsonl \
    --stage 1

# MacBook 本地快速验证
python -m deep_compressor.train \
    --config configs/macbook_debug.yaml \
    --data_path data/ntp_tiny.jsonl \
    --stage 1

# 多 GPU 分布式训练
accelerate launch --multi_gpu --num_processes 4 \
    -m deep_compressor.train \
    --config configs/default.yaml \
    --data_path data/ntp_train.jsonl \
    --stage 1
```

#### 5.2 第二阶段 — QA 微调 + 蒸馏

在第一阶段的基础上，用 QA 数据集做微调，联合三个损失函数训练。

```bash
python -m deep_compressor.train \
    --config configs/default.yaml \
    --data_path data/qa_train.json \
    --eval_data_path data/qa_dev.json \
    --stage 2 \
    --resume_from outputs/checkpoint-final
```

#### 5.3 添加 wandb 实验追踪

在任何训练命令后追加 `--wandb` 参数：

```bash
# 在线模式
python -m deep_compressor.train \
    --config configs/default.yaml \
    --data_path data/ntp_train.jsonl \
    --stage 1 \
    --wandb --wandb_project deep-compressor

# 离线模式（无网络环境）
python -m deep_compressor.train \
    --config configs/tiny_subset.yaml \
    --data_path data/ntp_train.jsonl \
    --stage 1 \
    --wandb --wandb_offline
```

#### 5.4 使用 Hydra 覆写配置

```bash
python -m deep_compressor.train \
    --config-path ../configs --config-name default \
    data_path=data/ntp_train.jsonl \
    training.learning_rate=1e-3 \
    training.max_steps=100 \
    perceiver.dropout=0.2 \
    wandb.enabled=true wandb.run_name=experiment-1
```

### 第六步：运行诊断实验

共 9 个诊断实验，分为训练前、训练中、训练后三个阶段。

#### 6.1 训练前诊断（实验 1-3）

随机初始化模型，单个 batch，快速健全性检查。

```bash
python scripts/diagnostics/pre_training.py \
    --config configs/macbook_debug.yaml \
    --data_path data/ntp_tiny.jsonl \
    --steps 300 --experiments 1,2,3
```

| ID | 名称 | 说明 |
|----|------|------|
| 1 | 过拟合验证 | 在 1 个 batch 上训练，验证损失能下降 |
| 2 | 梯度流检测 | 各层梯度范数，检测梯度消失/爆炸 |
| 3 | 信息瓶颈 | 线性/MLP 探针 vs 完整 Perceiver |

#### 6.2 训练中诊断（实验 4-5）

```bash
# 独立运行（可选 checkpoint）
python scripts/diagnostics/mid_training.py \
    --config configs/macbook_debug.yaml \
    --data_path data/ntp_tiny.jsonl --experiments 4,5

# 作为训练回调函数运行（每 50 步执行一次）
python -m deep_compressor.train \
    --config configs/tiny_subset.yaml \
    --data_path data/ntp_tiny.jsonl --stage 1 \
    --diagnostic_every 50 --diagnostic_experiments 4,5
```

| ID | 名称 | 说明 |
|----|------|------|
| 4 | 查询多样性 | 余弦相似度、有效秩、死查询检测 |
| 5 | 逐阶段信息增益 | 各 Perceiver 截断点的损失 |

#### 6.3 训练后诊断（实验 6-9）

需要训练好的 checkpoint 和评估数据。

```bash
python scripts/diagnostics/post_training.py \
    --config configs/benchmark.yaml \
    --checkpoint outputs/checkpoint-final/trainable_weights.pt \
    --eval_data data/qa_dev.json --experiments 6,7,8,9
```

| ID | 名称 | 说明 |
|----|------|------|
| 6 | 注意力模式 | 熵、Gini、Jaccard 查询覆盖率 |
| 7 | 压缩保真度 | 余弦相似度 + 随机基线对比 |
| 8 | 长度缩放 | NTP 损失 vs 文档 token 长度 |
| 9 | 蒸馏质量 | KL 散度、token 一致率、隐藏状态 MSE |

### 第七步：超参数搜索

```bash
# 第一阶段（NTP） — 最小化损失
python scripts/hp_search.py --n_trials 50 --stage 1 \
    --data_path data/ntp_train.jsonl \
    --config configs/hp_search.yaml

# 第二阶段（QA） — 最大化 F1
python scripts/hp_search.py --n_trials 30 --stage 2 \
    --data_path data/qa_train.json \
    --eval_data_path data/qa_dev.json

# 可恢复搜索（持久化存储）
python scripts/hp_search.py --n_trials 50 --stage 1 \
    --data_path data/ntp_train.jsonl \
    --storage sqlite:///outputs/hp_search/study.db

# 带 wandb 追踪
python scripts/hp_search.py --n_trials 50 --stage 1 \
    --data_path data/ntp_tiny.jsonl \
    --wandb --wandb_project dc-hp-search
```

搜索空间：learning_rate, warmup_steps, weight_decay, perceiver_dropout, projection_dropout（第二阶段额外搜索 kl_temperature, kl_weight, hidden_mse_weight）。

### 第八步：消融实验

```bash
# 查看所有可用的消融实验（17 个）
python scripts/ablation.py --list

# 运行指定消融
python scripts/ablation.py --stage 1 \
    --config configs/ablation_base.yaml \
    --data_path data/ablation/ntp_ablation.jsonl \
    --ablation full_pipeline,no_stage_c,down_proj_identity

# 运行全部消融（带 wandb）
python scripts/ablation.py --stage 1 \
    --data_path data/ablation/ntp_ablation.jsonl \
    --wandb --wandb_project dc-ablation
```

可用消融实验列表：

| 类别 | 实验名 | 说明 |
|------|--------|------|
| 投影方式 | `down_proj_identity`, `down_proj_linear` | DownProj 用恒等/线性替代 MLP |
| 投影方式 | `up_proj_identity`, `up_proj_linear` | UpMLP 用恒等/线性替代 MLP |
| 查询条件 | `query_no_question` | 关闭问题条件化查询偏置 |
| Perceiver 阶段 | `no_stage_a`, `no_stage_b`, `no_stage_c` | 分别禁用各阶段 |
| Perceiver 深度 | `shallow_perceiver`, `deep_perceiver` | 浅层 / 深层 Perceiver |
| 蒸馏 | `no_kl_distill`, `no_mse_distill`, `no_distillation` | 关闭各蒸馏损失 |
| 查询数量 | `queries_16`, `queries_32`, `queries_128` | 不同查询数量 |
| 对照组 | `full_pipeline` | 完整默认管线（基线） |

### 第九步：基线对比评估

```bash
python scripts/benchmark.py \
    --config configs/benchmark.yaml \
    --checkpoint outputs/checkpoint-final/trainable_weights.pt \
    --eval_data data/qa_dev.json \
    --wandb --wandb_project dc-benchmark
```

对比的方法：

| 方法 | 说明 |
|------|------|
| `direct_qwen` | Qwen 直接读完整文档 + 问题（性能上界） |
| `random_prefix` | 随机高斯前缀（性能下界） |
| `mean_pool_linear` | 均值池化文档 + 线性投影 |
| `mean_pool_mlp` | 均值池化文档 + 两层 MLP |
| `deep_compressor` | 完整训练管线（需要 checkpoint） |

---

## 配置文件说明

| 配置文件 | 用途 | 文档 token 数 | 训练步数 |
|----------|------|--------------|---------|
| `default.yaml` | 完整生产训练 | 8192 | 50K |
| `tiny_subset.yaml` | 快速迭代 / 超参搜索 | 256 | 200 |
| `macbook_debug.yaml` | MacBook 本地调试（MPS） | 256 | 300 |
| `ablation_base.yaml` | 消融实验 | 512 | 500 |
| `hp_search.yaml` | 超参数搜索 | 256 | 200 |
| `benchmark.yaml` | 评估对比 | 8192 | — |

两种配置加载方式：
1. **YAML + `DeepCompressorConfig.from_yaml(path)`**：所有脚本默认使用
2. **Hydra + `RunConf`**：支持命令行参数覆写

## 许可

本项目仅用于研究目的。
