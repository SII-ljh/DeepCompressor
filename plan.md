# Deep Compressor 技术计划书

> 日期：2026-02-19

---

## 一、项目目标

针对超长金融文本（8K-64K tokens），通过用户问题引导，将原文压缩为固定长度的语义向量序列（latent array）。这组向量作为前缀送入冻结的 Qwen 大模型，使其能回答细节级别的金融问题。训练时引入 Teacher 蒸馏监督，确保压缩前缀提供的信息等效于完整原文。

系统默认仅依赖 Qwen，不依赖 FinBERT。FinBERT 作为可选的增强模块，启用后可提供金融实体级别的注意力引导，进一步提升压缩质量。

---

## 二、整体数据流

输入是一篇金融文本和一个用户问题。

**文档编码**：文本经过冻结的 Qwen Encoder 转为隐藏状态，再通过 DownProj 投影到 Perceiver 的工作维度，得到 byte_array。

**实体引导（可选）**：如果启用 FinBERT 增强，文本同时经过冻结的 FinBERT 加 NER Head，产出两样东西。一是 top-K 关键实体的隐藏向量，经过 AnchorAlign 对齐到 Perceiver 空间，得到 anchor_embs。二是每个 token 的可微重要性分数 anchor_scores，表示该 token 是金融实体的概率。未启用 FinBERT 时，不产生 anchor_embs 和 anchor_scores，Perceiver 仅基于问题引导进行压缩。

**问题编码**：用户问题经过冻结的 Qwen Encoder 编码并均值池化，得到问题向量。QueryInit 模块用一组可学习的基础查询加上问题的加性偏置，生成固定数量的初始查询向量。

**压缩**：Guided Perceiver 执行三阶段压缩。Stage A 做全局粗粒度压缩，查询向量从 byte_array 中提取信息。Stage B 做锚点精读（启用 FinBERT 时），查询向量从锚点向量中吸收精确事实信息；未启用 FinBERT 时此阶段跳过或退化为额外的自注意力层。Stage C 再次回读全文做深层推理。如果有 anchor_scores，Stage A 和 Stage C 的 cross-attention 会用它作为 attention bias，引导模型更关注实体位置。输出是固定长度的 latent array。

**生成**：latent array 经过 UpMLP 映射回 Qwen 的表示空间，作为前缀拼接在问题 token 前面，送入冻结的 Qwen Decoder 生成回答。

---

## 三、蒸馏监督机制

训练时引入 Teacher 路径。Teacher 是同一个冻结的 Qwen 模型，直接将完整原文、问题和答案拼接后输入，不经过任何压缩。Teacher 全程不产生梯度，推理时完全移除。

蒸馏损失包含两部分。

第一部分是输出分布对齐。Teacher 和 Student 在生成答案时，每个 token 位置都有一个词表上的概率分布。通过 KL 散度要求两边分布接近，使用温度缩放让分布更平滑以暴露更多信息。

第二部分是隐藏状态对齐。虽然 Teacher 和 Student 的输入序列长度不同（Teacher 前面是完整原文，Student 前面是压缩前缀），但问题和答案部分的 token 完全相同。在这些共享 token 的位置上，要求 Decoder 对应层的隐藏状态接近。这个约束确保中间推理过程也是正确的，不只是最终输出碰巧对了。

总损失由四部分组成：标准答案生成损失、输出分布蒸馏损失、隐藏状态蒸馏损失、辅助锚点还原损失（仅启用 FinBERT 时）。

---

## 四、各模块设计要点

**DownProj**：两层 MLP，将 Qwen 隐藏状态从 Qwen 维度投影到 Perceiver 工作维度。

**QueryInit**：一组可学习的基础查询参数加上问题向量的加性偏置。基础查询提供多样性，问题偏置提供针对性。查询数量固定，由配置指定。参数量很小，约 1M。

**AnchorAlign（FinBERT 启用时）**：2-3 层 MLP 带残差连接，将 FinBERT 的表示映射到 Perceiver 空间。两个预训练模型的表示空间差异较大，多层非线性加残差连接提供足够的对齐容量。

**anchor_scores（FinBERT 启用时）**：NER Head 将分类 logits 通过 softmax 后取非 O 标签的概率总和，作为每个 token 的实体概率。这个分数是可微的，QA 损失的梯度可以回传到 NER Head，自动学习什么 token 对回答问题最重要。由于 FinBERT 和 Qwen 的 tokenizer 不同，需要通过字符级 span 重叠将 anchor_scores 从 FinBERT 的 token 粒度映射到 Qwen 的 token 粒度。

**Guided Perceiver**：三阶段注意力压缩架构。

Stage A 是全局压缩。查询向量通过 cross-attention 从 byte_array 中提取全文信息，然后做自注意力交换信息。如果有 anchor_scores，cross-attention 的 logits 会加上一个可学习缩放因子乘以 anchor_scores 作为 bias，让实体位置获得更高的注意力权重。

Stage B 是锚点精读。启用 FinBERT 时，锚点向量先通过 cross-attention 从 byte_array 中增强自身表示，然后查询向量从增强后的锚点中吸收精确事实，最后二者联合做自注意力深度整合。未启用 FinBERT 时，此阶段的层退化为查询向量的额外自注意力层，保持总层数不变。

Stage C 是深层推理。查询向量再次回读 byte_array（同样可带 anchor_scores bias），然后做多层自注意力完成全局推理。

输出是固定长度的 latent array。

**UpMLP**：两层 MLP，将 latent array 从 Perceiver 维度映射回 Qwen 维度，使其可以作为 Decoder 的输入前缀。

**FactDecodeHead（FinBERT 启用时）**：辅助损失头，要求经过 Perceiver 处理后的锚点向量仍能还原出原始锚点信息。防止压缩过程中丢失精确事实。

**DistillationLoss**：计算输出分布蒸馏和隐藏状态蒸馏两部分损失。需要正确处理 Teacher 和 Student 序列长度不同时的位置对齐。

---

## 五、FinBERT 可选机制

系统通过配置开关控制 FinBERT 的启用状态。默认关闭。

关闭时，整个数据流中不涉及 FinBERT、NER Head、AnchorAlign、anchor_embs、anchor_scores、AnchorSelector、FactDecodeHead。Perceiver 的 Stage B 退化为纯自注意力层，Stage A 和 Stage C 的 cross-attention 不使用 attention bias。系统仅依赖 Qwen Encoder 的表示和问题引导完成压缩。

开启时，FinBERT 相关模块全部激活，提供实体级别的注意力引导和锚点精读能力。总损失中加入辅助锚点还原损失。

这个设计确保系统在没有 FinBERT 的情况下完整可用，FinBERT 是锦上添花而非必要依赖。

---

## 六、训练策略

分两阶段训练，先让压缩模块学会生成有意义的前缀，再引入蒸馏对齐 Decoder 的推理过程。两个阶段都不启用 FinBERT，先用最简配置跑通。

**第一阶段：NTP 预训练。** 目的是解决冷启动问题。训练刚开始时 Perceiver 参数是随机的，输出的 latent array 对 Decoder 来说是噪声，Decoder 会学会忽略前缀直接瞎猜，后续很难纠正。NTP 预训练先让前缀变得有意义。

做法是：取一篇文档，压缩成 latent array 前缀，拼上文档中某个位置的几句话，让冻结的 Qwen Decoder 续写接下来的内容。损失就是标准的 next token prediction。不需要还原全文，只要求前缀提供的信息能帮助 Decoder 在文档的任意位置做续写。

这个阶段不需要任何标注数据，不需要 Teacher，不需要 QA 数据集。任何大量文本都可以用，金融文本或通用文本均可，从 HuggingFace 上获取即可。训练结束后，Perceiver 和 UpMLP 已经学会把文档信息编码成 Decoder 能理解的前缀形式。

**第二阶段：QA 微调 + 蒸馏。** 前缀已经有意义了，在此基础上用开源 QA 数据集（SQuAD、CMRC2018、DuReader 等）做微调。三个损失联合训练：标准 QA 交叉熵（对比 ground truth 答案）、Teacher 输出分布蒸馏（KL 散度，对齐 answer token 的概率分布）、隐藏状态蒸馏（MSE，只对齐 question+answer 位置的共享 token）。其中隐藏状态蒸馏的权重从 0 开始渐进增长，给模型适应时间，避免训练初期两边隐藏状态差距过大导致梯度混乱。

Teacher 是同一个冻结的 Qwen，直接读完整原文加问题加答案，全程不产生梯度。如果 Teacher 处理长文本的显存开销过大，可以预计算并缓存 Teacher 的 logits 和 hidden states。

**验证标准**：第一阶段看 NTP 损失能否收敛；第二阶段看 QA 准确率和蒸馏损失是否持续下降。先在短文本上跑通，后续再扩展到长文本场景、启用 FinBERT 增强、以及在金融领域数据上做正式评测。

---

## 七、核心风险与应对

**数字精度**：所有信息通过连续向量传递，存在精确数字丢失的风险。蒸馏损失是主要缓解手段——只要 Student 的输出分布接近直接读原文的 Teacher，数字自然就对了。评估时专门测试数字类问题的 exact match。如果精度不可接受，保底方案是恢复锚点文本拼接给 Decoder，同时保留蒸馏监督等其他改进。

**Teacher 显存开销**：64K token 的 Teacher 前向传播显存占用大。通过预计算缓存 Teacher 输出来解决，训练时不需要实时跑 Teacher。

**tokenizer 对齐（FinBERT 启用时）**：FinBERT 和 Qwen 的 tokenizer 不同，anchor_scores 需要跨 tokenizer 映射。通过字符级 span 重叠取最大值实现。

**长文本固定长度压缩**：8K 和 64K 的文本都压缩到相同长度的 latent array，长文本的信息密度更高，压缩难度更大。蒸馏监督和渐进式长文本训练是主要应对策略。

---

## 八、成功标准

在金融 QA 数据集上，模型能基于压缩前缀准确回答涉及具体数字、百分比、日期、机构名的细节问题。数字类问题的 exact match 和整体 QA 的 F1 是核心指标。启用 FinBERT 增强后的效果应优于纯 Qwen 基线，验证实体引导的价值。
