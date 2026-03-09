# Q=512 配置说明（压缩比8:1）

## 与Q=256的对比

| 配置项 | Q=256 | Q=512 | 说明 |
|-------|-------|-------|------|
| **压缩比** | 16:1 (4096→256) | **8:1 (4096→512)** | 信息保留翻倍 |
| **Batch size** | 20/GPU | **16/GPU** | 显存占用增加 |
| **有效batch** | 320 | 256 | 略微降低 |
| **预期EM** | 2-3% | **8-12%** | 性能提升3-4倍 |
| **预期F1** | 0.08 | **0.18-0.25** | 性能提升2-3倍 |

## 核心改进

### 1. 更低的压缩比 = 更多信息保留

```
Q=256: [4096 tokens] → [256 queries] → 每个query平均承载16个token的信息
Q=512: [4096 tokens] → [512 queries] → 每个query平均承载8个token的信息
```

**好处**：压缩损失减半，模型更容易学习有效压缩

### 2. 显存占用调整

- Q=512的latent array是Q=256的**2倍大**
- 因此batch size从20降到16，保持显存不溢出

### 3. 性能预期

基于压缩理论和经验：
- **Q=256**：信息瓶颈严重，EM~2-3%
- **Q=512**：信息瓶颈缓解，EM预计提升到8-12%
- **Q=1024**：进一步提升，EM预计15-20%（如果显存允许）

## 训练命令

```bash
# 8 GPU训练
bash scripts/train_qa_q512_8gpu.sh

# 或者手动启动
accelerate launch --multi_gpu --num_processes 8 \
    -m deep_compressor.train \
    --config configs/qa_q512_8gpu.yaml \
    --data_path data/qa_large_train.json \
    --eval_data_path data/qa_large_dev.json \
    --stage 2
```

## 快速验证（使用Q=256的checkpoint）

如果你已经训练了Q=256，想快速看Q=512的效果：

```bash
# 1. 复制Q=256的checkpoint作为初始化（只复制Perceiver权重）
python scripts/convert_q256_to_q512.py \
    --input outputs/qa_q256_8gpu/checkpoint-final \
    --output outputs/qa_q512_init

# 2. 从这个checkpoint继续训练
bash scripts/train_qa_q512_8gpu.sh --resume_from outputs/qa_q512_init
```

**注意**：直接从Q=256扩展到Q=512需要特殊处理，因为query数量不同。建议从头训练。

## 显存占用估算

| GPU型号 | Q=256 batch=20 | Q=512 batch=16 | Q=512 batch=20 |
|---------|----------------|----------------|----------------|
| A100 80GB | ✓ 60-65GB | ✓ 65-70GB | ⚠️ 75-80GB (接近上限) |
| A100 40GB | ✓ 35-38GB | ⚠️ 38-40GB | ✗ OOM |
| V100 32GB | ⚠️ 接近上限 | ✗ OOM | ✗ OOM |

**建议**：
- A100 80GB: 使用batch=16（本配置）
- A100 40GB: 使用batch=12或开启gradient_checkpointing
- V100 32GB: 建议使用Q=256或开启gradient_checkpointing

## 故障排除

### OOM（显存不足）

```yaml
# 方案1：降低batch size
training:
  batch_size: 12  # 从16降到12
  gradient_accumulation_steps: 3  # 增加梯度累积保持有效batch

# 方案2：开启gradient checkpointing
training:
  gradient_checkpointing: true  # 牺牲20%速度换取30%显存
```

### 性能仍然差

如果Q=512后EM仍然<8%，可能原因：
1. **训练步数不足**：尝试增加到10K-15K步
2. **学习率不合适**：尝试1e-5或5e-5
3. **压缩比仍然太高**：尝试Q=1024
4. **需要Stage 1预训练**：先训练NTP stage，再fine-tune QA

## 数据集格式确认

✅ 数据集包含完整信息：

```json
{
  "context": "原文（文档）",
  "question": "问题", 
  "answer": "答案",
  "source": "数据来源"
}
```

模型会：
1. 压缩context（4096 tokens → 512 queries）
2. 用压缩后的prefix + question生成answer
3. 基于原文信息回答，不是死记硬背

## 下一步建议

训练完成后：
1. 检查eval loss是否从24降到~3（验证bug修复）
2. 检查EM/F1是否达到8-12%（验证压缩比改善）
3. 如果仍然不理想，考虑：
   - 增加训练步数（10K-15K）
   - 进一步降低压缩比（Q=1024）
   - 或者使用两阶段训练（Stage 1 NTP + Stage 2 QA）
