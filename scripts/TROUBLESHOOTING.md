# 训练问题排查指南

如果运行训练脚本后没有反馈或GPU没有利用率，按照以下步骤排查。

## 🔍 快速诊断

### 1. 运行状态检查脚本

```bash
bash scripts/check_training_status.sh
```

这会检查：
- ✅ Tmux会话是否运行
- ✅ 训练进程是否存在
- ✅ GPU是否被使用
- ✅ 日志文件内容
- ✅ Checkpoint状态

### 2. 测试单个训练

先测试单个训练任务是否能正常工作：

```bash
bash scripts/test_single_training.sh
```

这会在**前台运行**一个简短的训练（10个样本），你可以直接看到输出和错误。

## 📋 常见问题

### 问题1: 脚本运行后没有任何输出

**原因**: 脚本使用tmux或nohup在后台运行，不会直接显示输出。

**解决方案**:

```bash
# 方法1: 进入tmux会话查看
tmux attach -t stage1_training
# 按 Ctrl+B 然后按数字键切换窗口
# 按 Ctrl+B d 退出（训练继续）

# 方法2: 查看日志文件
tail -f logs/q64_*.log

# 方法3: 检查状态
bash scripts/check_training_status.sh
```

### 问题2: GPU利用率为0

**可能原因**:

1. **训练进程还未启动完成**
   - 加载模型需要时间（1-2分钟）
   - 解决方案: 等待2-3分钟后再检查

2. **训练进程启动失败**
   - 检查日志: `tail -50 logs/q64_*.log`
   - 查看错误信息

3. **数据加载问题**
   - 检查数据文件: `ls -lh data/ntp_train.jsonl`
   - 运行测试: `bash scripts/test_single_training.sh`

4. **环境问题**
   - 检查CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
   - 检查accelerate: `accelerate env`

### 问题3: 进程存在但没有训练

**检查方法**:

```bash
# 1. 查看进程
ps aux | grep deep_compressor.train

# 2. 查看最新日志
ls -lt logs/*.log | head -5
tail -100 logs/$(ls -t logs/*.log | head -1)

# 3. 检查是否有错误
grep -i "error\|exception\|failed" logs/*.log
```

### 问题4: Tmux会话找不到

**检查**:

```bash
# 列出所有tmux会话
tmux ls

# 如果没有会话，说明训练可能失败了
# 查看日志找原因
tail -100 logs/$(ls -t logs/*.log | head -1)
```

## 🛠️ 分步排查流程

### Step 1: 基础检查

```bash
# 1. 检查数据
ls -lh data/ntp_train.jsonl

# 2. 检查配置
ls configs/stage1_q*.yaml

# 3. 检查GPU
nvidia-smi

# 4. 检查Python环境
python -c "import torch; import accelerate; print('OK')"
```

### Step 2: 测试单个训练

```bash
# 前台运行，可以直接看到输出
bash scripts/test_single_training.sh
```

如果这步失败，说明环境有问题，需要先解决。

### Step 3: 检查后台训练

```bash
# 1. 检查整体状态
bash scripts/check_training_status.sh

# 2. 查看实时日志
tail -f logs/q64_*.log

# 3. 监控GPU
watch -n 1 nvidia-smi
```

## 📝 常见错误及解决

### 错误: CUDA out of memory

**解决方案**:

```bash
# 1. 减小batch size
# 编辑配置文件
vim configs/stage1_q64.yaml

# 修改:
training:
  batch_size: 4  # 从8改为4
  gradient_accumulation_steps: 4  # 从2改为4

# 2. 启用混合精度
training:
  mixed_precision: "fp16"

# 3. 启用梯度检查点
training:
  gradient_checkpointing: true
```

### 错误: FileNotFoundError: data/ntp_train.jsonl

**解决方案**:

```bash
# 准备数据
python scripts/prepare_data.py
```

### 错误: accelerate命令未找到

**解决方案**:

```bash
# 安装accelerate
pip install accelerate

# 配置accelerate
accelerate config
```

### 错误: No module named 'deep_compressor'

**解决方案**:

```bash
# 确保在项目根目录
cd /path/to/deep_compressor

# 检查目录结构
ls deep_compressor/

# 如果不在conda环境中
conda activate deep_compressor
```

## 🔬 详细诊断命令

### 查看训练进程详情

```bash
# 查看所有训练进程
ps aux | grep deep_compressor.train | grep -v grep

# 查看特定GPU的进程
nvidia-smi pmon -i 0,1,2,3,4,5,6,7
```

### 查看日志统计

```bash
# 查看所有日志文件
ls -lh logs/

# 搜索错误
grep -i "error\|exception\|failed\|killed" logs/*.log

# 查看最后的训练步数
grep "step" logs/q64_*.log | tail -20
```

### 检查磁盘空间

```bash
# 检查磁盘使用
df -h .

# 检查outputs目录大小
du -sh outputs/
```

## 💡 前台运行（用于调试）

如果需要在前台运行看到实时输出：

```bash
# 单卡训练（前台）
CUDA_VISIBLE_DEVICES=0 python -m deep_compressor.train \
    --config configs/stage1_q64.yaml \
    --data_path data/ntp_train.jsonl \
    --stage 1

# 8卡DDP训练（前台）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --multi_gpu --num_processes 8 \
    -m deep_compressor.train \
    --config configs/stage1_q64.yaml \
    --data_path data/ntp_train.jsonl \
    --stage 1
```

## 📞 获取帮助

如果以上方法都无法解决问题：

1. **收集信息**:
   ```bash
   # 生成诊断报告
   bash scripts/check_training_status.sh > diagnosis.txt
   cat logs/$(ls -t logs/*.log | head -1) >> diagnosis.txt
   ```

2. **检查日志**:
   ```bash
   # 查看完整日志
   cat logs/$(ls -t logs/*.log | head -1)
   ```

3. **提供环境信息**:
   ```bash
   python -c "import torch; print('PyTorch:', torch.__version__)"
   nvidia-smi
   accelerate env
   ```

## ✅ 成功运行的标志

训练正常运行时，你应该看到：

1. **GPU利用率** > 50%
   ```
   nvidia-smi
   ```

2. **训练日志**有持续输出
   ```
   tail -f logs/q64_*.log
   ```

3. **训练进程**存在
   ```
   ps aux | grep deep_compressor.train
   ```

4. **定期保存checkpoint**
   ```
   ls outputs/stage1_q64/checkpoint-*/
   ```

5. **WandB**有曲线更新（如果启用）
   访问 https://wandb.ai
