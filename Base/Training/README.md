# GPT-SoVITS 训练模块

## 📋 模块说明

本模块提供 GPT-SoVITS 模型训练的 FastAPI 服务化实现，包括两阶段训练：GPT 模型训练（S1）和 SoVITS 模型训练（S2）。

---

## 🎯 训练架构

### 两阶段训练策略

```
阶段1 (S1): 文本 → 语义token
    ↓
阶段2 (S2): 语义token + 参考音频 → 目标音频
```

---

## 🏗️ 目录结构

```
Training/
├── README.md                        # 本文档
├── __init__.py                      # 模块初始化
├── gpt_training/                    # GPT模型训练 (阶段1)
│   ├── __init__.py                 # 模块初始化
│   ├── service.py                      # 核心API类
│   ├── server.py                   # FastAPI服务
│   ├── utils.py                    # 工具函数
│   ├── test.py                     # 测试脚本
│   ├── example.py                  # 使用示例
│   └── run_server.py               # 服务启动
├── sovits_training/                 # SoVITS模型训练 (阶段2)
│   ├── __init__.py                 # 模块初始化
│   ├── service.py                      # 核心API类
│   ├── server.py                   # FastAPI服务
│   ├── utils.py                    # 工具函数
│   ├── test.py                     # 测试脚本
│   ├── example.py                  # 使用示例
│   └── run_server.py               # 服务启动
└── unified_server.py                # 统一训练服务
```

---

## 🚀 API 接口

### GPT训练服务 (端口: 8007)

**启动服务**:
```bash
cd gpt_training
python server.py
# 或
python run_server.py --port 8007
```

**主要接口**:
- `POST /train/start` - 开始GPT训练
- `GET /train/status/{job_id}` - 获取训练状态
- `POST /train/stop/{job_id}` - 停止训练
- `GET /train/jobs` - 列出所有训练任务
- `GET /train/log/{job_id}` - 获取训练日志

### SoVITS训练服务 (端口: 8006)

**启动服务**:
```bash
cd sovits_training
python server.py
# 或
python run_server.py --port 8006
```

**主要接口**:
- `POST /train/start` - 开始SoVITS训练
- `GET /train/status/{job_id}` - 获取训练状态
- `POST /train/stop/{job_id}` - 停止训练
- `GET /train/jobs` - 列出所有训练任务
- `GET /train/log/{job_id}` - 获取训练日志

### 统一训练服务 (端口: 8008)

**启动服务**:
```bash
python unified_server.py
```

**主要接口**:
- `POST /gpt/train/start` - 开始GPT训练
- `POST /sovits/train/start` - 开始SoVITS训练
- `GET /train/status/{job_id}` - 获取任意训练状态
- `POST /train/pipeline` - 一键完整训练流程

---

## 📊 使用示例

### 1. GPT训练示例

```python
import requests

# 训练配置
gpt_config = {
    "exp_name": "my_voice_model",
    "exp_root": "/path/to/experiments",
    "config": {
        "batch_size": 8,
        "total_epoch": 15,
        "learning_rate": 0.01,
        "gpu_numbers": "0"
    }
}

# 开始训练
response = requests.post("http://localhost:8007/train/start", json=gpt_config)
job_id = response.json()["job_id"]

# 监控训练状态
status_response = requests.get(f"http://localhost:8007/train/status/{job_id}")
print(status_response.json())
```

### 2. SoVITS训练示例

```python
import requests

# 训练配置
sovits_config = {
    "exp_name": "my_voice_model",
    "exp_root": "/path/to/experiments", 
    "config": {
        "version": "v2Pro",
        "batch_size": 32,
        "total_epoch": 8,
        "gpu_numbers": "0"
    }
}

# 开始训练
response = requests.post("http://localhost:8006/train/start", json=sovits_config)
job_id = response.json()["job_id"]

# 监控训练状态
status_response = requests.get(f"http://localhost:8006/train/status/{job_id}")
print(status_response.json())
```

### 3. 完整训练流程

```python
import requests
import time

# 1. 先训练GPT模型
gpt_response = requests.post("http://localhost:8008/gpt/train/start", json=gpt_config)
gpt_job_id = gpt_response.json()["job_id"]

# 2. 等待GPT训练完成
while True:
    status = requests.get(f"http://localhost:8008/train/status/{gpt_job_id}").json()
    if status["status"] == "completed":
        break
    elif status["status"] == "failed":
        print("GPT训练失败")
        exit(1)
    time.sleep(30)

# 3. 开始SoVITS训练
sovits_response = requests.post("http://localhost:8008/sovits/train/start", json=sovits_config)
sovits_job_id = sovits_response.json()["job_id"]

# 4. 监控SoVITS训练
while True:
    status = requests.get(f"http://localhost:8008/train/status/{sovits_job_id}").json()
    if status["status"] in ["completed", "failed"]:
        break
    time.sleep(30)

print("训练完成!")
```

---

## 🔧 配置参数详解

### GPT训练配置 (GPTTrainingConfig)

```python
{
    "batch_size": 8,              # 批次大小
    "total_epoch": 15,            # 总训练轮数
    "save_every_epoch": 5,        # 保存间隔
    "learning_rate": 0.01,        # 学习率
    "warmup_steps": 2000,         # 预热步数
    "decay_steps": 40000,         # 衰减步数
    "if_save_latest": true,       # 只保存最新权重
    "if_save_every_weights": true, # 保存所有权重
    "gpu_numbers": "0",           # GPU设备
    "pretrained_s1": "",          # 预训练模型路径
    "if_dpo": false,              # DPO训练选项
    "precision": "16-mixed",      # 训练精度
    "gradient_clip": 1.0,         # 梯度裁剪
    "max_sec": 54,                # 最大音频长度
    "num_workers": 4              # 数据加载进程
}
```

### SoVITS训练配置 (SoVITSTrainingConfig)

```python
{
    "version": "v2Pro",           # 模型版本
    "batch_size": 32,             # 批次大小
    "total_epoch": 8,             # 总训练轮数
    "save_every_epoch": 4,        # 保存间隔
    "text_low_lr_rate": 0.4,      # 文本模块学习率权重
    "learning_rate": 0.0001,      # 基础学习率
    "lr_decay": 0.999875,         # 学习率衰减
    "if_save_latest": true,       # 只保存最新权重
    "if_save_every_weights": true, # 保存所有权重
    "gpu_numbers": "0",           # GPU设备
    "pretrained_s2G": "",         # 预训练SoVITS-G模型
    "pretrained_s2D": "",         # 预训练SoVITS-D模型
    "if_grad_ckpt": false,        # 梯度检查点
    "lora_rank": 32,              # LoRA秩(v3/v4)
    "fp16_run": true,             # 半精度训练
    "c_mel": 45.0,                # Mel损失权重
    "c_kl": 1.0,                  # KL损失权重
    "segment_size": 20480,        # 音频片段大小
    "sampling_rate": 32000        # 采样率
}
```

---

## ⚙️ 训练技巧

### 1. 分层学习率
```python
# 文本相关层使用较低学习率
optimizer = AdamW([
    {"params": base_params, "lr": 1e-4},
    {"params": text_embedding, "lr": 1e-4 * 0.4},
    {"params": encoder_text, "lr": 1e-4 * 0.4}
])
```

### 2. 混合精度训练
```python
# 使用 GradScaler
scaler = GradScaler(enabled=fp16_run)
with autocast(enabled=fp16_run):
    loss = model(inputs)
scaler.scale(loss).backward()
```

### 3. 动态批处理
```python
# 按音频长度分桶
DistributedBucketSampler(
    dataset, batch_size,
    [32, 300, 400, 500, 600, 700, 800, 900, 1000, ...]
)
```

### 4. 梯度裁剪
```python
# 防止梯度爆炸
grad_norm = clip_grad_value_(model.parameters(), 1.0)
```

### 5. 学习率调度
```python
# 指数衰减
scheduler = ExponentialLR(optimizer, gamma=0.999875)
```

---

## 📊 训练监控

### TensorBoard 可视化
```bash
# 启动 TensorBoard
tensorboard --logdir logs/

# 查看内容
# - 损失曲线
# - 频谱图对比
# - 音频样本
# - 学习率变化
```

### 关键指标
| 指标 | 正常范围 | 异常情况 |
|------|----------|----------|
| Generator Loss | 逐渐下降 | 持续上升/震荡 |
| Discriminator Loss | 0.5-1.0 | <0.3 或 >2.0 |
| Mel Loss | 快速下降 | 不下降 |
| Top-3 Acc (S1) | >0.8 | <0.5 |

---

## 🐛 常见问题

### Q1: 训练出现 NaN
**原因**：学习率过大、梯度爆炸、数据问题

**解决方案**：
```python
# 1. 降低学习率
learning_rate = 0.00005  # 从0.0001降到0.00005

# 2. 增强梯度裁剪
gradient_clip = 0.5  # 从1.0降到0.5

# 3. 检查数据
if torch.isnan(loss):
    print("NaN detected, skipping batch")
    continue
```

### Q2: 显存不足
**解决方案**：
```json
{
  "train": {
    "batch_size": 16,        // 从32减到16
    "segment_size": 10240,   // 从20480减到10240
    "fp16_run": true,        // 启用半精度
    "grad_ckpt": true        // 启用梯度检查点
  }
}
```

### Q3: 训练速度慢
**优化方案**：
```python
# 1. 增加数据加载进程
num_workers = 8

# 2. 启用持久化工作进程
persistent_workers = True

# 3. 增加预取因子
prefetch_factor = 4

# 4. 启用 cudnn benchmark
torch.backends.cudnn.benchmark = True
```

### Q4: 判别器过强/过弱
**调整方案**：
```python
# 判别器过强（D_loss < 0.3）
# - 降低判别器学习率
# - 增加生成器训练频率

# 判别器过弱（D_loss > 2.0）
# - 提高判别器学习率
# - 减少生成器训练频率
```

---

## 📈 超参数调优

### 学习率调整
```python
# 初始值
learning_rate = 1e-4

# 如果损失震荡
learning_rate = 5e-5

# 如果收敛慢
learning_rate = 2e-4
```

### 损失权重调整
```python
c_mel = 45      # Mel损失权重，影响音质
c_kl = 1.0      # KL损失权重，影响多样性

# 如果音质不好，增加 c_mel
c_mel = 60

# 如果多样性不足，增加 c_kl
c_kl = 1.5
```

### 模型容量调整
```python
# 增加模型容量（提高质量但增加计算量）
hidden_channels = 256    # 从192增加到256
n_layers = 8            # 从6增加到8

# 减少模型容量（加快训练但可能降低质量）
hidden_channels = 128
n_layers = 4
```

---

## 🔄 训练流程

### 完整训练流程
```bash
# 1. 准备预处理数据
cd ../预处理
python 1-get-text.py
python 2-get-hubert-wav32k.py
python 3-get-semantic.py

# 2. 训练 GPT 模型（阶段1）
cd ../训练/s1_train
python s1_train.py -c configs/s1longer.yaml

# 3. 训练 SoVITS 模型（阶段2）
cd ../s2_train
export exp_name="my_experiment"
export version="v2"
python s2_train.py

# 4. 监控训练
tensorboard --logdir logs/

# 5. 评估模型
cd ../推理
python inference_test.py
```

---

## 📝 最佳实践

1. **数据质量优先**：高质量数据比大量低质量数据更重要
2. **渐进式训练**：先用小数据集验证，再用完整数据集
3. **定期保存检查点**：避免训练中断导致损失
4. **监控训练指标**：及时发现和解决问题
5. **版本控制**：记录每次实验的配置和结果

---

*最后更新：2024年*
