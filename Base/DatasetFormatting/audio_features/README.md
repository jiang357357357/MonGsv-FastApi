# 音频特征提取模块

## 📋 概述

音频特征提取模块实现GPT-SoVITS训练流程中的**步骤1Ab**，负责从音频文件中提取训练所需的特征：

- **CNHubert SSL特征**：768维语音表示特征
- **32kHz音频重采样**：标准化音频格式
- **说话人特征**：v2Pro版本的20480维说话人嵌入

---

## 🎯 功能特性

### 🔧 核心功能
- **CNHubert特征提取**：使用中文HuBERT模型提取SSL特征
- **音频预处理**：重采样、增强、归一化处理
- **说话人特征**：ERes2NetV2模型提取说话人嵌入
- **批量处理**：支持多进程并行处理
- **智能配置**：根据数据集特征自动优化参数

### 🚀 高级特性
- **版本兼容**：支持v1/v2/v2Pro/v2ProPlus/v3/v4版本
- **设备自适应**：CPU/GPU自动选择，半精度支持
- **错误恢复**：NaN检测和float32降级处理
- **进度监控**：实时处理进度和状态反馈
- **质量验证**：输出文件完整性检查

---

## 📁 文件结构

```
audio_features/
├── __init__.py          # 模块初始化
├── service.py               # 核心API实现
├── server.py            # FastAPI服务器
├── utils.py             # 工具函数集合
├── test.py              # 测试脚本
├── example.py           # 使用示例
├── run_server.py        # 服务启动脚本
└── README.md            # 本文档
```

---

## 🚀 快速开始

### 1. 启动服务
```bash
# 启动HTTP服务
python run_server.py --port 8003

# 开发模式（自动重载）
python run_server.py --port 8003 --reload

# 多进程模式
python run_server.py --port 8003 --workers 4
```

### 2. 基础使用
```python
from audio_features import AudioFeaturesService, AudioFeaturesRequest, AudioFeaturesConfig

# 初始化API
api = AudioFeaturesService()

# 配置参数
config = AudioFeaturesConfig(
    version="v2Pro",           # GPT-SoVITS版本
    device="auto",             # 自动选择设备
    n_parts=4,                 # 并行处理数
    save_cnhubert=True,        # 保存CNHubert特征
    save_speaker=True          # 保存说话人特征
)

# 创建请求
request = AudioFeaturesRequest(
    input_text_file="train_list.txt",    # 标注文件
    input_wav_dir="wavs/",               # 音频目录
    experiment_name="my_experiment",      # 实验名称
    output_dir="features/",              # 输出目录
    config=config
)

# 执行特征提取
result = await api.extract_features(request)
print(f"处理完成: {result.processed_count} 个文件")
```

### 3. HTTP API调用
```bash
# 同步特征提取
curl -X POST "http://localhost:8003/extract" \
  -H "Content-Type: application/json" \
  -d '{
    "input_text_file": "train_list.txt",
    "input_wav_dir": "wavs/",
    "experiment_name": "test",
    "output_dir": "output/",
    "config": {
      "version": "v2Pro",
      "n_parts": 2
    }
  }'

# 数据集分析
curl -X POST "http://localhost:8003/analyze" \
  -d "input_text_file=train_list.txt&input_wav_dir=wavs/"

# 配置建议
curl "http://localhost:8003/config/suggest?input_text_file=train_list.txt&version=v2Pro"
```

---

## ⚙️ 配置参数

### AudioFeaturesConfig 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cnhubert_base_dir` | str | "GPT_SoVITS/pretrained_models/chinese-hubert-base" | CNHubert模型目录 |
| `sv_model_path` | str | "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt" | 说话人模型路径 |
| `version` | str | "v2" | GPT-SoVITS版本 |
| `is_half` | bool | True | 是否使用半精度 |
| `device` | str | "auto" | 计算设备 |
| `maxx` | float | 0.95 | 最大归一化值 |
| `alpha` | float | 0.5 | 混合比例 |
| `max_audio_value` | float | 2.2 | 音频过滤阈值 |
| `n_parts` | int | 1 | 并行处理数 |
| `save_wav32k` | bool | True | 保存32kHz音频 |
| `save_cnhubert` | bool | True | 保存CNHubert特征 |
| `save_speaker` | bool | None | 保存说话人特征 |

### 版本差异

| 版本 | CNHubert特征 | 32kHz音频 | 说话人特征 | 特征维度 |
|------|-------------|-----------|-----------|----------|
| v1 | ✅ | ✅ | ❌ | 768 |
| v2 | ✅ | ✅ | ❌ | 768 |
| v2Pro | ✅ | ✅ | ✅ | 768 + 20480 |
| v2ProPlus | ✅ | ✅ | ✅ | 768 + 20480 |
| v3 | ✅ | ✅ | ✅ | 768 + 20480 |
| v4 | ✅ | ✅ | ✅ | 768 + 20480 |

---

## 📊 输出格式

### 目录结构
```
output_dir/
├── 4-cnhubert/          # CNHubert特征文件
│   ├── audio1.pt        # torch.Size([1, 768, time_steps])
│   └── audio2.pt
├── 5-wav32k/            # 32kHz音频文件
│   ├── audio1.wav       # 32000Hz, 16bit
│   └── audio2.wav
└── 7-sv_cn/             # 说话人特征文件（v2Pro+）
    ├── audio1.pt        # torch.Size([1, 20480])
    └── audio2.pt
```

### 特征格式
- **CNHubert特征**：`torch.Tensor([1, 768, time_steps])`，float32
- **说话人特征**：`torch.Tensor([1, 20480])`，float32
- **32kHz音频**：WAV格式，32000Hz采样率，16bit深度

---

## 🔧 API接口

### 核心接口

#### `POST /extract`
同步特征提取
```json
{
  "input_text_file": "train_list.txt",
  "input_wav_dir": "wavs/",
  "experiment_name": "experiment",
  "output_dir": "output/",
  "config": { ... }
}
```

#### `POST /extract_async`
异步特征提取，返回任务ID
```json
{
  "task_id": "uuid-string",
  "status": "accepted",
  "message": "任务已提交"
}
```

#### `GET /status/{task_id}`
查询任务状态
```json
{
  "status": "processing|completed|failed",
  "progress": 50,
  "message": "处理中...",
  "result": { ... }
}
```

### 分析接口

#### `POST /analyze`
数据集分析
```json
{
  "total_files": 100,
  "valid_files": 95,
  "total_duration": 3600.0,
  "speakers": ["speaker1", "speaker2"],
  "languages": ["ZH", "EN"]
}
```

#### `GET /config/suggest`
配置建议
```json
{
  "suggested_config": {
    "version": "v2Pro",
    "n_parts": 4,
    "is_half": true
  },
  "analysis": {
    "estimated_processing_time": 120.5
  }
}
```

---

## 🧪 测试和示例

### 运行测试
```bash
# 运行所有测试
python test.py

# 运行示例
python example.py
```

### 测试覆盖
- ✅ 直接API调用测试
- ✅ HTTP服务测试
- ✅ 工具函数测试
- ✅ 数据集分析测试
- ✅ 配置建议测试
- ✅ 错误处理测试

---

## 🚨 注意事项

### 模型依赖
1. **CNHubert模型**：需要下载chinese-hubert-base模型
2. **说话人模型**：v2Pro版本需要ERes2NetV2模型
3. **路径配置**：确保模型路径正确设置

### 性能优化
1. **并行处理**：根据CPU核心数调整`n_parts`
2. **内存管理**：大数据集建议使用半精度
3. **设备选择**：GPU加速显著提升处理速度
4. **批量大小**：避免单次处理过多文件

### 常见问题
1. **NaN错误**：自动降级到float32处理
2. **内存不足**：减少并行数或使用半精度
3. **模型加载失败**：检查模型路径和权限
4. **音频格式错误**：确保音频文件完整性

---

## 📈 性能基准

### 处理速度（参考）
- **CPU单进程**：~0.05秒/秒音频（CNHubert）
- **CPU多进程**：~0.01秒/秒音频（4进程）
- **GPU加速**：~0.005秒/秒音频
- **说话人特征**：~0.02秒/秒音频

### 内存使用
- **CNHubert模型**：~1.2GB显存
- **说话人模型**：~200MB显存
- **音频缓存**：~50MB/分钟音频

---

## 🔗 相关链接

- [GPT-SoVITS官方仓库](https://github.com/RVC-Boss/GPT-SoVITS)
- [CNHubert模型](https://huggingface.co/TencentGameMate/chinese-hubert-base)
- [FastAPI文档](https://fastapi.tiangolo.com/)

---

*最后更新：2024年*