# 语义编码模块

## 📋 概述

语义编码模块实现GPT-SoVITS训练流程中的**步骤1Ac**，负责从CNHubert特征中提取语义Token序列：

- **语义Token提取**：使用预训练SoVITS-G编码器提取语义表示
- **多版本支持**：自动检测并支持v1/v2/v3/v4/v2Pro/v2ProPlus版本
- **输出格式**：生成6-name2semantic.tsv或JSON格式文件

---

## 🎯 功能特性

### 🔧 核心功能
- **语义Token提取**：使用预训练SoVITS模型的VQ编码器
- **版本自动检测**：根据模型文件大小自动识别版本
- **批量处理**：支持多进程并行处理
- **智能配置**：根据数据集特征自动优化参数

### 🚀 高级特性
- **多版本兼容**：支持所有GPT-SoVITS版本
- **设备自适应**：CPU/GPU自动选择，半精度支持
- **错误恢复**：完善的错误处理和状态反馈
- **进度监控**：实时处理进度和状态反馈
- **质量验证**：输出文件完整性检查

---

## 📁 文件结构

```
semantic_encoding/
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
python run_server.py --port 8005

# 开发模式（自动重载）
python run_server.py --port 8005 --reload

# 多进程模式
python run_server.py --port 8005 --workers 4
```

### 2. 基础使用
```python
from semantic_encoding import SemanticEncodingService, SemanticEncodingRequest, SemanticEncodingConfig

# 初始化API
api = SemanticEncodingService()

# 配置参数
config = SemanticEncodingConfig(
    pretrained_s2G="GPT_SoVITS/pretrained_models/s2G2333k.pth",
    s2config_path="GPT_SoVITS/configs/s2.json",
    version=None,              # 自动检测版本
    device="auto",             # 自动选择设备
    n_parts=4,                 # 并行处理数
    output_format="tsv"        # 输出格式
)

# 创建请求
request = SemanticEncodingRequest(
    input_text_file="train_list.txt",       # 标注文件
    cnhubert_dir="4-cnhubert/",             # CNHubert特征目录
    experiment_name="my_experiment",         # 实验名称
    output_dir="semantic_output/",          # 输出目录
    config=config
)

# 执行语义编码
result = await api.encode_semantic(request)
print(f"处理完成: {result.processed_count} 个文件")
```

### 3. HTTP API调用
```bash
# 同步语义编码
curl -X POST "http://localhost:8005/encode" \
  -H "Content-Type: application/json" \
  -d '{
    "input_text_file": "train_list.txt",
    "cnhubert_dir": "4-cnhubert/",
    "experiment_name": "test",
    "output_dir": "output/",
    "config": {
      "version": "v2",
      "n_parts": 2
    }
  }'

# 数据集分析
curl -X POST "http://localhost:8005/analyze" \
  -d "input_text_file=train_list.txt&cnhubert_dir=4-cnhubert/"

# 配置建议
curl "http://localhost:8005/suggest-config?input_text_file=train_list.txt&cnhubert_dir=4-cnhubert/"
```

---

## ⚙️ 配置参数

### SemanticEncodingConfig 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `pretrained_s2G` | str | "GPT_SoVITS/pretrained_models/s2G2333k.pth" | 预训练SoVITS模型路径 |
| `s2config_path` | str | "GPT_SoVITS/configs/s2.json" | 模型配置文件路径 |
| `version` | str | None | GPT-SoVITS版本（None=自动检测） |
| `is_half` | bool | True | 是否使用半精度 |
| `device` | str | "auto" | 计算设备 |
| `n_parts` | int | 1 | 并行处理数 |
| `output_format` | str | "tsv" | 输出格式（tsv/json） |

### 版本检测规则

| 文件大小 | 检测版本 | 说明 |
|----------|----------|------|
| < 81MB | v1 | 早期版本 |
| 81MB - 100MB | v2 | 标准版本 |
| 101MB - 700MB | v2 | 改进版本 |
| > 700MB | v3 | 大模型版本 |

---

## 📊 输出格式

### TSV格式 (6-name2semantic.tsv)
```
audio1	1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
audio2	2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
audio3	3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
```

### JSON格式 (6-name2semantic.json)
```json
{
  "audio1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  "audio2": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
  "audio3": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
}
```

### 语义Token说明
- **Token范围**：通常为0-1024的整数
- **序列长度**：与音频长度成正比
- **采样率**：25Hz（每秒25个Token）

---

## 🔧 API接口

### 核心接口

#### `POST /encode`
同步语义编码
```json
{
  "input_text_file": "train_list.txt",
  "cnhubert_dir": "4-cnhubert/",
  "experiment_name": "experiment",
  "output_dir": "output/",
  "config": { ... }
}
```

#### `POST /encode/upload`
上传文件并编码
- 支持标注文件上传
- 支持CNHubert特征压缩包上传
- 自动解压和处理

#### `POST /encode/batch`
批量编码，处理整个目录

### 分析接口

#### `POST /analyze`
数据集分析
```json
{
  "total_lines": 100,
  "valid_lines": 95,
  "missing_cnhubert": 5,
  "speakers": ["speaker1", "speaker2"],
  "languages": ["zh", "en"],
  "cnhubert_stats": {
    "total_size_mb": 150.5,
    "avg_size_kb": 1505.2
  }
}
```

#### `POST /suggest-config`
配置建议
```json
{
  "device": "cuda",
  "is_half": true,
  "n_parts": 4,
  "output_format": "tsv"
}
```

#### `POST /estimate-time`
处理时间估算
```json
{
  "estimated_total_time": 120.5,
  "processing_time": 100.0,
  "io_time": 20.5,
  "parallel_speedup": 3.2
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
- ✅ 版本检测测试

---

## 🚨 注意事项

### 模型依赖
1. **预训练模型**：需要下载对应版本的SoVITS模型
2. **配置文件**：确保s2.json配置文件存在
3. **CNHubert特征**：必须先完成音频特征提取

### 性能优化
1. **并行处理**：根据CPU核心数调整`n_parts`
2. **内存管理**：大数据集建议使用半精度
3. **设备选择**：GPU加速显著提升处理速度
4. **批量大小**：避免单次处理过多文件

### 常见问题
1. **模型加载失败**：检查模型路径和权限
2. **版本检测错误**：手动指定版本参数
3. **内存不足**：减少并行数或使用半精度
4. **CNHubert文件缺失**：确保音频特征提取完成

---

## 📈 性能基准

### 处理速度（参考）
- **CPU单进程**：~2.0秒/文件
- **CPU多进程**：~0.5秒/文件（4进程）
- **GPU加速**：~0.3秒/文件
- **GPU半精度**：~0.2秒/文件

### 内存使用
- **模型加载**：~500MB-2GB（取决于版本）
- **特征缓存**：~10MB/分钟音频
- **并行处理**：每进程额外~200MB

---

## 🔗 相关链接

- [GPT-SoVITS官方仓库](https://github.com/RVC-Boss/GPT-SoVITS)
- [语义编码原理](https://arxiv.org/abs/2301.02111)
- [FastAPI文档](https://fastapi.tiangolo.com/)

---

## 🐛 故障排除

### Q1: 模型加载失败
**原因**：模型文件路径错误或文件损坏

**解决方案**：
```python
# 1. 检查文件存在
import os
assert os.path.exists("GPT_SoVITS/pretrained_models/s2G2333k.pth")

# 2. 检查文件完整性
import torch
checkpoint = torch.load("GPT_SoVITS/pretrained_models/s2G2333k.pth")
print(checkpoint.keys())

# 3. 手动指定版本
config.version = "v2"  # 不使用自动检测
```

### Q2: CNHubert特征文件不匹配
**原因**：标注文件与CNHubert特征文件不对应

**解决方案**：
```python
# 验证文件对应关系
validation = SemanticEncodingUtils.validate_input_files(
    "train_list.txt", "4-cnhubert/", check_model_files=False
)
print(validation["statistics"])
```

### Q3: 处理速度慢
**优化方案**：
```python
# 1. 增加并行数
config.n_parts = 8

# 2. 使用GPU
config.device = "cuda"
config.is_half = True

# 3. 检查I/O瓶颈
# 确保CNHubert文件在SSD上
```

### Q4: 内存不足
**解决方案**：
```python
# 1. 启用半精度
config.is_half = True

# 2. 减少并行数
config.n_parts = 2

# 3. 使用CPU
config.device = "cpu"
```

---

## 📝 最佳实践

1. **数据验证优先**：处理前先进行数据分析和验证
2. **渐进式处理**：先用小数据集测试，再处理完整数据
3. **定期检查点**：大数据集分批处理，避免全部重来
4. **监控资源使用**：关注内存和GPU使用情况
5. **版本记录**：记录使用的模型版本和配置

---

*最后更新：2024年*