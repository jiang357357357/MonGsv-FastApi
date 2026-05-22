# GPT-SoVITS 推理模块

## 📋 概述

GPT-SoVITS 推理模块提供完整的语音合成推理服务，支持文本到语音的转换，具备模型管理、音频处理和高级配置功能。

---

## 🏗️ 模块结构

```
Inference/
├── __init__.py              # 模块初始化
├── service.py                   # 核心推理API
├── model_manager.py         # 模型管理器
├── audio_processor.py       # 音频处理器
├── server.py               # FastAPI服务器
├── utils.py                # 工具函数
├── test.py                 # 测试脚本
├── example.py              # 使用示例
├── run_server.py           # 服务启动脚本
└── README.md               # 本文档
```

---

## 🚀 快速开始

### 1. 启动推理服务

```bash
# 基础启动
python run_server.py

# 指定端口和主机
python run_server.py --host 0.0.0.0 --port 8009

# 开发模式（自动重载）
python run_server.py --reload --log-level debug

# 指定GPT-SoVITS根目录
python run_server.py --gpt-sovits-root /path/to/GPT-SoVITS-main
```

### 2. 访问API文档

启动服务后，访问 `http://localhost:8009/docs` 查看完整的API文档。

### 3. 基础使用示例

```python
import asyncio
from service import InferenceService, InferenceRequest, InferenceConfig

async def basic_inference():
    # 初始化API
    api = InferenceService()
    
    # 加载模型
    api.load_models("path/to/gpt.ckpt", "path/to/sovits.pth")
    
    # 配置推理参数
    config = InferenceConfig(
        top_k=20,
        top_p=0.6,
        temperature=0.6,
        how_to_cut="不切"
    )
    
    # 创建推理请求
    request = InferenceRequest(
        text="你好，欢迎使用GPT-SoVITS！",
        text_language="zh",
        ref_audio_path="reference.wav",
        prompt_text="参考音频内容",
        prompt_language="zh",
        config=config
    )
    
    # 执行推理
    response = await api.inference(request)
    
    if response.success:
        print(f"合成成功！音频文件: {response.audio_path}")
    else:
        print(f"合成失败: {response.message}")

# 运行示例
asyncio.run(basic_inference())
```

---

## 🔧 核心功能

### 1. 语音合成推理

- **文本处理**: 支持中文、英文、日文等多语言
- **音频合成**: 基于参考音频生成目标语音
- **格式支持**: WAV、MP3、OGG等多种音频格式
- **配置灵活**: 丰富的推理参数配置

### 2. 模型管理

- **模型注册**: 注册和管理多个模型
- **模型切换**: 动态切换不同的语音模型
- **模型验证**: 检查模型文件完整性
- **版本支持**: 支持v1-v4等多个版本

### 3. 音频处理

- **格式转换**: 多种音频格式互转
- **质量优化**: 音频归一化、去噪等
- **特征提取**: 音频特征分析
- **Base64编码**: 支持网络传输

### 4. 高级配置

- **文本切分**: 多种文本分割策略
- **采样控制**: Top-K、Top-P、温度等参数
- **语速调节**: 动态调整合成语速
- **批量处理**: 支持批量文本合成

---

## 📡 API 接口

### 核心推理接口

#### POST `/inference`
语音合成推理

**请求参数:**
```json
{
  "text": "要合成的文本",
  "text_language": "zh",
  "ref_audio_path": "参考音频路径",
  "prompt_text": "参考文本",
  "prompt_language": "zh",
  "config": {
    "top_k": 20,
    "top_p": 0.6,
    "temperature": 0.6,
    "how_to_cut": "不切",
    "speed": 1.0
  },
  "output_format": "wav",
  "return_base64": false
}
```

**响应:**
```json
{
  "success": true,
  "message": "推理完成",
  "audio_path": "/path/to/output.wav",
  "sample_rate": 22050,
  "duration": 3.5,
  "processing_time": 2.1
}
```

#### POST `/inference/file`
文件上传推理

支持通过表单上传参考音频文件进行推理。

### 模型管理接口

#### POST `/models/load`
加载模型
```json
{
  "gpt_path": "/path/to/gpt.ckpt",
  "sovits_path": "/path/to/sovits.pth"
}
```

#### GET `/models/list`
获取模型列表

#### POST `/models/register`
注册新模型
```json
{
  "name": "模型名称",
  "gpt_path": "/path/to/gpt.ckpt",
  "sovits_path": "/path/to/sovits.pth",
  "description": "模型描述"
}
```

#### POST `/models/switch/{model_name}`
切换当前模型

### 工具接口

#### GET `/utils/languages`
获取支持的语言列表

#### GET `/utils/formats`
获取支持的音频格式

#### POST `/utils/clear_cache`
清理推理缓存

---

## ⚙️ 配置参数

### 推理配置 (InferenceConfig)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `top_k` | int | 20 | Top-K采样参数 |
| `top_p` | float | 0.6 | Top-P采样参数 |
| `temperature` | float | 0.6 | 温度参数，控制随机性 |
| `how_to_cut` | str | "不切" | 文本切分方式 |
| `speed` | float | 1.0 | 语速调节 |
| `pause_second` | float | 0.3 | 句间停顿时长 |
| `ref_free` | bool | false | 无参考模式 |
| `batch_size` | int | 1 | 批处理大小 |
| `seed` | int | -1 | 随机种子 |

### 文本切分方式

- `"不切"`: 不分割文本
- `"凑四句一切"`: 每4句分割一次
- `"凑50字一切"`: 每50字分割一次
- `"按中文句号。切"`: 按中文句号分割
- `"按英文句号.切"`: 按英文句号分割
- `"按标点符号切"`: 按所有标点符号分割

---

## 🧪 测试和示例

### 运行测试

```bash
# 运行完整测试套件
python test.py

# 运行使用示例
python example.py
```

### 测试内容

- ✅ 文本验证和处理
- ✅ 语言检测
- ✅ 音频处理功能
- ✅ 模型管理
- ✅ 推理API调用
- ✅ 系统兼容性

---

## 🔍 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件路径是否正确
   - 确认模型文件格式（.ckpt, .pth）
   - 验证模型文件完整性

2. **推理失败**
   - 检查参考音频文件是否存在
   - 确认音频格式支持
   - 验证文本内容和语言设置

3. **服务启动失败**
   - 检查端口是否被占用
   - 确认依赖包安装完整
   - 查看错误日志信息

### 调试模式

```bash
# 启用详细日志
python run_server.py --log-level debug --access-log

# 开发模式
python run_server.py --reload
```

---

## 📈 性能优化

### 推理速度优化

1. **GPU加速**: 使用CUDA设备
2. **批量处理**: 合并多个请求
3. **缓存机制**: 缓存常用结果
4. **模型量化**: 使用半精度模型

### 内存优化

1. **及时清理**: 定期清理缓存
2. **流式处理**: 大文本分段处理
3. **资源监控**: 监控内存使用

---

## 🔗 相关链接

- [GPT-SoVITS 官方项目](https://github.com/RVC-Boss/GPT-SoVITS)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [PyTorch 文档](https://pytorch.org/docs/)

---

## 📝 更新日志

### v1.0.0 (2024-12)
- ✅ 完整的推理API实现
- ✅ 模型管理功能
- ✅ 音频处理工具
- ✅ FastAPI服务器
- ✅ 完整的测试和示例

---

*最后更新：2024年12月 - GPT-SoVITS 推理模块 v1.0.0*