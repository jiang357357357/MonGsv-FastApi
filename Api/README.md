# GPT-SoVITS API客户端

这是GPT-SoVITS项目的统一API客户端模块，提供对所有功能模块的便捷调用接口。

## 功能特性

- 🚀 **异步/同步双模式**: 支持异步和同步两种调用方式
- 📦 **完整功能覆盖**: 涵盖数据准备、格式化、训练、推理全流程
- 🛡️ **健壮错误处理**: 完善的异常处理和重试机制
- 🔧 **灵活配置**: 支持配置文件和环境变量
- 📊 **进度跟踪**: 内置进度跟踪和状态监控
- 🧪 **全面测试**: 提供完整的测试套件

## 快速开始

### 安装依赖

```bash
pip install aiohttp pydantic soundfile numpy
```

### 基础使用

```python
from FastApi.Api import SyncGPTSoVITSClient, get_request_builder

# 创建客户端
client = SyncGPTSoVITSClient(base_url="http://localhost:8000")

# 创建请求构建器
builder = get_request_builder()

# 健康检查
health = client.health_check()
print(f"服务状态: {health}")
```

### 异步使用

```python
import asyncio
from FastApi.Api import GPTSoVITSClient

async def main():
    async with GPTSoVITSClient() as client:
        # 健康检查
        health = await client.health_check()
        print(f"服务状态: {health}")

asyncio.run(main())
```

## 完整工作流示例

### 1. 数据准备阶段

```python
from FastApi.Api import SyncGPTSoVITSClient, get_request_builder

client = SyncGPTSoVITSClient()
builder = get_request_builder()

# 1.1 音频切分
slice_request = builder.audio_slice_request(
    input_path="/path/to/raw/audio",
    output_dir="/path/to/sliced/audio",
    threshold=-34.0,
    min_length=4000
)
slice_response = client.audio_slice(slice_request)
print(f"切分完成，输出文件: {len(slice_response.output_files)}")

# 1.2 ASR识别
asr_request = builder.asr_request(
    input_path=slice_response.output_dir,
    output_file="/path/to/transcription.list",
    model_type="funasr",
    language="zh"
)
asr_response = client.asr_recognize(asr_request)
print(f"识别完成，标注文件: {asr_response.output_file}")
```

### 2. 数据格式化阶段

```python
# 2.1 文本特征提取
text_request = builder.text_processing_request(
    list_file=asr_response.output_file,
    output_dir="/path/to/text/features",
    language="zh"
)
text_response = client.text_processing(text_request)

# 2.2 音频特征提取
audio_request = builder.audio_features_request(
    list_file=asr_response.output_file,
    output_dir="/path/to/audio/features",
    version="v2Pro"
)
audio_response = client.audio_features(audio_request)

# 2.3 语义编码
semantic_request = builder.semantic_encoding_request(
    list_file=asr_response.output_file,
    output_dir="/path/to/semantic/features",
    version="v2Pro"
)
semantic_response = client.semantic_encoding(semantic_request)
```

### 3. 训练阶段

```python
# 3.1 SoVITS训练
sovits_request = builder.sovits_training_request(
    exp_name="my_voice_model",
    exp_root="/path/to/experiments",
    version="v2Pro",
    batch_size=32,
    total_epoch=8
)
sovits_response = client.start_sovits_training(sovits_request)
print(f"SoVITS训练启动，任务ID: {sovits_response.job_id}")

# 监控训练状态
import time
while True:
    status = client.get_training_status(sovits_response.job_id)
    print(f"训练状态: {status.status}, 进度: {status.progress}")
    
    if status.status in ["completed", "failed", "stopped"]:
        break
    
    time.sleep(30)

# 3.2 GPT训练
gpt_request = builder.gpt_training_request(
    exp_name="my_voice_model",
    exp_root="/path/to/experiments",
    batch_size=8,
    total_epoch=15
)
gpt_response = client.start_gpt_training(gpt_request)
```

### 4. 推理阶段

```python
# 4.1 文本转语音
inference_request = builder.inference_request(
    text="你好，这是使用我的声音模型合成的语音。",
    ref_audio_path="/path/to/reference.wav",
    prompt_text="参考音频对应的文本",
    text_language="zh",
    prompt_language="zh",
    return_base64=True
)
inference_response = client.inference(inference_request)

# 保存合成音频
if inference_response.audio_data:
    from FastApi.Api.utils import AudioUtils
    AudioUtils.decode_audio_base64(
        inference_response.audio_data,
        "/path/to/output.wav"
    )
    print("语音合成完成！")
```

### 5. 一键工作流

```python
# 使用完整工作流自动处理
workflow_request = builder.workflow_request(
    project_name="my_tts_project",
    input_audio_dir="/path/to/raw/audio",
    output_dir="/path/to/project/output",
    language="zh",
    version="v2Pro"
)
workflow_response = client.complete_workflow(workflow_request)
print(f"工作流启动，ID: {workflow_response.workflow_id}")
```

## 高级功能

### 配置管理

```python
from FastApi.Api.utils import APIConfig

# 创建配置
config = APIConfig()

# 设置服务地址
config.set("base_url", "http://your-server:8000")
config.set("api_key", "your-api-key")
config.set("timeout", 600)

# 使用配置创建客户端
client = SyncGPTSoVITSClient(
    base_url=config.get("base_url"),
    api_key=config.get("api_key"),
    timeout=config.get("timeout")
)
```

### 批量处理

```python
from FastApi.Api.models import BatchProject, BatchRequest

# 定义多个项目
projects = [
    BatchProject(
        name="project1",
        input_dir="/path/to/project1/audio",
        output_dir="/path/to/project1/output",
        language="zh"
    ),
    BatchProject(
        name="project2", 
        input_dir="/path/to/project2/audio",
        output_dir="/path/to/project2/output",
        language="en"
    )
]

# 批量处理
batch_request = BatchRequest(
    projects=projects,
    max_concurrent=2
)
batch_response = client.batch_process(batch_request)
```

### 错误处理

```python
from FastApi.Api.exceptions import *

try:
    response = client.inference(request)
except ValidationError as e:
    print(f"参数验证错误: {e}")
except AuthenticationError as e:
    print(f"认证失败: {e}")
except RateLimitError as e:
    print(f"请求频率过高: {e}")
except ServiceUnavailableError as e:
    print(f"服务不可用: {e.service_name}")
except GPTSoVITSAPIError as e:
    print(f"API错误: {e.message} (状态码: {e.status_code})")
```

### 进度跟踪

```python
from FastApi.Api.utils import ProgressTracker

# 创建进度跟踪器
tracker = ProgressTracker(total_steps=5)

# 执行步骤并更新进度
tracker.update(1, "开始音频切分...")
slice_response = client.audio_slice(slice_request)

tracker.update(2, "开始ASR识别...")
asr_response = client.asr_recognize(asr_request)

tracker.update(3, "开始特征提取...")
# ... 更多步骤

tracker.finish("所有步骤完成！")
```

## 工具函数

### 音频处理

```python
from FastApi.Api.utils import AudioUtils

# 音频文件编码/解码
base64_data = AudioUtils.encode_audio_file("input.wav")
AudioUtils.decode_audio_base64(base64_data, "output.wav")

# 获取音频信息
info = AudioUtils.get_audio_info("audio.wav")
print(f"时长: {info['duration']}秒, 采样率: {info['sample_rate']}")

# 验证音频文件
AudioUtils.validate_audio_file("audio.wav", min_duration=1.0, max_duration=30.0)
```

### 文件处理

```python
from FastApi.Api.utils import FileUtils

# 确保目录存在
FileUtils.ensure_dir("/path/to/output")

# 计算文件哈希
file_hash = FileUtils.get_file_hash("file.txt")

# 查找文件
wav_files = FileUtils.find_files("/path/to/audio", "*.wav", recursive=True)

# 获取目录大小
size = FileUtils.get_directory_size("/path/to/directory")
```

## 测试

### 运行测试

```bash
# 运行所有测试
python -m FastApi.Api.test

# 指定服务地址
python -m FastApi.Api.test http://localhost:8000

# 只运行基础测试
python -c "from FastApi.Api.test import run_basic_tests; run_basic_tests()"

# 只运行性能测试
python -c "from FastApi.Api.test import run_performance_tests; run_performance_tests()"
```

### 测试覆盖

- ✅ 健康检查和服务状态
- ✅ 数据准备（音频切分、ASR识别）
- ✅ 数据格式化（文本处理、音频特征、语义编码）
- ✅ 训练（GPT训练、SoVITS训练、状态监控）
- ✅ 推理（标准推理、Base64音频推理）
- ✅ 工作流（完整工作流、批量处理）
- ✅ 工具函数（音频处理、文件处理）
- ✅ 性能测试（并发请求、大文件处理）

## API参考

### 客户端类

#### GPTSoVITSClient (异步)
- `async health_check()` - 健康检查
- `async audio_slice(request)` - 音频切分
- `async asr_recognize(request)` - ASR识别
- `async text_processing(request)` - 文本处理
- `async audio_features(request)` - 音频特征提取
- `async semantic_encoding(request)` - 语义编码
- `async start_gpt_training(request)` - GPT训练
- `async start_sovits_training(request)` - SoVITS训练
- `async get_training_status(job_id)` - 获取训练状态
- `async inference(request)` - 推理
- `async complete_workflow(request)` - 完整工作流
- `async batch_process(request)` - 批量处理

#### SyncGPTSoVITSClient (同步)
提供与异步客户端相同的方法，但以同步方式调用。

### 请求构建器

#### RequestBuilder
- `audio_slice_request()` - 构建音频切分请求
- `asr_request()` - 构建ASR请求
- `text_processing_request()` - 构建文本处理请求
- `audio_features_request()` - 构建音频特征请求
- `semantic_encoding_request()` - 构建语义编码请求
- `gpt_training_request()` - 构建GPT训练请求
- `sovits_training_request()` - 构建SoVITS训练请求
- `inference_request()` - 构建推理请求
- `workflow_request()` - 构建工作流请求

### 工具类

#### AudioUtils
- `encode_audio_file(file_path)` - 音频文件编码
- `decode_audio_base64(base64_data, output_path)` - Base64解码
- `get_audio_info(file_path)` - 获取音频信息
- `validate_audio_file(file_path)` - 验证音频文件

#### FileUtils
- `ensure_dir(dir_path)` - 确保目录存在
- `get_file_hash(file_path)` - 获取文件哈希
- `find_files(directory, pattern)` - 查找文件
- `get_directory_size(directory)` - 获取目录大小
- `cleanup_temp_files(file_paths)` - 清理临时文件

#### ProgressTracker
- `update(step, message)` - 更新进度
- `finish(message)` - 完成进度

## 配置选项

### 环境变量

```bash
export GPT_SOVITS_BASE_URL="http://localhost:8000"
export GPT_SOVITS_API_KEY="your-api-key"
export GPT_SOVITS_TIMEOUT="300"
export GPT_SOVITS_MAX_RETRIES="3"
```

### 配置文件

默认配置文件位置: `~/.gpt_sovits_api.json`

```json
{
  "base_url": "http://localhost:8000",
  "api_key": null,
  "timeout": 300,
  "max_retries": 3,
  "default_language": "zh",
  "default_version": "v2Pro",
  "temp_dir": "/tmp"
}
```

## 常见问题

### Q: 如何处理大文件上传？
A: 客户端自动处理文件上传，支持大文件分块传输。可以通过调整`timeout`参数来适应大文件处理时间。

### Q: 如何监控长时间运行的任务？
A: 使用`get_training_status()`方法定期查询任务状态，或使用`wait_for_training_completion()`方法自动等待完成。

### Q: 如何处理网络错误？
A: 客户端内置重试机制，可以通过`max_retries`参数调整重试次数。对于特定错误类型，可以使用异常处理。

### Q: 如何优化性能？
A: 
- 使用异步客户端处理并发请求
- 合理设置批处理大小
- 使用批量处理接口处理多个项目
- 启用并行处理选项

## 更新日志

### v1.0.0
- 初始版本发布
- 支持完整的GPT-SoVITS工作流
- 提供异步/同步双模式客户端
- 完善的错误处理和测试覆盖

## 许可证

本项目遵循与GPT-SoVITS主项目相同的许可证。

## 贡献

欢迎提交Issue和Pull Request来改进这个API客户端！