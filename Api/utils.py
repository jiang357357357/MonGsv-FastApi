#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS API工具函数

提供API调用相关的辅助功能
"""

import os
import json
import time
import hashlib
import tempfile
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import asyncio

from ...Code.FastApi.Api.models import *
from .exceptions import *


class APIConfig:
    """API配置管理"""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file or os.path.expanduser("~/.gpt_sovits_api.json")
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载配置文件失败: {e}")
        
        # 默认配置
        return {
            "base_url": "http://localhost:8000",
            "api_key": None,
            "timeout": 300,
            "max_retries": 3,
            "default_language": "zh",
            "default_version": "v2Pro",
            "temp_dir": tempfile.gettempdir()
        }
    
    def save_config(self):
        """保存配置文件"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """设置配置值"""
        self.config[key] = value
        self.save_config()


class RequestBuilder:
    """请求构建器"""
    
    def __init__(self, config: APIConfig = None):
        self.config = config or APIConfig()
    
    def audio_slice_request(self, 
                           input_path: str,
                           output_dir: str,
                           threshold: float = -34.0,
                           min_length: int = 4000,
                           **kwargs) -> AudioSliceRequest:
        """构建音频切分请求"""
        config = AudioSliceConfig(
            threshold=threshold,
            min_length=min_length,
            **kwargs
        )
        
        return AudioSliceRequest(
            input_path=input_path,
            output_dir=output_dir,
            config=config,
            timeout=self.config.get("timeout", 300)
        )
    
    def asr_request(self,
                   input_path: str,
                   output_file: str,
                   model_type: str = "funasr",
                   language: str = None,
                   **kwargs) -> ASRRequest:
        """构建ASR识别请求"""
        language = language or self.config.get("default_language", "zh")
        
        config = ASRConfig(
            model_type=model_type,
            language=language,
            **kwargs
        )
        
        return ASRRequest(
            input_path=input_path,
            output_file=output_file,
            config=config,
            timeout=self.config.get("timeout", 300)
        )
    
    def text_processing_request(self,
                               list_file: str,
                               output_dir: str,
                               language: str = None,
                               **kwargs) -> TextProcessingRequest:
        """构建文本处理请求"""
        language = language or self.config.get("default_language", "zh")
        
        config = TextProcessingConfig(
            language=language,
            **kwargs
        )
        
        return TextProcessingRequest(
            list_file=list_file,
            output_dir=output_dir,
            config=config,
            timeout=self.config.get("timeout", 600)
        )
    
    def audio_features_request(self,
                              list_file: str,
                              output_dir: str,
                              version: str = None,
                              **kwargs) -> AudioFeaturesRequest:
        """构建音频特征请求"""
        version = version or self.config.get("default_version", "v2Pro")
        
        config = AudioFeaturesConfig(
            version=version,
            **kwargs
        )
        
        return AudioFeaturesRequest(
            list_file=list_file,
            output_dir=output_dir,
            config=config,
            timeout=self.config.get("timeout", 1200)
        )
    
    def semantic_encoding_request(self,
                                 list_file: str,
                                 output_dir: str,
                                 version: str = None,
                                 **kwargs) -> SemanticEncodingRequest:
        """构建语义编码请求"""
        version = version or self.config.get("default_version", "v2Pro")
        
        config = SemanticEncodingConfig(
            version=version,
            **kwargs
        )
        
        return SemanticEncodingRequest(
            list_file=list_file,
            output_dir=output_dir,
            config=config,
            timeout=self.config.get("timeout", 900)
        )
    
    def gpt_training_request(self,
                            exp_name: str,
                            exp_root: str,
                            **kwargs) -> GPTTrainingRequest:
        """构建GPT训练请求"""
        config = GPTTrainingConfig(**kwargs)
        
        return GPTTrainingRequest(
            exp_name=exp_name,
            exp_root=exp_root,
            config=config,
            timeout=self.config.get("timeout", 7200)
        )
    
    def sovits_training_request(self,
                               exp_name: str,
                               exp_root: str,
                               version: str = None,
                               **kwargs) -> SoVITSTrainingRequest:
        """构建SoVITS训练请求"""
        version = version or self.config.get("default_version", "v2Pro")
        
        config = SoVITSTrainingConfig(
            version=version,
            **kwargs
        )
        
        return SoVITSTrainingRequest(
            exp_name=exp_name,
            exp_root=exp_root,
            config=config,
            timeout=self.config.get("timeout", 3600)
        )
    
    def inference_request(self,
                         text: str,
                         ref_audio_path: str = None,
                         ref_audio_base64: str = None,
                         text_language: str = None,
                         **kwargs) -> InferenceRequest:
        """构建推理请求"""
        text_language = text_language or self.config.get("default_language", "zh")
        
        config = InferenceConfig(**kwargs)
        
        return InferenceRequest(
            text=text,
            text_language=text_language,
            ref_audio_path=ref_audio_path,
            ref_audio_base64=ref_audio_base64,
            config=config,
            timeout=self.config.get("timeout", 300)
        )
    
    def workflow_request(self,
                        project_name: str,
                        input_audio_dir: str,
                        output_dir: str,
                        language: str = None,
                        version: str = None,
                        **kwargs) -> WorkflowRequest:
        """构建工作流请求"""
        language = language or self.config.get("default_language", "zh")
        version = version or self.config.get("default_version", "v2Pro")
        
        config = WorkflowConfig(
            language=language,
            version=version,
            **kwargs
        )
        
        return WorkflowRequest(
            project_name=project_name,
            input_audio_dir=input_audio_dir,
            output_dir=output_dir,
            config=config,
            timeout=self.config.get("timeout", 3600)
        )


class AudioUtils:
    """音频处理工具"""
    
    @staticmethod
    def encode_audio_file(file_path: str) -> str:
        """将音频文件编码为Base64"""
        try:
            with open(file_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            raise ValidationError(f"音频文件编码失败: {e}")
    
    @staticmethod
    def decode_audio_base64(base64_data: str, output_path: str):
        """将Base64数据解码为音频文件"""
        try:
            audio_data = base64.b64decode(base64_data)
            with open(output_path, 'wb') as f:
                f.write(audio_data)
        except Exception as e:
            raise ValidationError(f"Base64音频解码失败: {e}")
    
    @staticmethod
    def get_audio_info(file_path: str) -> Dict[str, Any]:
        """获取音频文件信息"""
        try:
            import soundfile as sf
            info = sf.info(file_path)
            return {
                "duration": info.duration,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "frames": info.frames,
                "format": info.format,
                "file_size": os.path.getsize(file_path)
            }
        except ImportError:
            # 如果没有soundfile，使用基础信息
            return {
                "file_size": os.path.getsize(file_path),
                "exists": True
            }
        except Exception as e:
            raise ValidationError(f"获取音频信息失败: {e}")
    
    @staticmethod
    def validate_audio_file(file_path: str, 
                           min_duration: float = 1.0,
                           max_duration: float = 30.0) -> bool:
        """验证音频文件"""
        if not os.path.exists(file_path):
            raise ValidationError(f"音频文件不存在: {file_path}")
        
        try:
            info = AudioUtils.get_audio_info(file_path)
            duration = info.get("duration", 0)
            
            if duration < min_duration:
                raise ValidationError(f"音频时长过短: {duration}s < {min_duration}s")
            
            if duration > max_duration:
                raise ValidationError(f"音频时长过长: {duration}s > {max_duration}s")
            
            return True
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"音频文件验证失败: {e}")


class FileUtils:
    """文件处理工具"""
    
    @staticmethod
    def ensure_dir(dir_path: str):
        """确保目录存在"""
        os.makedirs(dir_path, exist_ok=True)
    
    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """获取文件MD5哈希"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            raise ValidationError(f"计算文件哈希失败: {e}")
    
    @staticmethod
    def find_files(directory: str, pattern: str = "*", recursive: bool = True) -> List[str]:
        """查找文件"""
        from pathlib import Path
        
        path = Path(directory)
        if not path.exists():
            return []
        
        if recursive:
            return [str(f) for f in path.rglob(pattern) if f.is_file()]
        else:
            return [str(f) for f in path.glob(pattern) if f.is_file()]
    
    @staticmethod
    def get_directory_size(directory: str) -> int:
        """获取目录大小"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size
    
    @staticmethod
    def cleanup_temp_files(file_paths: List[str]):
        """清理临时文件"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"清理临时文件失败 {file_path}: {e}")


class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, total_steps: int = 100):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.step_times = []
    
    def update(self, step: int = None, message: str = None):
        """更新进度"""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        current_time = time.time()
        self.step_times.append(current_time)
        
        progress = (self.current_step / self.total_steps) * 100
        elapsed_time = current_time - self.start_time
        
        if self.current_step > 0:
            avg_time_per_step = elapsed_time / self.current_step
            remaining_steps = self.total_steps - self.current_step
            estimated_remaining = avg_time_per_step * remaining_steps
        else:
            estimated_remaining = 0
        
        print(f"进度: {progress:.1f}% ({self.current_step}/{self.total_steps}) "
              f"已用时: {elapsed_time:.1f}s 预计剩余: {estimated_remaining:.1f}s")
        
        if message:
            print(f"  {message}")
    
    def finish(self, message: str = "完成"):
        """完成进度"""
        total_time = time.time() - self.start_time
        print(f"{message} - 总耗时: {total_time:.1f}s")


class RetryHelper:
    """重试助手"""
    
    @staticmethod
    async def retry_async(func, max_retries: int = 3, delay: float = 1.0, 
                         backoff: float = 2.0, exceptions: tuple = (Exception,)):
        """异步重试装饰器"""
        for attempt in range(max_retries + 1):
            try:
                return await func()
            except exceptions as e:
                if attempt == max_retries:
                    raise
                
                wait_time = delay * (backoff ** attempt)
                print(f"重试 {attempt + 1}/{max_retries} 失败: {e}, {wait_time:.1f}s后重试...")
                await asyncio.sleep(wait_time)
    
    @staticmethod
    def retry_sync(func, max_retries: int = 3, delay: float = 1.0,
                  backoff: float = 2.0, exceptions: tuple = (Exception,)):
        """同步重试装饰器"""
        for attempt in range(max_retries + 1):
            try:
                return func()
            except exceptions as e:
                if attempt == max_retries:
                    raise
                
                wait_time = delay * (backoff ** attempt)
                print(f"重试 {attempt + 1}/{max_retries} 失败: {e}, {wait_time:.1f}s后重试...")
                time.sleep(wait_time)


class ResponseValidator:
    """响应验证器"""
    
    @staticmethod
    def validate_response(response: BaseResponse, expected_fields: List[str] = None):
        """验证响应"""
        if not response.success:
            raise ProcessingError(response.message)
        
        if expected_fields:
            for field in expected_fields:
                if not hasattr(response, field) or getattr(response, field) is None:
                    raise ValidationError(f"响应缺少必需字段: {field}")
    
    @staticmethod
    def validate_file_response(response: BaseResponse, file_field: str):
        """验证文件响应"""
        ResponseValidator.validate_response(response, [file_field])
        
        file_path = getattr(response, file_field)
        if not os.path.exists(file_path):
            raise ProcessingError(f"响应中的文件不存在: {file_path}")
    
    @staticmethod
    def validate_training_response(response: Union[GPTTrainingResponse, SoVITSTrainingResponse]):
        """验证训练响应"""
        ResponseValidator.validate_response(response, ["job_id"])
        
        if not response.job_id:
            raise ProcessingError("训练响应缺少任务ID")


class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_paths(paths: Dict[str, str], check_exists: bool = True):
        """验证路径配置"""
        for name, path in paths.items():
            if not path:
                raise ValidationError(f"路径 {name} 不能为空")
            
            if check_exists and not os.path.exists(path):
                raise ValidationError(f"路径 {name} 不存在: {path}")
    
    @staticmethod
    def validate_audio_slice_config(config: AudioSliceConfig):
        """验证音频切分配置"""
        if config.threshold > -10:
            raise ValidationError("静音阈值过高，可能导致切分过碎")
        
        if config.min_length < 1000:
            raise ValidationError("最短时长过短，建议至少1秒")
        
        if config.min_interval < 100:
            raise ValidationError("最短间隔过短，可能导致切分不准确")
    
    @staticmethod
    def validate_training_config(config: Union[GPTTrainingConfig, SoVITSTrainingConfig]):
        """验证训练配置"""
        if config.batch_size < 1:
            raise ValidationError("批次大小必须大于0")
        
        if config.total_epoch < 1:
            raise ValidationError("训练轮数必须大于0")
        
        if config.learning_rate <= 0:
            raise ValidationError("学习率必须大于0")
        
        if hasattr(config, 'gpu_numbers'):
            gpu_numbers = config.gpu_numbers.split('-')
            for gpu in gpu_numbers:
                if not gpu.isdigit():
                    raise ValidationError(f"无效的GPU编号: {gpu}")


# ==================== 便捷函数 ====================

def create_temp_audio_file(duration: float = 3.0, sample_rate: int = 22050) -> str:
    """创建临时测试音频文件"""
    import numpy as np
    
    # 生成简单的正弦波
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t) * 0.3
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_file.close()
    
    try:
        import soundfile as sf
        sf.write(temp_file.name, audio, sample_rate)
    except ImportError:
        # 如果没有soundfile，创建一个空文件作为占位
        with open(temp_file.name, 'wb') as f:
            f.write(b'RIFF' + b'\x00' * 40)  # 简单的WAV头
    
    return temp_file.name


def format_duration(seconds: float) -> str:
    """格式化时长显示"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}分{secs:.0f}秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}小时{minutes}分钟"


def format_file_size(bytes_size: int) -> str:
    """格式化文件大小显示"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}TB"


def get_default_config() -> APIConfig:
    """获取默认配置"""
    return APIConfig()


def get_request_builder(config: APIConfig = None) -> RequestBuilder:
    """获取请求构建器"""
    return RequestBuilder(config or get_default_config())