#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频处理工具

提供音频文件的编解码、信息获取、验证等功能
"""

import os
import base64
from typing import Dict, Any
from ..core.exceptions import ValidationError


class AudioUtils:
    """音频处理工具类"""
    
    @staticmethod
    def encode_audio_file(file_path: str) -> str:
        """
        将音频文件编码为Base64
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            Base64编码的音频数据
            
        Raises:
            ValidationError: 文件不存在或编码失败
        """
        if not os.path.exists(file_path):
            raise ValidationError(f"音频文件不存在: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            raise ValidationError(f"音频文件编码失败: {e}")
    
    @staticmethod
    def decode_audio_base64(base64_data: str, output_path: str):
        """
        将Base64数据解码为音频文件
        
        Args:
            base64_data: Base64编码的音频数据
            output_path: 输出文件路径
            
        Raises:
            ValidationError: 解码失败
        """
        try:
            audio_data = base64.b64decode(base64_data)
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'wb') as f:
                f.write(audio_data)
        except Exception as e:
            raise ValidationError(f"Base64音频解码失败: {e}")
    
    @staticmethod
    def get_audio_info(file_path: str) -> Dict[str, Any]:
        """
        获取音频文件信息
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            音频文件信息字典
            
        Raises:
            ValidationError: 文件不存在或获取信息失败
        """
        if not os.path.exists(file_path):
            raise ValidationError(f"音频文件不存在: {file_path}")
        
        try:
            # 尝试使用soundfile获取详细信息
            try:
                import soundfile as sf
                info = sf.info(file_path)
                return {
                    "duration": info.duration,
                    "sample_rate": info.samplerate,
                    "channels": info.channels,
                    "frames": info.frames,
                    "format": info.format,
                    "subtype": info.subtype,
                    "file_size": os.path.getsize(file_path),
                    "exists": True
                }
            except ImportError:
                # 如果没有soundfile，使用基础信息
                return {
                    "file_size": os.path.getsize(file_path),
                    "exists": True,
                    "format": os.path.splitext(file_path)[1].lower()
                }
        except Exception as e:
            raise ValidationError(f"获取音频信息失败: {e}")
    
    @staticmethod
    def validate_audio_file(file_path: str, 
                           min_duration: float = 1.0,
                           max_duration: float = 30.0,
                           required_sample_rate: int = None,
                           required_channels: int = None) -> bool:
        """
        验证音频文件
        
        Args:
            file_path: 音频文件路径
            min_duration: 最小时长(秒)
            max_duration: 最大时长(秒)
            required_sample_rate: 要求的采样率
            required_channels: 要求的声道数
            
        Returns:
            验证是否通过
            
        Raises:
            ValidationError: 验证失败
        """
        if not os.path.exists(file_path):
            raise ValidationError(f"音频文件不存在: {file_path}")
        
        try:
            info = AudioUtils.get_audio_info(file_path)
            
            # 检查时长
            duration = info.get("duration", 0)
            if duration > 0:  # 只有在能获取到时长时才检查
                if duration < min_duration:
                    raise ValidationError(f"音频时长过短: {duration:.2f}s < {min_duration}s")
                
                if duration > max_duration:
                    raise ValidationError(f"音频时长过长: {duration:.2f}s > {max_duration}s")
            
            # 检查采样率
            if required_sample_rate and info.get("sample_rate"):
                if info["sample_rate"] != required_sample_rate:
                    raise ValidationError(
                        f"采样率不匹配: {info['sample_rate']} != {required_sample_rate}"
                    )
            
            # 检查声道数
            if required_channels and info.get("channels"):
                if info["channels"] != required_channels:
                    raise ValidationError(
                        f"声道数不匹配: {info['channels']} != {required_channels}"
                    )
            
            # 检查文件大小
            file_size = info.get("file_size", 0)
            if file_size == 0:
                raise ValidationError("音频文件为空")
            
            # 检查文件格式
            supported_formats = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in supported_formats:
                raise ValidationError(f"不支持的音频格式: {file_ext}")
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"音频文件验证失败: {e}")
    
    @staticmethod
    def get_supported_formats() -> list:
        """获取支持的音频格式列表"""
        return ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
    
    @staticmethod
    def is_audio_file(file_path: str) -> bool:
        """
        检查文件是否为音频文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否为音频文件
        """
        if not os.path.exists(file_path):
            return False
        
        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in AudioUtils.get_supported_formats()
    
    @staticmethod
    def convert_sample_rate(input_path: str, output_path: str, target_rate: int):
        """
        转换音频采样率
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            target_rate: 目标采样率
            
        Raises:
            ValidationError: 转换失败
        """
        try:
            import soundfile as sf
            import librosa
            
            # 读取音频
            audio, sr = librosa.load(input_path, sr=None)
            
            # 重采样
            if sr != target_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_rate)
            
            # 保存音频
            sf.write(output_path, audio, target_rate)
            
        except ImportError:
            raise ValidationError("需要安装 soundfile 和 librosa 库来进行采样率转换")
        except Exception as e:
            raise ValidationError(f"采样率转换失败: {e}")
    
    @staticmethod
    def normalize_audio(input_path: str, output_path: str, target_db: float = -20.0):
        """
        音频音量标准化
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            target_db: 目标音量(dB)
            
        Raises:
            ValidationError: 标准化失败
        """
        try:
            import soundfile as sf
            import librosa
            import numpy as np
            
            # 读取音频
            audio, sr = sf.read(input_path)
            
            # 计算当前RMS
            rms = np.sqrt(np.mean(audio**2))
            
            # 计算目标RMS
            target_rms = 10**(target_db/20)
            
            # 标准化
            if rms > 0:
                audio = audio * (target_rms / rms)
            
            # 防止削波
            audio = np.clip(audio, -1.0, 1.0)
            
            # 保存音频
            sf.write(output_path, audio, sr)
            
        except ImportError:
            raise ValidationError("需要安装 soundfile 和 librosa 库来进行音频标准化")
        except Exception as e:
            raise ValidationError(f"音频标准化失败: {e}")