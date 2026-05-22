#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频切分工具函数

提供音频处理、参数优化等辅助功能
"""

import os
import multiprocessing
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import soundfile as sf

from .service import SliceConfig


class AudioSliceUtils:
    """音频切分工具类"""
    
    @staticmethod
    def analyze_audio_file(file_path: str) -> Dict:
        """
        分析音频文件信息
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            Dict: 音频信息
        """
        try:
            info = sf.info(file_path)
            return {
                'duration': info.duration,
                'sample_rate': info.samplerate,
                'channels': info.channels,
                'frames': info.frames,
                'format': info.format,
                'subtype': info.subtype,
                'file_size_mb': os.path.getsize(file_path) / (1024 * 1024)
            }
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def suggest_slice_config(file_path: str, target_segment_duration: float = 5.0) -> SliceConfig:
        """
        根据音频文件特征建议切分配置
        
        Args:
            file_path: 音频文件路径
            target_segment_duration: 目标片段时长（秒）
            
        Returns:
            SliceConfig: 建议的配置
        """
        info = AudioSliceUtils.analyze_audio_file(file_path)
        
        if 'error' in info:
            return SliceConfig()  # 返回默认配置
        
        duration = info['duration']
        file_size_mb = info['file_size_mb']
        
        # 根据文件大小调整并行处理数
        if file_size_mb > 100:
            n_parts = min(multiprocessing.cpu_count(), 8)
        elif file_size_mb > 50:
            n_parts = min(multiprocessing.cpu_count(), 4)
        else:
            n_parts = 1
        
        # 根据目标时长调整参数
        min_length = int(target_segment_duration * 1000 * 0.8)  # 80%的目标时长
        min_interval = max(200, int(target_segment_duration * 100))  # 目标时长的10%
        
        # 根据文件时长调整阈值
        if duration > 3600:  # 超过1小时
            threshold = -38.0  # 更敏感
        elif duration > 1800:  # 超过30分钟
            threshold = -36.0
        else:
            threshold = -34.0  # 默认
        
        return SliceConfig(
            threshold=threshold,
            min_length=min_length,
            min_interval=min_interval,
            hop_size=10,
            max_sil_kept=min(500, int(target_segment_duration * 100)),
            max_volume=0.9,
            alpha=0.25,
            n_parts=n_parts
        )
    
    @staticmethod
    def batch_analyze_directory(directory: str, extensions: List[str] = None) -> Dict:
        """
        批量分析目录中的音频文件
        
        Args:
            directory: 音频文件目录
            extensions: 支持的文件扩展名
            
        Returns:
            Dict: 分析结果
        """
        if extensions is None:
            extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac']
        
        directory = Path(directory)
        audio_files = []
        
        for ext in extensions:
            audio_files.extend(directory.glob(f"*{ext}"))
            audio_files.extend(directory.glob(f"*{ext.upper()}"))
        
        results = {
            'total_files': len(audio_files),
            'total_duration': 0,
            'total_size_mb': 0,
            'files': [],
            'summary': {}
        }
        
        for file_path in audio_files:
            info = AudioSliceUtils.analyze_audio_file(str(file_path))
            info['file_name'] = file_path.name
            info['file_path'] = str(file_path)
            
            if 'error' not in info:
                results['total_duration'] += info['duration']
                results['total_size_mb'] += info['file_size_mb']
            
            results['files'].append(info)
        
        # 生成摘要
        if results['total_files'] > 0:
            avg_duration = results['total_duration'] / results['total_files']
            results['summary'] = {
                'average_duration': avg_duration,
                'total_duration_hours': results['total_duration'] / 3600,
                'total_size_gb': results['total_size_mb'] / 1024,
                'suggested_config': AudioSliceUtils.suggest_slice_config_for_batch(results)
            }
        
        return results
    
    @staticmethod
    def suggest_slice_config_for_batch(batch_info: Dict) -> SliceConfig:
        """
        为批量处理建议切分配置
        
        Args:
            batch_info: 批量分析结果
            
        Returns:
            SliceConfig: 建议的配置
        """
        total_files = batch_info['total_files']
        total_size_mb = batch_info['total_size_mb']
        avg_duration = batch_info['summary'].get('average_duration', 10)
        
        # 根据文件数量和大小调整并行处理
        if total_files > 100 or total_size_mb > 1000:
            n_parts = min(multiprocessing.cpu_count(), 8)
        elif total_files > 50 or total_size_mb > 500:
            n_parts = min(multiprocessing.cpu_count(), 4)
        else:
            n_parts = 2
        
        # 根据平均时长调整参数
        if avg_duration > 60:  # 长音频
            min_length = 8000
            threshold = -38.0
        elif avg_duration > 30:  # 中等音频
            min_length = 5000
            threshold = -36.0
        else:  # 短音频
            min_length = 3000
            threshold = -34.0
        
        return SliceConfig(
            threshold=threshold,
            min_length=min_length,
            min_interval=300,
            hop_size=10,
            max_sil_kept=500,
            max_volume=0.9,
            alpha=0.25,
            n_parts=n_parts
        )
    
    @staticmethod
    def validate_slice_result(output_dir: str, min_files: int = 1) -> Dict:
        """
        验证切分结果
        
        Args:
            output_dir: 输出目录
            min_files: 最少文件数
            
        Returns:
            Dict: 验证结果
        """
        if not os.path.exists(output_dir):
            return {
                'valid': False,
                'error': '输出目录不存在'
            }
        
        output_files = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
        
        if len(output_files) < min_files:
            return {
                'valid': False,
                'error': f'输出文件数量不足，期望至少{min_files}个，实际{len(output_files)}个'
            }
        
        # 检查文件大小
        total_size = 0
        invalid_files = []
        
        for file_name in output_files:
            file_path = os.path.join(output_dir, file_name)
            file_size = os.path.getsize(file_path)
            total_size += file_size
            
            if file_size < 1024:  # 小于1KB可能是无效文件
                invalid_files.append(file_name)
        
        return {
            'valid': len(invalid_files) == 0,
            'file_count': len(output_files),
            'total_size_mb': total_size / (1024 * 1024),
            'invalid_files': invalid_files,
            'files': output_files
        }
    
    @staticmethod
    def estimate_processing_time(file_info: Dict, config: SliceConfig) -> float:
        """
        估算处理时间
        
        Args:
            file_info: 文件信息
            config: 切分配置
            
        Returns:
            float: 估算时间（秒）
        """
        if 'error' in file_info:
            return 0
        
        duration = file_info['duration']
        file_size_mb = file_info['file_size_mb']
        
        # 基础处理时间（经验值）
        base_time = duration * 0.1  # 音频时长的10%
        
        # 文件大小影响
        size_factor = min(file_size_mb / 100, 2.0)  # 最多2倍
        
        # 并行处理影响
        parallel_factor = 1.0 / config.n_parts
        
        # 参数复杂度影响
        complexity_factor = 1.0
        if config.hop_size < 10:
            complexity_factor *= 1.5
        if config.threshold < -40:
            complexity_factor *= 1.2
        
        estimated_time = base_time * size_factor * parallel_factor * complexity_factor
        
        return max(estimated_time, 1.0)  # 至少1秒


def create_test_audio_with_silence(
    duration: float = 10.0,
    silence_segments: List[Tuple[float, float]] = None,
    sample_rate: int = 32000,
    output_path: str = "test_audio.wav"
) -> str:
    """
    创建包含指定静音段的测试音频
    
    Args:
        duration: 总时长（秒）
        silence_segments: 静音段列表 [(开始时间, 结束时间), ...]
        sample_rate: 采样率
        output_path: 输出路径
        
    Returns:
        str: 输出文件路径
    """
    if silence_segments is None:
        silence_segments = [(2.0, 3.0), (6.0, 7.0)]  # 默认静音段
    
    t = np.linspace(0, duration, int(duration * sample_rate))
    audio = np.zeros_like(t)
    
    # 生成基础音频（正弦波）
    frequency = 440  # A4音符
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # 应用静音段
    for start, end in silence_segments:
        start_idx = int(start * sample_rate)
        end_idx = int(end * sample_rate)
        if 0 <= start_idx < len(audio) and 0 <= end_idx <= len(audio):
            audio[start_idx:end_idx] = 0
    
    # 保存音频
    sf.write(output_path, audio, sample_rate)
    return output_path


def optimize_slice_config_for_training(
    target_segment_duration: float = 5.0,
    training_type: str = "standard"
) -> SliceConfig:
    """
    为训练优化切分配置
    
    Args:
        target_segment_duration: 目标片段时长（秒）
        training_type: 训练类型 ("standard", "short", "long")
        
    Returns:
        SliceConfig: 优化的配置
    """
    configs = {
        "standard": {
            "threshold": -34.0,
            "min_length": int(target_segment_duration * 800),  # 80%
            "min_interval": 300,
            "max_sil_kept": 500
        },
        "short": {
            "threshold": -32.0,  # 更宽松，保留更多内容
            "min_length": int(target_segment_duration * 600),  # 60%
            "min_interval": 200,
            "max_sil_kept": 300
        },
        "long": {
            "threshold": -38.0,  # 更严格，切分更多
            "min_length": int(target_segment_duration * 1000),  # 100%
            "min_interval": 500,
            "max_sil_kept": 800
        }
    }
    
    config_params = configs.get(training_type, configs["standard"])
    
    return SliceConfig(
        threshold=config_params["threshold"],
        min_length=config_params["min_length"],
        min_interval=config_params["min_interval"],
        hop_size=10,
        max_sil_kept=config_params["max_sil_kept"],
        max_volume=0.9,
        alpha=0.25,
        n_parts=multiprocessing.cpu_count() // 2
    )