#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 推理工具函数

提供推理相关的辅助功能
"""

import os
import re
import json
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
import librosa
import soundfile as sf
from datetime import datetime


def validate_text_input(text: str, language: str = "zh") -> Dict[str, Any]:
    """
    验证文本输入
    
    Args:
        text: 输入文本
        language: 语言类型
        
    Returns:
        Dict: 验证结果
    """
    issues = []
    
    # 基础检查
    if not text or not text.strip():
        issues.append("文本不能为空")
        return {"valid": False, "issues": issues}
    
    text = text.strip()
    
    # 长度检查
    if len(text) < 2:
        issues.append("文本长度过短")
    elif len(text) > 1000:
        issues.append("文本长度过长，建议分段处理")
    
    # 语言特定检查
    if language == "zh":
        # 中文文本检查
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        if chinese_chars == 0:
            issues.append("中文文本中未检测到中文字符")
    elif language == "en":
        # 英文文本检查
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        if english_chars == 0:
            issues.append("英文文本中未检测到英文字符")
    elif language == "ja":
        # 日文文本检查
        japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', text))
        if japanese_chars == 0:
            issues.append("日文文本中未检测到日文字符")
    
    # 特殊字符检查
    special_chars = re.findall(r'[^\w\s\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff.,!?;:()""''，。！？；：（）【】《》]', text)
    if special_chars:
        issues.append(f"包含特殊字符: {set(special_chars)}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "text_length": len(text),
        "word_count": len(text.split()),
        "language": language
    }


def detect_language(text: str) -> str:
    """
    自动检测文本语言
    
    Args:
        text: 输入文本
        
    Returns:
        str: 检测到的语言代码
    """
    # 统计各语言字符数量
    chinese_count = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_count = len(re.findall(r'[a-zA-Z]', text))
    japanese_count = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))
    korean_count = len(re.findall(r'[\uac00-\ud7af]', text))
    
    # 根据字符数量判断主要语言
    counts = {
        "zh": chinese_count,
        "en": english_count,
        "ja": japanese_count,
        "ko": korean_count
    }
    
    # 返回字符数最多的语言
    detected_lang = max(counts, key=counts.get)
    
    # 如果所有语言字符都很少，默认返回中文
    if counts[detected_lang] < 3:
        return "zh"
    
    return detected_lang


def clean_text(text: str, language: str = "zh") -> str:
    """
    清理文本内容
    
    Args:
        text: 原始文本
        language: 语言类型
        
    Returns:
        str: 清理后的文本
    """
    # 基础清理
    text = text.strip()
    
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 语言特定清理
    if language == "zh":
        # 中文标点符号标准化
        text = text.replace('，', ',').replace('。', '.').replace('！', '!').replace('？', '?')
        text = text.replace('；', ';').replace('：', ':').replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
    
    # 移除控制字符
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    return text


def split_text_by_punctuation(text: str, max_length: int = 100) -> List[str]:
    """
    按标点符号分割文本
    
    Args:
        text: 输入文本
        max_length: 最大长度
        
    Returns:
        List[str]: 分割后的文本片段
    """
    # 定义分割标点
    split_patterns = [
        r'[.!?。！？]',  # 句号、感叹号、问号
        r'[;；]',        # 分号
        r'[,，]',        # 逗号
    ]
    
    segments = [text]
    
    # 按优先级分割
    for pattern in split_patterns:
        new_segments = []
        for segment in segments:
            if len(segment) <= max_length:
                new_segments.append(segment)
                continue
            
            # 分割当前片段
            parts = re.split(f'({pattern})', segment)
            current_part = ""
            
            for i, part in enumerate(parts):
                if re.match(pattern, part):
                    # 标点符号，添加到当前部分
                    current_part += part
                    if current_part.strip():
                        new_segments.append(current_part.strip())
                    current_part = ""
                else:
                    # 文本内容
                    if len(current_part + part) <= max_length:
                        current_part += part
                    else:
                        if current_part.strip():
                            new_segments.append(current_part.strip())
                        current_part = part
            
            if current_part.strip():
                new_segments.append(current_part.strip())
        
        segments = new_segments
    
    # 过滤空片段
    return [seg for seg in segments if seg.strip()]


def calculate_audio_hash(audio_path: str) -> str:
    """
    计算音频文件的哈希值
    
    Args:
        audio_path: 音频文件路径
        
    Returns:
        str: MD5哈希值
    """
    hash_md5 = hashlib.md5()
    
    try:
        with open(audio_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"计算音频哈希失败: {e}")
        return ""


def create_temp_file(suffix: str = ".wav", prefix: str = "gpt_sovits_") -> str:
    """
    创建临时文件
    
    Args:
        suffix: 文件后缀
        prefix: 文件前缀
        
    Returns:
        str: 临时文件路径
    """
    temp_file = tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, prefix=prefix
    )
    temp_file.close()
    return temp_file.name


def cleanup_temp_files(file_paths: List[str]):
    """
    清理临时文件
    
    Args:
        file_paths: 文件路径列表
    """
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"删除临时文件失败 {file_path}: {e}")


def format_duration(seconds: float) -> str:
    """
    格式化时长显示
    
    Args:
        seconds: 秒数
        
    Returns:
        str: 格式化的时长字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m{secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h{minutes}m{secs:.0f}s"


def format_file_size(bytes_size: int) -> str:
    """
    格式化文件大小显示
    
    Args:
        bytes_size: 字节大小
        
    Returns:
        str: 格式化的大小字符串
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}TB"


def validate_model_files(gpt_path: str, sovits_path: str) -> Dict[str, Any]:
    """
    验证模型文件
    
    Args:
        gpt_path: GPT模型路径
        sovits_path: SoVITS模型路径
        
    Returns:
        Dict: 验证结果
    """
    issues = []
    
    # 检查文件存在性
    if not os.path.exists(gpt_path):
        issues.append(f"GPT模型文件不存在: {gpt_path}")
    
    if not os.path.exists(sovits_path):
        issues.append(f"SoVITS模型文件不存在: {sovits_path}")
    
    if issues:
        return {"valid": False, "issues": issues}
    
    # 检查文件格式
    gpt_ext = os.path.splitext(gpt_path)[1].lower()
    sovits_ext = os.path.splitext(sovits_path)[1].lower()
    
    if gpt_ext not in ['.ckpt', '.pth']:
        issues.append(f"GPT模型文件格式不正确: {gpt_ext}")
    
    if sovits_ext not in ['.pth', '.ckpt']:
        issues.append(f"SoVITS模型文件格式不正确: {sovits_ext}")
    
    # 检查文件大小
    try:
        gpt_size = os.path.getsize(gpt_path)
        sovits_size = os.path.getsize(sovits_path)
        
        if gpt_size < 1024 * 1024:  # 小于1MB
            issues.append("GPT模型文件过小，可能损坏")
        
        if sovits_size < 1024 * 1024:  # 小于1MB
            issues.append("SoVITS模型文件过小，可能损坏")
        
    except Exception as e:
        issues.append(f"无法获取文件大小: {e}")
    
    # 尝试加载模型检查格式
    try:
        gpt_checkpoint = torch.load(gpt_path, map_location='cpu')
        if not isinstance(gpt_checkpoint, dict):
            issues.append("GPT模型格式不正确")
    except Exception as e:
        issues.append(f"GPT模型加载失败: {e}")
    
    try:
        sovits_checkpoint = torch.load(sovits_path, map_location='cpu')
        if not isinstance(sovits_checkpoint, dict):
            issues.append("SoVITS模型格式不正确")
    except Exception as e:
        issues.append(f"SoVITS模型加载失败: {e}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "gpt_size": gpt_size if 'gpt_size' in locals() else 0,
        "sovits_size": sovits_size if 'sovits_size' in locals() else 0
    }


def create_inference_cache_key(text: str, ref_audio_path: str, 
                              config_dict: Dict[str, Any]) -> str:
    """
    创建推理缓存键
    
    Args:
        text: 文本内容
        ref_audio_path: 参考音频路径
        config_dict: 配置字典
        
    Returns:
        str: 缓存键
    """
    # 计算参考音频哈希
    audio_hash = calculate_audio_hash(ref_audio_path)
    
    # 创建配置哈希
    config_str = json.dumps(config_dict, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    
    # 创建文本哈希
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    # 组合缓存键
    cache_key = f"{text_hash}_{audio_hash}_{config_hash}"
    return cache_key


def save_inference_result(audio_data: np.ndarray, sample_rate: int, 
                         output_dir: str, filename: str = None) -> str:
    """
    保存推理结果
    
    Args:
        audio_data: 音频数据
        sample_rate: 采样率
        output_dir: 输出目录
        filename: 文件名，None为自动生成
        
    Returns:
        str: 保存的文件路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成文件名
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"inference_result_{timestamp}.wav"
    
    output_path = os.path.join(output_dir, filename)
    
    # 保存音频文件
    try:
        # 确保音频数据在合理范围内
        audio_data = np.clip(audio_data, -1.0, 1.0)
        sf.write(output_path, audio_data, sample_rate, format='WAV')
        return output_path
    except Exception as e:
        raise RuntimeError(f"保存音频文件失败: {e}")


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """
    从文件加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Dict: 配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.json'):
                return json.load(f)
            elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                import yaml
                return yaml.safe_load(f)
            else:
                raise ValueError("不支持的配置文件格式")
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return {}


def save_config_to_file(config: Dict[str, Any], config_path: str):
    """
    保存配置到文件
    
    Args:
        config: 配置字典
        config_path: 配置文件路径
    """
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.endswith('.json'):
                json.dump(config, f, indent=2, ensure_ascii=False)
            elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                import yaml
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            else:
                raise ValueError("不支持的配置文件格式")
    except Exception as e:
        print(f"保存配置文件失败: {e}")


def get_system_info() -> Dict[str, Any]:
    """
    获取系统信息
    
    Returns:
        Dict: 系统信息
    """
    import platform
    import psutil
    
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available,
        "disk_usage": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent
    }
    
    # GPU信息
    if torch.cuda.is_available():
        info["gpu_available"] = True
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory
    else:
        info["gpu_available"] = False
    
    return info


def benchmark_inference_speed(inference_func, test_cases: List[Dict], 
                             iterations: int = 3) -> Dict[str, Any]:
    """
    基准测试推理速度
    
    Args:
        inference_func: 推理函数
        test_cases: 测试用例列表
        iterations: 迭代次数
        
    Returns:
        Dict: 基准测试结果
    """
    import time
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        case_results = []
        
        for iteration in range(iterations):
            start_time = time.time()
            
            try:
                result = inference_func(**test_case)
                end_time = time.time()
                
                processing_time = end_time - start_time
                case_results.append({
                    "iteration": iteration + 1,
                    "processing_time": processing_time,
                    "success": True,
                    "error": None
                })
                
            except Exception as e:
                end_time = time.time()
                case_results.append({
                    "iteration": iteration + 1,
                    "processing_time": end_time - start_time,
                    "success": False,
                    "error": str(e)
                })
        
        # 计算统计信息
        successful_results = [r for r in case_results if r["success"]]
        if successful_results:
            times = [r["processing_time"] for r in successful_results]
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
        else:
            avg_time = min_time = max_time = 0
        
        results.append({
            "test_case": i + 1,
            "iterations": case_results,
            "success_rate": len(successful_results) / iterations,
            "avg_processing_time": avg_time,
            "min_processing_time": min_time,
            "max_processing_time": max_time
        })
    
    return {
        "test_cases": results,
        "overall_success_rate": sum(r["success_rate"] for r in results) / len(results),
        "overall_avg_time": sum(r["avg_processing_time"] for r in results) / len(results)
    }