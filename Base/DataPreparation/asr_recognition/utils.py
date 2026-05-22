#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR语音识别工具函数

提供音频分析、配置优化等辅助功能
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import soundfile as sf

from .service import ASRConfig


class ASRUtils:
    """ASR识别工具类"""
    
    # 语言代码映射
    LANGUAGE_MAPPING = {
        "中文": "zh",
        "粤语": "yue", 
        "英文": "en",
        "日文": "ja",
        "韩文": "ko",
        "自动": "auto"
    }
    
    # 模型推荐配置
    MODEL_RECOMMENDATIONS = {
        "zh": {
            "model_type": "funasr",
            "model_size": "large",
            "precision": "float32",
            "description": "中文语音识别，准确率高"
        },
        "yue": {
            "model_type": "funasr", 
            "model_size": "large",
            "precision": "float32",
            "description": "粤语语音识别"
        },
        "en": {
            "model_type": "faster_whisper",
            "model_size": "large-v3",
            "precision": "float16",
            "description": "英文语音识别，速度快"
        },
        "ja": {
            "model_type": "faster_whisper",
            "model_size": "large-v3",
            "precision": "float16", 
            "description": "日文语音识别"
        },
        "ko": {
            "model_type": "faster_whisper",
            "model_size": "large-v3",
            "precision": "float16",
            "description": "韩文语音识别"
        },
        "auto": {
            "model_type": "faster_whisper",
            "model_size": "large-v3",
            "precision": "float16",
            "description": "自动语言检测"
        }
    }
    
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
                'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
                'estimated_words': int(info.duration * 3),  # 估算词数（每秒3个词）
                'complexity': ASRUtils._estimate_complexity(info.duration, info.samplerate)
            }
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def _estimate_complexity(duration: float, sample_rate: int) -> str:
        """估算音频复杂度"""
        if duration < 10:
            return "simple"
        elif duration < 60:
            return "medium"
        elif duration < 300:
            return "complex"
        else:
            return "very_complex"
    
    @staticmethod
    def detect_language_from_filename(filename: str) -> str:
        """
        从文件名检测可能的语言
        
        Args:
            filename: 文件名
            
        Returns:
            str: 语言代码
        """
        filename_lower = filename.lower()
        
        # 中文相关关键词
        if any(keyword in filename_lower for keyword in ['zh', 'cn', 'chinese', '中文', '普通话']):
            return "zh"
        
        # 粤语相关关键词
        if any(keyword in filename_lower for keyword in ['yue', 'cantonese', '粤语', '广东话']):
            return "yue"
        
        # 英文相关关键词
        if any(keyword in filename_lower for keyword in ['en', 'eng', 'english', '英文']):
            return "en"
        
        # 日文相关关键词
        if any(keyword in filename_lower for keyword in ['ja', 'jp', 'japanese', '日文', '日语']):
            return "ja"
        
        # 韩文相关关键词
        if any(keyword in filename_lower for keyword in ['ko', 'kr', 'korean', '韩文', '韩语']):
            return "ko"
        
        return "auto"  # 默认自动检测
    
    @staticmethod
    def suggest_config_for_file(file_path: str, target_language: str = None) -> ASRConfig:
        """
        为单个文件建议ASR配置
        
        Args:
            file_path: 音频文件路径
            target_language: 目标语言（可选）
            
        Returns:
            ASRConfig: 建议的配置
        """
        # 分析音频文件
        info = ASRUtils.analyze_audio_file(file_path)
        
        if 'error' in info:
            return ASRConfig()  # 返回默认配置
        
        # 确定语言
        if target_language:
            language = target_language
        else:
            language = ASRUtils.detect_language_from_filename(os.path.basename(file_path))
        
        # 获取推荐配置
        recommendation = ASRUtils.MODEL_RECOMMENDATIONS.get(language, ASRUtils.MODEL_RECOMMENDATIONS["auto"])
        
        # 根据音频复杂度调整配置
        config = ASRConfig(
            model_type=recommendation["model_type"],
            model_size=recommendation["model_size"],
            language=language,
            precision=recommendation["precision"]
        )
        
        # 根据音频时长调整参数
        if info['duration'] > 300:  # 超过5分钟
            config.batch_size = 1  # 减少批大小
        elif info['duration'] < 10:  # 短音频
            config.beam_size = 3  # 减少束搜索
        
        return config
    
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
            'language_distribution': {},
            'files': [],
            'summary': {}
        }
        
        for file_path in audio_files:
            info = ASRUtils.analyze_audio_file(str(file_path))
            info['file_name'] = file_path.name
            info['file_path'] = str(file_path)
            
            if 'error' not in info:
                results['total_duration'] += info['duration']
                results['total_size_mb'] += info['file_size_mb']
                
                # 统计语言分布
                detected_lang = ASRUtils.detect_language_from_filename(file_path.name)
                results['language_distribution'][detected_lang] = results['language_distribution'].get(detected_lang, 0) + 1
            
            results['files'].append(info)
        
        # 生成摘要
        if results['total_files'] > 0:
            avg_duration = results['total_duration'] / results['total_files']
            
            # 确定主要语言
            main_language = max(results['language_distribution'].items(), key=lambda x: x[1])[0] if results['language_distribution'] else "auto"
            
            results['summary'] = {
                'average_duration': avg_duration,
                'total_duration_hours': results['total_duration'] / 3600,
                'total_size_gb': results['total_size_mb'] / 1024,
                'main_language': main_language,
                'suggested_config': ASRUtils.suggest_config_for_batch(results)
            }
        
        return results
    
    @staticmethod
    def suggest_config_for_batch(batch_info: Dict) -> ASRConfig:
        """
        为批量处理建议ASR配置
        
        Args:
            batch_info: 批量分析结果
            
        Returns:
            ASRConfig: 建议的配置
        """
        total_files = batch_info['total_files']
        total_duration = batch_info['total_duration']
        main_language = batch_info['summary'].get('main_language', 'auto')
        
        # 获取基础推荐配置
        recommendation = ASRUtils.MODEL_RECOMMENDATIONS.get(main_language, ASRUtils.MODEL_RECOMMENDATIONS["auto"])
        
        config = ASRConfig(
            model_type=recommendation["model_type"],
            model_size=recommendation["model_size"],
            language=main_language,
            precision=recommendation["precision"]
        )
        
        # 根据批量特征调整配置
        if total_files > 100 or total_duration > 3600:  # 大批量或长时间
            config.precision = "float16"  # 使用半精度加速
            config.batch_size = 1  # 减少内存使用
        
        return config
    
    @staticmethod
    def parse_recognition_result(list_file_path: str) -> List[Dict]:
        """
        解析识别结果文件
        
        Args:
            list_file_path: .list文件路径
            
        Returns:
            List[Dict]: 解析后的识别结果
        """
        results = []
        
        try:
            with open(list_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split('|')
                    if len(parts) >= 4:
                        result = {
                            'line_number': line_num,
                            'audio_path': parts[0],
                            'speaker': parts[1],
                            'language': parts[2],
                            'text': parts[3],
                            'text_length': len(parts[3]),
                            'word_count': len(parts[3].split()) if parts[2].upper() in ['EN'] else len(parts[3])
                        }
                        results.append(result)
                    else:
                        print(f"警告: 第{line_num}行格式不正确: {line}")
        
        except Exception as e:
            print(f"解析识别结果文件失败: {e}")
        
        return results
    
    @staticmethod
    def validate_recognition_result(list_file_path: str) -> Dict:
        """
        验证识别结果质量
        
        Args:
            list_file_path: .list文件路径
            
        Returns:
            Dict: 验证结果
        """
        if not os.path.exists(list_file_path):
            return {
                'valid': False,
                'error': '识别结果文件不存在'
            }
        
        results = ASRUtils.parse_recognition_result(list_file_path)
        
        if not results:
            return {
                'valid': False,
                'error': '识别结果为空'
            }
        
        # 统计信息
        total_entries = len(results)
        empty_text_count = sum(1 for r in results if not r['text'].strip())
        avg_text_length = sum(r['text_length'] for r in results) / total_entries
        
        # 语言分布
        language_dist = {}
        for result in results:
            lang = result['language']
            language_dist[lang] = language_dist.get(lang, 0) + 1
        
        # 质量评估
        quality_score = max(0, 100 - (empty_text_count / total_entries * 100))
        
        return {
            'valid': empty_text_count < total_entries * 0.1,  # 空文本少于10%认为有效
            'total_entries': total_entries,
            'empty_text_count': empty_text_count,
            'empty_text_ratio': empty_text_count / total_entries,
            'average_text_length': avg_text_length,
            'language_distribution': language_dist,
            'quality_score': quality_score,
            'file_size_kb': os.path.getsize(list_file_path) / 1024
        }
    
    @staticmethod
    def estimate_processing_time(file_info: Dict, config: ASRConfig) -> float:
        """
        估算处理时间
        
        Args:
            file_info: 文件信息
            config: ASR配置
            
        Returns:
            float: 估算时间（秒）
        """
        if 'error' in file_info:
            return 0
        
        duration = file_info['duration']
        
        # 基础处理时间（经验值）
        if config.model_type == "funasr":
            base_factor = 0.3  # FunASR相对较快
        else:
            base_factor = 0.5  # Faster Whisper
        
        # 模型大小影响
        size_factors = {
            "medium": 0.8,
            "large": 1.0,
            "large-v2": 1.1,
            "large-v3": 1.2,
            "large-v3-turbo": 0.9
        }
        size_factor = size_factors.get(config.model_size, 1.0)
        
        # 精度影响
        precision_factors = {
            "int8": 0.7,
            "float16": 0.8,
            "float32": 1.0
        }
        precision_factor = precision_factors.get(config.precision, 1.0)
        
        # 语言复杂度影响
        language_factors = {
            "zh": 1.2,  # 中文相对复杂
            "yue": 1.3,  # 粤语更复杂
            "en": 1.0,   # 英文基准
            "ja": 1.1,   # 日文稍复杂
            "ko": 1.1,   # 韩文稍复杂
            "auto": 1.3  # 自动检测需要额外时间
        }
        language_factor = language_factors.get(config.language, 1.0)
        
        estimated_time = duration * base_factor * size_factor * precision_factor * language_factor
        
        return max(estimated_time, 1.0)  # 至少1秒


def create_test_audio_list(audio_dir: str, output_path: str, speaker_name: str = "speaker", language: str = "zh"):
    """
    创建测试用的.list文件
    
    Args:
        audio_dir: 音频文件目录
        output_path: 输出.list文件路径
        speaker_name: 说话人名称
        language: 语言代码
    """
    audio_files = []
    for ext in ['.wav', '.mp3', '.flac']:
        audio_files.extend(Path(audio_dir).glob(f"*{ext}"))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, audio_file in enumerate(audio_files):
            # 生成示例文本
            sample_texts = {
                "zh": f"这是第{i+1}个测试音频文件。",
                "en": f"This is test audio file number {i+1}.",
                "ja": f"これは{i+1}番目のテストオーディオファイルです。"
            }
            text = sample_texts.get(language, f"Test audio {i+1}")
            
            line = f"{audio_file}|{speaker_name}|{language.upper()}|{text}"
            f.write(line + '\n')
    
    print(f"创建测试.list文件: {output_path}")
    return output_path