"""
GPT-SoVITS 音频特征提取工具函数

提供音频分析、配置建议等实用功能
"""

import os
import json
import librosa
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict

from .service import AudioFeaturesConfig


class AudioFeaturesUtils:
    """音频特征提取工具类"""
    
    @staticmethod
    def analyze_audio_file(file_path: str) -> Dict[str, Any]:
        """
        分析单个音频文件
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            分析结果
        """
        if not os.path.exists(file_path):
            return {"error": f"文件不存在: {file_path}"}
        
        try:
            # 加载音频信息
            y, sr = librosa.load(file_path, sr=None)
            duration = len(y) / sr
            
            # 音频统计
            max_amplitude = np.abs(y).max()
            rms_energy = np.sqrt(np.mean(y**2))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
            
            return {
                "file_path": file_path,
                "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
                "duration": duration,
                "sample_rate": sr,
                "channels": 1 if len(y.shape) == 1 else y.shape[0],
                "samples": len(y),
                "max_amplitude": float(max_amplitude),
                "rms_energy": float(rms_energy),
                "zero_crossing_rate": float(zero_crossing_rate),
                "is_too_loud": max_amplitude > 2.2,  # GPT-SoVITS过滤阈值
                "estimated_cnhubert_time": duration * 0.05,  # 估算CNHubert处理时间
                "estimated_speaker_time": duration * 0.02 if duration > 0 else 0  # 估算说话人特征时间
            }
            
        except Exception as e:
            return {"error": f"分析失败: {str(e)}"}
    
    @staticmethod
    def analyze_dataset_from_list(list_file: str, wav_dir: str = None) -> Dict[str, Any]:
        """
        从标注文件分析数据集
        
        Args:
            list_file: 标注文件路径
            wav_dir: 音频目录路径
            
        Returns:
            数据集分析结果
        """
        if not os.path.exists(list_file):
            return {"error": f"标注文件不存在: {list_file}"}
        
        try:
            with open(list_file, "r", encoding="utf8") as f:
                lines = f.read().strip().split("\n")
            
            total_lines = len(lines)
            valid_files = []
            invalid_files = []
            total_duration = 0
            total_size_mb = 0
            max_amplitudes = []
            speakers = set()
            languages = set()
            
            for line in lines:
                try:
                    parts = line.split("|")
                    if len(parts) >= 4:
                        wav_name, speaker, language, text = parts[:4]
                        
                        # 构建音频路径
                        if wav_dir and wav_dir.strip():
                            wav_path = os.path.join(wav_dir, os.path.basename(wav_name))
                        else:
                            wav_path = wav_name
                        
                        # 分析音频文件
                        if os.path.exists(wav_path):
                            audio_info = AudioFeaturesUtils.analyze_audio_file(wav_path)
                            if "error" not in audio_info:
                                valid_files.append({
                                    "wav_name": wav_name,
                                    "wav_path": wav_path,
                                    "speaker": speaker,
                                    "language": language,
                                    **audio_info
                                })
                                total_duration += audio_info["duration"]
                                total_size_mb += audio_info["file_size_mb"]
                                max_amplitudes.append(audio_info["max_amplitude"])
                                speakers.add(speaker)
                                languages.add(language)
                            else:
                                invalid_files.append({"wav_name": wav_name, "error": audio_info["error"]})
                        else:
                            invalid_files.append({"wav_name": wav_name, "error": "文件不存在"})
                    else:
                        invalid_files.append({"line": line, "error": "格式错误"})
                        
                except Exception as e:
                    invalid_files.append({"line": line, "error": str(e)})
            
            # 统计信息
            loud_files = [f for f in valid_files if f.get("is_too_loud", False)]
            
            return {
                "list_file": list_file,
                "wav_dir": wav_dir,
                "total_lines": total_lines,
                "valid_files": len(valid_files),
                "invalid_files": len(invalid_files),
                "speakers": list(speakers),
                "languages": list(languages),
                "total_duration": total_duration,
                "total_size_mb": total_size_mb,
                "avg_duration": total_duration / len(valid_files) if valid_files else 0,
                "max_amplitude_stats": {
                    "min": float(np.min(max_amplitudes)) if max_amplitudes else 0,
                    "max": float(np.max(max_amplitudes)) if max_amplitudes else 0,
                    "mean": float(np.mean(max_amplitudes)) if max_amplitudes else 0,
                    "std": float(np.std(max_amplitudes)) if max_amplitudes else 0
                },
                "loud_files_count": len(loud_files),
                "loud_files": [f["wav_name"] for f in loud_files[:10]],  # 只显示前10个
                "estimated_processing_time": {
                    "cnhubert": total_duration * 0.05,
                    "speaker": total_duration * 0.02,
                    "total": total_duration * 0.07
                },
                "file_details": valid_files[:5],  # 只显示前5个文件的详细信息
                "invalid_details": invalid_files[:5]  # 只显示前5个无效文件
            }
            
        except Exception as e:
            return {"error": f"分析失败: {str(e)}"}
    
    @staticmethod
    def suggest_processing_config(
        list_file: str,
        wav_dir: str = None,
        target_processing_time: float = None,
        available_memory_gb: float = None,
        version: str = "v2"
    ) -> AudioFeaturesConfig:
        """
        建议处理配置
        
        Args:
            list_file: 标注文件路径
            wav_dir: 音频目录路径
            target_processing_time: 目标处理时间（秒）
            available_memory_gb: 可用内存（GB）
            version: GPT-SoVITS版本
            
        Returns:
            建议的配置
        """
        analysis = AudioFeaturesUtils.analyze_dataset_from_list(list_file, wav_dir)
        
        if "error" in analysis:
            return AudioFeaturesConfig(version=version)
        
        config = AudioFeaturesConfig(version=version)
        
        # 根据数据量调整并行数
        valid_files = analysis["valid_files"]
        total_duration = analysis["total_duration"]
        
        if valid_files > 1000 or total_duration > 3600:  # 超过1000个文件或1小时
            config.n_parts = 8
        elif valid_files > 500 or total_duration > 1800:  # 超过500个文件或30分钟
            config.n_parts = 4
        elif valid_files > 100 or total_duration > 600:   # 超过100个文件或10分钟
            config.n_parts = 2
        else:
            config.n_parts = 1
        
        # 根据内存限制调整
        if available_memory_gb:
            if available_memory_gb < 4:
                config.n_parts = min(config.n_parts, 2)
                config.is_half = True
            elif available_memory_gb < 8:
                config.n_parts = min(config.n_parts, 4)
        
        # 根据目标时间调整
        if target_processing_time:
            estimated_time = analysis["estimated_processing_time"]["total"]
            if estimated_time > target_processing_time:
                suggested_parts = int(estimated_time / target_processing_time) + 1
                config.n_parts = min(suggested_parts, 16)
        
        # 根据版本设置说话人特征
        config.save_speaker = "Pro" in version
        
        # 根据音频质量调整参数
        max_amp_stats = analysis["max_amplitude_stats"]
        if max_amp_stats["max"] > 1.5:
            config.max_audio_value = 2.0  # 降低过滤阈值
        
        return config
    
    @staticmethod
    def validate_input_files(
        list_file: str,
        wav_dir: str = None,
        check_audio_existence: bool = True,
        check_audio_quality: bool = True
    ) -> Dict[str, Any]:
        """
        验证输入文件
        
        Args:
            list_file: 标注文件路径
            wav_dir: 音频目录路径
            check_audio_existence: 是否检查音频文件存在性
            check_audio_quality: 是否检查音频质量
            
        Returns:
            验证结果
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        # 检查标注文件
        if not os.path.exists(list_file):
            result["valid"] = False
            result["errors"].append(f"标注文件不存在: {list_file}")
            return result
        
        # 检查音频目录
        if wav_dir and not os.path.exists(wav_dir):
            result["valid"] = False
            result["errors"].append(f"音频目录不存在: {wav_dir}")
            return result
        
        # 分析数据集
        if check_audio_existence or check_audio_quality:
            analysis = AudioFeaturesUtils.analyze_dataset_from_list(list_file, wav_dir)
            if "error" in analysis:
                result["valid"] = False
                result["errors"].append(analysis["error"])
                return result
            
            result["statistics"] = analysis
            
            # 检查文件存在性
            if analysis["invalid_files"] > 0:
                result["warnings"].append(f"找不到 {analysis['invalid_files']} 个音频文件")
                
                if analysis["invalid_files"] > analysis["valid_files"] * 0.1:  # 超过10%缺失
                    result["errors"].append("缺失音频文件过多，可能影响处理质量")
            
            # 检查音频质量
            if check_audio_quality:
                if analysis["loud_files_count"] > 0:
                    result["warnings"].append(f"检测到 {analysis['loud_files_count']} 个音频幅值过大的文件")
                
                avg_duration = analysis["avg_duration"]
                if avg_duration < 1.0:
                    result["warnings"].append(f"平均音频时长过短: {avg_duration:.2f}秒")
                elif avg_duration > 30.0:
                    result["warnings"].append(f"平均音频时长过长: {avg_duration:.2f}秒")
        
        return result
    
    @staticmethod
    def estimate_processing_time(
        list_file: str,
        wav_dir: str = None,
        config: AudioFeaturesConfig = None
    ) -> Dict[str, float]:
        """
        估算处理时间
        
        Args:
            list_file: 标注文件路径
            wav_dir: 音频目录路径
            config: 处理配置
            
        Returns:
            时间估算结果
        """
        analysis = AudioFeaturesUtils.analyze_dataset_from_list(list_file, wav_dir)
        
        if "error" in analysis:
            return {"error": analysis["error"]}
        
        if config is None:
            config = AudioFeaturesConfig()
        
        total_duration = analysis["total_duration"]
        valid_files = analysis["valid_files"]
        
        # 基础处理时间（每秒音频）
        cnhubert_time_per_sec = 0.05  # CNHubert处理时间
        speaker_time_per_sec = 0.02   # 说话人特征处理时间
        io_time_per_file = 0.1        # 文件I/O时间
        
        # 计算总时间
        cnhubert_time = total_duration * cnhubert_time_per_sec if config.save_cnhubert else 0
        speaker_time = total_duration * speaker_time_per_sec if config.save_speaker else 0
        io_time = valid_files * io_time_per_file
        
        total_time = cnhubert_time + speaker_time + io_time
        
        # 考虑并行处理
        parallel_factor = 1.0 / config.n_parts if config.n_parts > 1 else 1.0
        parallel_overhead = 0.1 if config.n_parts > 1 else 0  # 并行开销
        
        estimated_time = (total_time * parallel_factor) + parallel_overhead
        
        return {
            "estimated_total_time": estimated_time,
            "cnhubert_time": cnhubert_time,
            "speaker_time": speaker_time,
            "io_time": io_time,
            "parallel_speedup": 1.0 / parallel_factor if parallel_factor < 1 else 1.0,
            "includes_cnhubert": config.save_cnhubert,
            "includes_speaker": config.save_speaker,
            "total_files": valid_files,
            "total_duration": total_duration
        }
    
    @staticmethod
    def create_processing_report(
        input_file: str,
        output_dir: str,
        processing_result: Dict[str, Any]
    ) -> str:
        """
        创建处理报告
        
        Args:
            input_file: 输入文件路径
            output_dir: 输出目录
            processing_result: 处理结果
            
        Returns:
            报告文件路径
        """
        report_path = os.path.join(output_dir, "audio_features_report.json")
        
        # 分析输入文件
        input_analysis = AudioFeaturesUtils.analyze_dataset_from_list(input_file)
        
        report = {
            "timestamp": str(torch.tensor(0).item()),  # 简单的时间戳
            "input_analysis": input_analysis,
            "processing_result": processing_result,
            "output_files": {
                "cnhubert_dir": os.path.join(output_dir, "4-cnhubert"),
                "wav32k_dir": os.path.join(output_dir, "5-wav32k"),
                "speaker_dir": os.path.join(output_dir, "7-sv_cn"),
                "report_file": report_path
            }
        }
        
        with open(report_path, "w", encoding="utf8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report_path
    
    @staticmethod
    def check_output_completeness(
        output_dir: str,
        expected_files: List[str],
        check_cnhubert: bool = True,
        check_wav32k: bool = True,
        check_speaker: bool = False
    ) -> Dict[str, Any]:
        """
        检查输出文件完整性
        
        Args:
            output_dir: 输出目录
            expected_files: 期望的文件列表
            check_cnhubert: 是否检查CNHubert特征
            check_wav32k: 是否检查32kHz音频
            check_speaker: 是否检查说话人特征
            
        Returns:
            完整性检查结果
        """
        result = {
            "complete": True,
            "missing_files": [],
            "statistics": {}
        }
        
        # 检查各类输出文件
        checks = []
        if check_cnhubert:
            checks.append(("cnhubert", "4-cnhubert", ".pt"))
        if check_wav32k:
            checks.append(("wav32k", "5-wav32k", ".wav"))
        if check_speaker:
            checks.append(("speaker", "7-sv_cn", ".pt"))
        
        for check_name, dir_name, ext in checks:
            check_dir = os.path.join(output_dir, dir_name)
            if not os.path.exists(check_dir):
                result["complete"] = False
                result["missing_files"].append(f"目录不存在: {check_dir}")
                continue
            
            existing_files = set()
            for file in os.listdir(check_dir):
                if file.endswith(ext):
                    base_name = file[:-len(ext)]
                    existing_files.add(base_name)
            
            missing = []
            for expected_file in expected_files:
                base_name = os.path.splitext(os.path.basename(expected_file))[0]
                if base_name not in existing_files:
                    missing.append(f"{base_name}{ext}")
            
            result["statistics"][check_name] = {
                "expected": len(expected_files),
                "found": len(existing_files),
                "missing": len(missing)
            }
            
            if missing:
                result["complete"] = False
                result["missing_files"].extend([f"{dir_name}/{f}" for f in missing[:5]])  # 只显示前5个
        
        return result
    
    @staticmethod
    def get_model_info(cnhubert_path: str = None, speaker_path: str = None) -> Dict[str, Any]:
        """
        获取模型信息
        
        Args:
            cnhubert_path: CNHubert模型路径
            speaker_path: 说话人模型路径
            
        Returns:
            模型信息
        """
        info = {
            "cnhubert": {"available": False},
            "speaker": {"available": False}
        }
        
        # 检查CNHubert模型
        if cnhubert_path and os.path.exists(cnhubert_path):
            try:
                config_path = os.path.join(cnhubert_path, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        config = json.load(f)
                    info["cnhubert"] = {
                        "available": True,
                        "path": cnhubert_path,
                        "config": config
                    }
                else:
                    info["cnhubert"] = {
                        "available": True,
                        "path": cnhubert_path,
                        "config": "配置文件不存在"
                    }
            except Exception as e:
                info["cnhubert"] = {
                    "available": False,
                    "error": str(e)
                }
        
        # 检查说话人模型
        if speaker_path and os.path.exists(speaker_path):
            try:
                model_size = os.path.getsize(speaker_path) / (1024 * 1024)  # MB
                info["speaker"] = {
                    "available": True,
                    "path": speaker_path,
                    "size_mb": model_size
                }
            except Exception as e:
                info["speaker"] = {
                    "available": False,
                    "error": str(e)
                }
        
        return info