"""
GPT-SoVITS 文本处理工具函数

提供文本分析、配置建议等实用功能
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict

from .service import TextProcessingConfig


class TextProcessingUtils:
    """文本处理工具类"""
    
    @staticmethod
    def analyze_text_file(file_path: str) -> Dict[str, Any]:
        """
        分析文本标注文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            分析结果
        """
        if not os.path.exists(file_path):
            return {"error": f"文件不存在: {file_path}"}
        
        try:
            with open(file_path, "r", encoding="utf8") as f:
                lines = f.read().strip().split("\n")
            
            total_lines = len(lines)
            valid_lines = 0
            languages = set()
            speakers = set()
            total_text_length = 0
            
            for line in lines:
                try:
                    parts = line.split("|")
                    if len(parts) >= 4:
                        wav_name, speaker, language, text = parts[:4]
                        valid_lines += 1
                        languages.add(language)
                        speakers.add(speaker)
                        total_text_length += len(text)
                except:
                    continue
            
            return {
                "file_path": file_path,
                "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
                "total_lines": total_lines,
                "valid_lines": valid_lines,
                "invalid_lines": total_lines - valid_lines,
                "languages": list(languages),
                "speakers": list(speakers),
                "total_text_length": total_text_length,
                "avg_text_length": total_text_length / valid_lines if valid_lines > 0 else 0,
                "has_chinese": any("zh" in lang.lower() for lang in languages),
                "has_multilingual": len(languages) > 1
            }
            
        except Exception as e:
            return {"error": f"分析失败: {str(e)}"}
    
    @staticmethod
    def suggest_processing_config(
        file_path: str,
        target_processing_time: float = None,
        available_memory_gb: float = None
    ) -> TextProcessingConfig:
        """
        建议处理配置
        
        Args:
            file_path: 输入文件路径
            target_processing_time: 目标处理时间（秒）
            available_memory_gb: 可用内存（GB）
            
        Returns:
            建议的配置
        """
        analysis = TextProcessingUtils.analyze_text_file(file_path)
        
        if "error" in analysis:
            return TextProcessingConfig()
        
        config = TextProcessingConfig()
        
        # 根据数据量调整并行数
        valid_lines = analysis["valid_lines"]
        
        if valid_lines > 1000:
            config.n_parts = 8
        elif valid_lines > 500:
            config.n_parts = 4
        elif valid_lines > 100:
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
            estimated_time_per_line = 0.1  # 估计每行处理时间
            estimated_total_time = valid_lines * estimated_time_per_line
            
            if estimated_total_time > target_processing_time:
                suggested_parts = int(estimated_total_time / target_processing_time) + 1
                config.n_parts = min(suggested_parts, 16)
        
        return config
    
    @staticmethod
    def validate_input_files(
        text_file: str,
        wav_dir: str,
        check_audio_existence: bool = True
    ) -> Dict[str, Any]:
        """
        验证输入文件
        
        Args:
            text_file: 文本标注文件
            wav_dir: 音频目录
            check_audio_existence: 是否检查音频文件存在性
            
        Returns:
            验证结果
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        # 检查文本文件
        if not os.path.exists(text_file):
            result["valid"] = False
            result["errors"].append(f"文本文件不存在: {text_file}")
            return result
        
        # 检查音频目录
        if not os.path.exists(wav_dir):
            result["valid"] = False
            result["errors"].append(f"音频目录不存在: {wav_dir}")
            return result
        
        # 分析文本文件
        analysis = TextProcessingUtils.analyze_text_file(text_file)
        if "error" in analysis:
            result["valid"] = False
            result["errors"].append(analysis["error"])
            return result
        
        result["statistics"] = analysis
        
        # 检查音频文件存在性
        if check_audio_existence:
            missing_files = []
            
            with open(text_file, "r", encoding="utf8") as f:
                lines = f.read().strip().split("\n")
            
            for line in lines:
                try:
                    parts = line.split("|")
                    if len(parts) >= 4:
                        wav_name = parts[0]
                        wav_path = os.path.join(wav_dir, os.path.basename(wav_name))
                        
                        if not os.path.exists(wav_path):
                            missing_files.append(wav_name)
                except:
                    continue
            
            if missing_files:
                result["warnings"].append(f"找不到 {len(missing_files)} 个音频文件")
                result["statistics"]["missing_audio_files"] = len(missing_files)
                
                if len(missing_files) > analysis["valid_lines"] * 0.1:  # 超过10%缺失
                    result["errors"].append("缺失音频文件过多，可能影响处理质量")
        
        return result
    
    @staticmethod
    def estimate_processing_time(
        text_file: str,
        config: TextProcessingConfig = None
    ) -> Dict[str, float]:
        """
        估算处理时间
        
        Args:
            text_file: 文本文件路径
            config: 处理配置
            
        Returns:
            时间估算结果
        """
        analysis = TextProcessingUtils.analyze_text_file(text_file)
        
        if "error" in analysis:
            return {"error": analysis["error"]}
        
        if config is None:
            config = TextProcessingConfig()
        
        valid_lines = analysis["valid_lines"]
        has_chinese = analysis["has_chinese"]
        
        # 基础处理时间（每行）
        base_time_per_line = 0.05  # 音素转换
        bert_time_per_line = 0.1 if has_chinese else 0  # BERT特征提取
        
        total_time_per_line = base_time_per_line + bert_time_per_line
        
        # 考虑并行处理
        parallel_factor = 1.0 / config.n_parts if config.n_parts > 1 else 1.0
        parallel_overhead = 0.1 if config.n_parts > 1 else 0  # 并行开销
        
        estimated_time = (valid_lines * total_time_per_line * parallel_factor) + parallel_overhead
        
        return {
            "estimated_total_time": estimated_time,
            "estimated_time_per_line": total_time_per_line,
            "parallel_speedup": 1.0 / parallel_factor if parallel_factor < 1 else 1.0,
            "includes_bert": has_chinese,
            "total_lines": valid_lines
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
        report_path = os.path.join(output_dir, "text_processing_report.json")
        
        # 分析输入文件
        input_analysis = TextProcessingUtils.analyze_text_file(input_file)
        
        report = {
            "timestamp": str(pd.Timestamp.now()) if 'pd' in globals() else "unknown",
            "input_analysis": input_analysis,
            "processing_result": processing_result,
            "output_files": {
                "text_file": os.path.join(output_dir, "2-name2text.txt"),
                "bert_dir": os.path.join(output_dir, "3-bert"),
                "report_file": report_path
            }
        }
        
        with open(report_path, "w", encoding="utf8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report_path
    
    @staticmethod
    def batch_analyze_directory(directory: str) -> Dict[str, Any]:
        """
        批量分析目录中的文本文件
        
        Args:
            directory: 目录路径
            
        Returns:
            批量分析结果
        """
        if not os.path.exists(directory):
            return {"error": f"目录不存在: {directory}"}
        
        text_files = []
        for ext in [".txt", ".list"]:
            text_files.extend(Path(directory).glob(f"*{ext}"))
        
        if not text_files:
            return {"error": "目录中没有找到文本文件"}
        
        results = {
            "directory": directory,
            "total_files": len(text_files),
            "files": [],
            "summary": {
                "total_lines": 0,
                "total_speakers": set(),
                "total_languages": set(),
                "has_chinese": False
            }
        }
        
        for file_path in text_files:
            analysis = TextProcessingUtils.analyze_text_file(str(file_path))
            results["files"].append(analysis)
            
            if "error" not in analysis:
                results["summary"]["total_lines"] += analysis["valid_lines"]
                results["summary"]["total_speakers"].update(analysis["speakers"])
                results["summary"]["total_languages"].update(analysis["languages"])
                if analysis["has_chinese"]:
                    results["summary"]["has_chinese"] = True
        
        # 转换set为list以便JSON序列化
        results["summary"]["total_speakers"] = list(results["summary"]["total_speakers"])
        results["summary"]["total_languages"] = list(results["summary"]["total_languages"])
        
        return results
    
    @staticmethod
    def suggest_batch_config(batch_analysis: Dict[str, Any]) -> TextProcessingConfig:
        """
        为批量处理建议配置
        
        Args:
            batch_analysis: 批量分析结果
            
        Returns:
            建议的配置
        """
        if "error" in batch_analysis:
            return TextProcessingConfig()
        
        total_lines = batch_analysis["summary"]["total_lines"]
        has_chinese = batch_analysis["summary"]["has_chinese"]
        
        config = TextProcessingConfig()
        
        # 根据总数据量调整并行数
        if total_lines > 5000:
            config.n_parts = 16
        elif total_lines > 2000:
            config.n_parts = 8
        elif total_lines > 500:
            config.n_parts = 4
        elif total_lines > 100:
            config.n_parts = 2
        else:
            config.n_parts = 1
        
        # 如果没有中文，可以跳过BERT
        if not has_chinese:
            config.bert_pretrained_dir = ""  # 空路径表示跳过BERT
        
        return config