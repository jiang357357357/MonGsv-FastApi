"""
GPT-SoVITS 语义编码工具函数

提供语义编码相关的辅助功能
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict

from .service import SemanticEncodingConfig


class SemanticEncodingUtils:
    """语义编码工具类"""
    
    @staticmethod
    def analyze_input_data(input_text_file: str, cnhubert_dir: str) -> Dict[str, Any]:
        """
        分析输入数据
        
        Args:
            input_text_file: 输入标注文件
            cnhubert_dir: CNHubert特征目录
            
        Returns:
            分析结果
        """
        try:
            analysis = {
                "total_lines": 0,
                "valid_lines": 0,
                "invalid_lines": 0,
                "missing_cnhubert": 0,
                "speakers": set(),
                "languages": set(),
                "files": []
            }
            
            if not os.path.exists(input_text_file):
                return {"error": f"输入文件不存在: {input_text_file}"}
            
            if not os.path.exists(cnhubert_dir):
                return {"error": f"CNHubert目录不存在: {cnhubert_dir}"}
            
            with open(input_text_file, "r", encoding="utf8") as f:
                lines = f.read().strip().split("\n")
            
            analysis["total_lines"] = len(lines)
            
            for line in lines:
                try:
                    parts = line.split("|")
                    if len(parts) >= 4:
                        wav_name, spk_name, language, text = parts[:4]
                        wav_name = os.path.basename(wav_name)
                        wav_name_no_ext = os.path.splitext(wav_name)[0]
                        
                        # 检查CNHubert特征文件
                        cnhubert_path = os.path.join(cnhubert_dir, f"{wav_name_no_ext}.pt")
                        has_cnhubert = os.path.exists(cnhubert_path)
                        
                        if has_cnhubert:
                            analysis["valid_lines"] += 1
                            analysis["speakers"].add(spk_name)
                            analysis["languages"].add(language)
                            
                            # 获取特征文件大小
                            try:
                                cnhubert_size = os.path.getsize(cnhubert_path)
                                cnhubert_tensor = torch.load(cnhubert_path, map_location="cpu")
                                cnhubert_shape = list(cnhubert_tensor.shape)
                            except:
                                cnhubert_size = 0
                                cnhubert_shape = []
                            
                            analysis["files"].append({
                                "wav_name": wav_name_no_ext,
                                "speaker": spk_name,
                                "language": language,
                                "text_length": len(text),
                                "cnhubert_size": cnhubert_size,
                                "cnhubert_shape": cnhubert_shape
                            })
                        else:
                            analysis["missing_cnhubert"] += 1
                            analysis["invalid_lines"] += 1
                    else:
                        analysis["invalid_lines"] += 1
                        
                except Exception as e:
                    analysis["invalid_lines"] += 1
            
            # 转换集合为列表
            analysis["speakers"] = sorted(list(analysis["speakers"]))
            analysis["languages"] = sorted(list(analysis["languages"]))
            
            # 统计信息
            analysis["speaker_count"] = len(analysis["speakers"])
            analysis["language_count"] = len(analysis["languages"])
            
            if analysis["files"]:
                # CNHubert特征统计
                cnhubert_sizes = [f["cnhubert_size"] for f in analysis["files"]]
                analysis["cnhubert_stats"] = {
                    "total_size_mb": sum(cnhubert_sizes) / (1024 * 1024),
                    "avg_size_kb": sum(cnhubert_sizes) / len(cnhubert_sizes) / 1024,
                    "min_size_kb": min(cnhubert_sizes) / 1024,
                    "max_size_kb": max(cnhubert_sizes) / 1024
                }
                
                # 文本长度统计
                text_lengths = [f["text_length"] for f in analysis["files"]]
                analysis["text_stats"] = {
                    "avg_length": sum(text_lengths) / len(text_lengths),
                    "min_length": min(text_lengths),
                    "max_length": max(text_lengths)
                }
            
            return analysis
            
        except Exception as e:
            return {"error": f"分析失败: {str(e)}"}
    
    @staticmethod
    def estimate_processing_time(input_text_file: str, 
                               cnhubert_dir: str,
                               config: SemanticEncodingConfig) -> Dict[str, Any]:
        """
        估算处理时间
        
        Args:
            input_text_file: 输入文件
            cnhubert_dir: CNHubert目录
            config: 处理配置
            
        Returns:
            时间估算结果
        """
        try:
            analysis = SemanticEncodingUtils.analyze_input_data(input_text_file, cnhubert_dir)
            
            if "error" in analysis:
                return analysis
            
            valid_files = analysis["valid_lines"]
            
            if valid_files == 0:
                return {"error": "没有有效文件"}
            
            # 基础处理时间（秒/文件）
            base_times = {
                "cpu": 2.0,      # CPU处理时间
                "cuda": 0.3,     # GPU处理时间
            }
            
            device = config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
            base_time = base_times.get(device, base_times["cpu"])
            
            # 半精度加速
            if config.is_half and device == "cuda":
                base_time *= 0.7
            
            # 并行加速
            parallel_speedup = min(config.n_parts, valid_files) * 0.8  # 80%效率
            
            # 总时间估算
            total_time = (valid_files * base_time) / parallel_speedup
            
            # I/O时间
            io_time = valid_files * 0.1  # 每文件0.1秒I/O
            
            estimated_total = total_time + io_time
            
            return {
                "valid_files": valid_files,
                "base_time_per_file": base_time,
                "parallel_speedup": parallel_speedup,
                "processing_time": total_time,
                "io_time": io_time,
                "estimated_total_time": estimated_total,
                "device": device,
                "estimated_completion": f"{estimated_total/60:.1f} 分钟" if estimated_total > 60 else f"{estimated_total:.1f} 秒"
            }
            
        except Exception as e:
            return {"error": f"估算失败: {str(e)}"}
    
    @staticmethod
    def suggest_processing_config(input_text_file: str,
                                cnhubert_dir: str,
                                target_processing_time: float = 300.0,
                                available_memory_gb: float = 8.0) -> SemanticEncodingConfig:
        """
        建议处理配置
        
        Args:
            input_text_file: 输入文件
            cnhubert_dir: CNHubert目录
            target_processing_time: 目标处理时间（秒）
            available_memory_gb: 可用内存（GB）
            
        Returns:
            建议的配置
        """
        config = SemanticEncodingConfig()
        
        try:
            analysis = SemanticEncodingUtils.analyze_input_data(input_text_file, cnhubert_dir)
            
            if "error" in analysis:
                return config
            
            valid_files = analysis["valid_lines"]
            
            # 设备选择
            if torch.cuda.is_available():
                config.device = "cuda"
                config.is_half = True
            else:
                config.device = "cpu"
                config.is_half = False
            
            # 并行数建议
            if valid_files <= 10:
                config.n_parts = 1
            elif valid_files <= 50:
                config.n_parts = 2
            elif valid_files <= 200:
                config.n_parts = 4
            else:
                config.n_parts = min(8, valid_files // 50)
            
            # 内存限制调整
            if available_memory_gb < 4:
                config.is_half = True
                config.n_parts = min(config.n_parts, 2)
            elif available_memory_gb < 8:
                config.n_parts = min(config.n_parts, 4)
            
            return config
            
        except Exception as e:
            print(f"配置建议失败: {e}")
            return config
    
    @staticmethod
    def validate_input_files(input_text_file: str, 
                           cnhubert_dir: str,
                           check_model_files: bool = True) -> Dict[str, Any]:
        """
        验证输入文件
        
        Args:
            input_text_file: 输入标注文件
            cnhubert_dir: CNHubert特征目录
            check_model_files: 是否检查模型文件
            
        Returns:
            验证结果
        """
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        try:
            # 检查输入文件
            if not os.path.exists(input_text_file):
                validation["errors"].append(f"输入文件不存在: {input_text_file}")
                validation["valid"] = False
            
            # 检查CNHubert目录
            if not os.path.exists(cnhubert_dir):
                validation["errors"].append(f"CNHubert目录不存在: {cnhubert_dir}")
                validation["valid"] = False
            
            if not validation["valid"]:
                return validation
            
            # 分析数据
            analysis = SemanticEncodingUtils.analyze_input_data(input_text_file, cnhubert_dir)
            
            if "error" in analysis:
                validation["errors"].append(analysis["error"])
                validation["valid"] = False
                return validation
            
            # 统计信息
            validation["statistics"] = {
                "total_lines": analysis["total_lines"],
                "valid_files": analysis["valid_lines"],
                "invalid_files": analysis["invalid_lines"],
                "missing_cnhubert": analysis["missing_cnhubert"],
                "speakers": analysis["speakers"],
                "languages": analysis["languages"]
            }
            
            # 检查有效文件数
            if analysis["valid_lines"] == 0:
                validation["errors"].append("没有找到有效的文件")
                validation["valid"] = False
            
            # 警告检查
            if analysis["missing_cnhubert"] > 0:
                validation["warnings"].append(f"缺少 {analysis['missing_cnhubert']} 个CNHubert特征文件")
            
            if analysis["invalid_lines"] > analysis["total_lines"] * 0.1:
                validation["warnings"].append(f"无效行数较多: {analysis['invalid_lines']}/{analysis['total_lines']}")
            
            # 检查模型文件
            if check_model_files:
                default_config = SemanticEncodingConfig()
                
                if not os.path.exists(default_config.pretrained_s2G):
                    validation["warnings"].append(f"默认预训练模型不存在: {default_config.pretrained_s2G}")
                
                if not os.path.exists(default_config.s2config_path):
                    validation["warnings"].append(f"默认配置文件不存在: {default_config.s2config_path}")
            
            return validation
            
        except Exception as e:
            validation["errors"].append(f"验证过程出错: {str(e)}")
            validation["valid"] = False
            return validation
    
    @staticmethod
    def check_output_completeness(output_dir: str, 
                                expected_files: List[str],
                                output_format: str = "tsv") -> Dict[str, Any]:
        """
        检查输出完整性
        
        Args:
            output_dir: 输出目录
            expected_files: 期望的文件列表
            output_format: 输出格式
            
        Returns:
            完整性检查结果
        """
        result = {
            "complete": False,
            "missing_files": [],
            "extra_files": [],
            "statistics": {}
        }
        
        try:
            if not os.path.exists(output_dir):
                result["missing_files"] = expected_files
                return result
            
            # 检查主输出文件
            if output_format == "tsv":
                output_file = os.path.join(output_dir, "6-name2semantic.tsv")
            else:
                output_file = os.path.join(output_dir, "6-name2semantic.json")
            
            if not os.path.exists(output_file):
                result["missing_files"].append(os.path.basename(output_file))
                return result
            
            # 解析输出文件检查内容
            if output_format == "tsv":
                with open(output_file, "r", encoding="utf8") as f:
                    lines = f.read().strip().split("\n")
                
                found_files = set()
                for line in lines:
                    if "\t" in line:
                        wav_name = line.split("\t")[0]
                        found_files.add(wav_name)
            else:
                with open(output_file, "r", encoding="utf8") as f:
                    data = json.load(f)
                found_files = set(data.keys())
            
            # 比较期望和实际文件
            expected_set = set(expected_files)
            missing = expected_set - found_files
            extra = found_files - expected_set
            
            result["missing_files"] = sorted(list(missing))
            result["extra_files"] = sorted(list(extra))
            result["complete"] = len(missing) == 0
            
            result["statistics"] = {
                "expected_count": len(expected_files),
                "found_count": len(found_files),
                "missing_count": len(missing),
                "extra_count": len(extra),
                "completion_rate": len(found_files) / len(expected_files) if expected_files else 0
            }
            
            return result
            
        except Exception as e:
            result["error"] = f"检查失败: {str(e)}"
            return result
    
    @staticmethod
    def get_supported_versions() -> Dict[str, Dict[str, Any]]:
        """
        获取支持的模型版本信息
        
        Returns:
            版本信息字典
        """
        return {
            "v1": {
                "description": "GPT-SoVITS v1 版本",
                "model_class": "SynthesizerTrn",
                "typical_size_mb": "80-100",
                "features": ["基础语义编码"]
            },
            "v2": {
                "description": "GPT-SoVITS v2 版本", 
                "model_class": "SynthesizerTrn",
                "typical_size_mb": "100-700",
                "features": ["改进的语义编码", "更好的音质"]
            },
            "v3": {
                "description": "GPT-SoVITS v3 版本",
                "model_class": "SynthesizerTrnV3", 
                "typical_size_mb": "700+",
                "features": ["CFM架构", "外部声码器", "更高质量"]
            },
            "v4": {
                "description": "GPT-SoVITS v4 版本",
                "model_class": "SynthesizerTrnV3",
                "typical_size_mb": "700+", 
                "features": ["v3的改进版", "更快推理"]
            },
            "v2Pro": {
                "description": "GPT-SoVITS v2Pro 版本",
                "model_class": "SynthesizerTrn",
                "typical_size_mb": "100-700",
                "features": ["说话人特征", "多说话人支持"]
            },
            "v2ProPlus": {
                "description": "GPT-SoVITS v2ProPlus 版本", 
                "model_class": "SynthesizerTrn",
                "typical_size_mb": "100-700",
                "features": ["v2Pro增强版", "更好的多说话人"]
            }
        }
    
    @staticmethod
    def batch_analyze_directory(directory: str) -> Dict[str, Any]:
        """
        批量分析目录中的标注文件
        
        Args:
            directory: 目录路径
            
        Returns:
            批量分析结果
        """
        result = {
            "total_files": 0,
            "valid_files": 0,
            "invalid_files": 0,
            "analyses": {},
            "summary": {
                "total_lines": 0,
                "valid_lines": 0,
                "speakers": set(),
                "languages": set()
            }
        }
        
        try:
            if not os.path.exists(directory):
                return {"error": f"目录不存在: {directory}"}
            
            # 查找所有.list和.txt文件
            list_files = []
            for ext in ["*.list", "*.txt"]:
                list_files.extend(Path(directory).glob(ext))
            
            result["total_files"] = len(list_files)
            
            for list_file in list_files:
                # 假设CNHubert目录在同级
                cnhubert_dir = os.path.join(os.path.dirname(list_file), "4-cnhubert")
                
                analysis = SemanticEncodingUtils.analyze_input_data(str(list_file), cnhubert_dir)
                
                if "error" not in analysis:
                    result["valid_files"] += 1
                    result["analyses"][str(list_file)] = analysis
                    
                    # 累计统计
                    result["summary"]["total_lines"] += analysis["total_lines"]
                    result["summary"]["valid_lines"] += analysis["valid_lines"]
                    result["summary"]["speakers"].update(analysis["speakers"])
                    result["summary"]["languages"].update(analysis["languages"])
                else:
                    result["invalid_files"] += 1
                    result["analyses"][str(list_file)] = analysis
            
            # 转换集合为列表
            result["summary"]["speakers"] = sorted(list(result["summary"]["speakers"]))
            result["summary"]["languages"] = sorted(list(result["summary"]["languages"]))
            
            return result
            
        except Exception as e:
            return {"error": f"批量分析失败: {str(e)}"}