#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 模型管理器

负责模型的加载、切换和管理
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel


@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    gpt_path: str
    sovits_path: str
    version: str
    description: str = ""
    created_time: Optional[datetime] = None
    file_size: Optional[int] = None
    is_loaded: bool = False


class ModelConfig(BaseModel):
    """模型配置"""
    name: str
    gpt_path: str
    sovits_path: str
    version: str = "auto"
    description: str = ""
    
    # 预训练模型路径
    bert_path: str = ""
    cnhubert_path: str = ""
    
    # 推理参数
    default_language: str = "zh"
    default_top_k: int = 20
    default_top_p: float = 0.6
    default_temperature: float = 0.6


class ModelManager:
    """模型管理器"""
    
    def __init__(self, models_dir: str = None, config_file: str = None):
        """
        初始化模型管理器
        
        Args:
            models_dir: 模型存储目录
            config_file: 模型配置文件路径
        """
        self.models_dir = models_dir or "models"
        self.config_file = config_file or os.path.join(self.models_dir, "models.json")
        
        # 确保目录存在
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 模型注册表
        self.models: Dict[str, ModelInfo] = {}
        self.current_model: Optional[str] = None
        
        # 加载模型配置
        self.load_models_config()
    
    def load_models_config(self):
        """加载模型配置文件"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                for model_data in config_data.get("models", []):
                    model_info = ModelInfo(
                        name=model_data["name"],
                        gpt_path=model_data["gpt_path"],
                        sovits_path=model_data["sovits_path"],
                        version=model_data.get("version", "auto"),
                        description=model_data.get("description", ""),
                        created_time=datetime.fromisoformat(model_data["created_time"]) if "created_time" in model_data else None
                    )
                    
                    # 检查文件是否存在并获取大小
                    if os.path.exists(model_info.gpt_path) and os.path.exists(model_info.sovits_path):
                        gpt_size = os.path.getsize(model_info.gpt_path)
                        sovits_size = os.path.getsize(model_info.sovits_path)
                        model_info.file_size = gpt_size + sovits_size
                    
                    self.models[model_info.name] = model_info
                
                self.current_model = config_data.get("current_model")
                
            except Exception as e:
                print(f"加载模型配置失败: {e}")
    
    def save_models_config(self):
        """保存模型配置文件"""
        try:
            config_data = {
                "current_model": self.current_model,
                "models": []
            }
            
            for model_info in self.models.values():
                model_data = {
                    "name": model_info.name,
                    "gpt_path": model_info.gpt_path,
                    "sovits_path": model_info.sovits_path,
                    "version": model_info.version,
                    "description": model_info.description
                }
                
                if model_info.created_time:
                    model_data["created_time"] = model_info.created_time.isoformat()
                
                config_data["models"].append(model_data)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"保存模型配置失败: {e}")
    
    def register_model(self, config: ModelConfig) -> bool:
        """
        注册新模型
        
        Args:
            config: 模型配置
            
        Returns:
            bool: 是否注册成功
        """
        try:
            # 检查模型文件是否存在
            if not os.path.exists(config.gpt_path):
                raise FileNotFoundError(f"GPT模型文件不存在: {config.gpt_path}")
            
            if not os.path.exists(config.sovits_path):
                raise FileNotFoundError(f"SoVITS模型文件不存在: {config.sovits_path}")
            
            # 检查模型名称是否已存在
            if config.name in self.models:
                raise ValueError(f"模型名称已存在: {config.name}")
            
            # 自动检测版本
            if config.version == "auto":
                config.version = self._detect_model_version(config.sovits_path)
            
            # 创建模型信息
            model_info = ModelInfo(
                name=config.name,
                gpt_path=config.gpt_path,
                sovits_path=config.sovits_path,
                version=config.version,
                description=config.description,
                created_time=datetime.now()
            )
            
            # 获取文件大小
            gpt_size = os.path.getsize(config.gpt_path)
            sovits_size = os.path.getsize(config.sovits_path)
            model_info.file_size = gpt_size + sovits_size
            
            # 注册模型
            self.models[config.name] = model_info
            
            # 保存配置
            self.save_models_config()
            
            print(f"模型注册成功: {config.name}")
            return True
            
        except Exception as e:
            print(f"模型注册失败: {e}")
            return False
    
    def unregister_model(self, model_name: str) -> bool:
        """
        注销模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 是否注销成功
        """
        if model_name not in self.models:
            return False
        
        # 如果是当前模型，清除当前模型
        if self.current_model == model_name:
            self.current_model = None
        
        # 删除模型
        del self.models[model_name]
        
        # 保存配置
        self.save_models_config()
        
        print(f"模型注销成功: {model_name}")
        return True
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """
        获取模型信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            ModelInfo: 模型信息，如果不存在返回None
        """
        return self.models.get(model_name)
    
    def list_models(self) -> List[ModelInfo]:
        """
        列出所有模型
        
        Returns:
            List[ModelInfo]: 模型信息列表
        """
        return list(self.models.values())
    
    def set_current_model(self, model_name: str) -> bool:
        """
        设置当前模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 是否设置成功
        """
        if model_name not in self.models:
            return False
        
        # 检查模型文件是否存在
        model_info = self.models[model_name]
        if not os.path.exists(model_info.gpt_path) or not os.path.exists(model_info.sovits_path):
            print(f"模型文件不存在: {model_name}")
            return False
        
        # 清除之前模型的加载状态
        if self.current_model:
            self.models[self.current_model].is_loaded = False
        
        # 设置当前模型
        self.current_model = model_name
        self.models[model_name].is_loaded = True
        
        # 保存配置
        self.save_models_config()
        
        print(f"当前模型设置为: {model_name}")
        return True
    
    def get_current_model(self) -> Optional[ModelInfo]:
        """
        获取当前模型信息
        
        Returns:
            ModelInfo: 当前模型信息，如果没有返回None
        """
        if self.current_model:
            return self.models.get(self.current_model)
        return None
    
    def _detect_model_version(self, sovits_path: str) -> str:
        """
        检测模型版本
        
        Args:
            sovits_path: SoVITS模型路径
            
        Returns:
            str: 模型版本
        """
        filename = os.path.basename(sovits_path).lower()
        
        if "v2proplus" in filename:
            return "v2ProPlus"
        elif "v2pro" in filename:
            return "v2Pro"
        elif "v4" in filename:
            return "v4"
        elif "v3" in filename:
            return "v3"
        elif "v2" in filename:
            return "v2"
        else:
            return "v1"
    
    def validate_model(self, model_name: str) -> Dict[str, any]:
        """
        验证模型完整性
        
        Args:
            model_name: 模型名称
            
        Returns:
            Dict: 验证结果
        """
        if model_name not in self.models:
            return {"valid": False, "error": "模型不存在"}
        
        model_info = self.models[model_name]
        issues = []
        
        # 检查GPT模型文件
        if not os.path.exists(model_info.gpt_path):
            issues.append(f"GPT模型文件不存在: {model_info.gpt_path}")
        else:
            try:
                # 尝试加载模型检查格式
                checkpoint = torch.load(model_info.gpt_path, map_location='cpu')
                if 'weight' not in checkpoint and 'model' not in checkpoint:
                    issues.append("GPT模型格式不正确")
            except Exception as e:
                issues.append(f"GPT模型加载失败: {e}")
        
        # 检查SoVITS模型文件
        if not os.path.exists(model_info.sovits_path):
            issues.append(f"SoVITS模型文件不存在: {model_info.sovits_path}")
        else:
            try:
                # 尝试加载模型检查格式
                checkpoint = torch.load(model_info.sovits_path, map_location='cpu')
                if 'weight' not in checkpoint and 'model' not in checkpoint:
                    issues.append("SoVITS模型格式不正确")
            except Exception as e:
                issues.append(f"SoVITS模型加载失败: {e}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "model_info": {
                "name": model_info.name,
                "version": model_info.version,
                "file_size": model_info.file_size,
                "created_time": model_info.created_time.isoformat() if model_info.created_time else None
            }
        }
    
    def search_models(self, search_dir: str) -> List[Dict[str, str]]:
        """
        搜索目录中的模型文件
        
        Args:
            search_dir: 搜索目录
            
        Returns:
            List[Dict]: 找到的模型文件信息
        """
        found_models = []
        
        if not os.path.exists(search_dir):
            return found_models
        
        # 搜索GPT模型文件
        gpt_files = []
        sovits_files = []
        
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_lower = file.lower()
                
                if file_lower.endswith('.ckpt') and ('gpt' in file_lower or 's1' in file_lower):
                    gpt_files.append(file_path)
                elif file_lower.endswith('.pth') and ('sovits' in file_lower or 's2' in file_lower):
                    sovits_files.append(file_path)
        
        # 尝试匹配GPT和SoVITS模型
        for gpt_file in gpt_files:
            gpt_name = os.path.basename(gpt_file)
            
            # 寻找对应的SoVITS模型
            for sovits_file in sovits_files:
                sovits_name = os.path.basename(sovits_file)
                
                # 简单的名称匹配逻辑
                if self._is_matching_pair(gpt_name, sovits_name):
                    found_models.append({
                        "suggested_name": self._generate_model_name(gpt_name, sovits_name),
                        "gpt_path": gpt_file,
                        "sovits_path": sovits_file,
                        "version": self._detect_model_version(sovits_file)
                    })
        
        return found_models
    
    def _is_matching_pair(self, gpt_name: str, sovits_name: str) -> bool:
        """判断GPT和SoVITS文件是否为匹配的一对"""
        # 移除文件扩展名
        gpt_base = os.path.splitext(gpt_name)[0].lower()
        sovits_base = os.path.splitext(sovits_name)[0].lower()
        
        # 移除常见的前缀后缀
        gpt_clean = gpt_base.replace('gpt', '').replace('s1', '').replace('-', '').replace('_', '')
        sovits_clean = sovits_base.replace('sovits', '').replace('s2g', '').replace('s2', '').replace('-', '').replace('_', '')
        
        # 检查是否有共同的标识符
        return gpt_clean == sovits_clean or gpt_clean in sovits_clean or sovits_clean in gpt_clean
    
    def _generate_model_name(self, gpt_name: str, sovits_name: str) -> str:
        """根据文件名生成模型名称"""
        # 提取共同的标识符
        gpt_base = os.path.splitext(gpt_name)[0]
        sovits_base = os.path.splitext(sovits_name)[0]
        
        # 寻找共同部分
        common_parts = []
        gpt_parts = gpt_base.replace('-', '_').split('_')
        sovits_parts = sovits_base.replace('-', '_').split('_')
        
        for part in gpt_parts:
            if part.lower() not in ['gpt', 's1', 'e'] and any(part.lower() in sp.lower() for sp in sovits_parts):
                common_parts.append(part)
        
        if common_parts:
            return '_'.join(common_parts)
        else:
            return f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def get_model_statistics(self) -> Dict[str, any]:
        """获取模型统计信息"""
        total_models = len(self.models)
        total_size = sum(model.file_size or 0 for model in self.models.values())
        
        version_counts = {}
        for model in self.models.values():
            version_counts[model.version] = version_counts.get(model.version, 0) + 1
        
        return {
            "total_models": total_models,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "version_distribution": version_counts,
            "current_model": self.current_model,
            "models_dir": self.models_dir
        }