#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS API配置管理

提供API配置的加载、保存和管理功能
"""

import os
import json
import tempfile
from typing import Dict, Any, Optional


class APIConfig:
    """API配置管理类"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径，默认为 ~/.gpt_sovits_api.json
        """
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
        
        # 返回默认配置
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "base_url": "http://localhost:8000",
            "api_key": None,
            "timeout": 300,
            "max_retries": 3,
            "default_language": "zh",
            "default_version": "v2Pro",
            "temp_dir": tempfile.gettempdir(),
            "log_level": "INFO",
            "enable_ssl_verify": True,
            "connection_pool_size": 10,
            "request_headers": {}
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
    
    def update(self, config_dict: Dict[str, Any]):
        """批量更新配置"""
        self.config.update(config_dict)
        self.save_config()
    
    def reset_to_default(self):
        """重置为默认配置"""
        self.config = self._get_default_config()
        self.save_config()
    
    def get_client_config(self) -> Dict[str, Any]:
        """获取客户端配置"""
        return {
            "base_url": self.get("base_url"),
            "api_key": self.get("api_key"),
            "timeout": self.get("timeout"),
            "max_retries": self.get("max_retries")
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"APIConfig(file={self.config_file}, base_url={self.get('base_url')})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return f"APIConfig({self.config})"


# 全局默认配置实例
_default_config = None


def get_default_config() -> APIConfig:
    """获取全局默认配置实例"""
    global _default_config
    if _default_config is None:
        _default_config = APIConfig()
    return _default_config


def set_default_config(config: APIConfig):
    """设置全局默认配置实例"""
    global _default_config
    _default_config = config