#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集格式化阶段的数据模型

包含文本处理、音频特征提取、语义编码等功能的请求和响应模型
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from .base import BaseRequest, BaseResponse


# ==================== 文本处理 ====================

class TextProcessingConfig(BaseModel):
    """文本处理配置"""
    language: str = Field(default="zh", description="文本语言")
    bert_model: str = Field(default="chinese-roberta-wwm-ext", description="BERT模型")
    batch_size: int = Field(default=32, description="批处理大小")
    max_length: int = Field(default=512, description="最大文本长度")
    enable_normalization: bool = Field(default=True, description="启用文本标准化")


class TextProcessingRequest(BaseRequest):
    """文本处理请求"""
    list_file: str = Field(description="标注文件路径")
    output_dir: str = Field(description="输出目录")
    config: TextProcessingConfig = Field(default_factory=TextProcessingConfig)


class TextProcessingResponse(BaseResponse):
    """文本处理响应"""
    output_dir: Optional[str] = None
    text_file: Optional[str] = None
    bert_dir: Optional[str] = None
    processed_count: Optional[int] = None
    average_text_length: Optional[float] = None
    vocabulary_size: Optional[int] = None


# ==================== 音频特征提取 ====================

class AudioFeaturesConfig(BaseModel):
    """音频特征配置"""
    version: str = Field(default="v2Pro", description="模型版本")
    device: str = Field(default="auto", description="计算设备")
    batch_size: int = Field(default=1, description="批处理大小")
    n_processes: int = Field(default=4, description="进程数")
    extract_cnhubert: bool = Field(default=True, description="提取CNHubert特征")
    extract_wav32k: bool = Field(default=True, description="提取32kHz音频")
    extract_speaker: bool = Field(default=True, description="提取说话人特征")


class AudioFeaturesRequest(BaseRequest):
    """音频特征请求"""
    list_file: str = Field(description="标注文件路径")
    output_dir: str = Field(description="输出目录")
    config: AudioFeaturesConfig = Field(default_factory=AudioFeaturesConfig)


class AudioFeaturesResponse(BaseResponse):
    """音频特征响应"""
    output_dir: Optional[str] = None
    cnhubert_dir: Optional[str] = None
    wav32k_dir: Optional[str] = None
    sv_dir: Optional[str] = None
    processed_count: Optional[int] = None
    total_audio_duration: Optional[float] = None
    feature_dimensions: Optional[Dict[str, int]] = None


# ==================== 语义编码 ====================

class SemanticEncodingConfig(BaseModel):
    """语义编码配置"""
    version: str = Field(default="v2Pro", description="模型版本")
    device: str = Field(default="auto", description="计算设备")
    batch_size: int = Field(default=1, description="批处理大小")
    max_sec: int = Field(default=30, description="最大音频长度(秒)")
    top_k: int = Field(default=20, description="Top-K采样")
    top_p: float = Field(default=0.6, description="Top-P采样")


class SemanticEncodingRequest(BaseRequest):
    """语义编码请求"""
    list_file: str = Field(description="标注文件路径")
    output_dir: str = Field(description="输出目录")
    config: SemanticEncodingConfig = Field(default_factory=SemanticEncodingConfig)


class SemanticEncodingResponse(BaseResponse):
    """语义编码响应"""
    output_dir: Optional[str] = None
    semantic_file: Optional[str] = None
    processed_count: Optional[int] = None
    total_tokens: Optional[int] = None
    average_tokens_per_second: Optional[float] = None