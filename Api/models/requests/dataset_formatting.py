#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集格式化阶段的请求数据模型

包含文本处理、音频特征提取、语义编码等功能的请求模型
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from .base import BaseRequest


# ==================== 配置模型 ====================

class TextProcessingConfig(BaseModel):
    """文本处理配置"""
    language: str = Field(default="zh", description="文本语言")
    bert_model: str = Field(default="chinese-roberta-wwm-ext", description="BERT模型")
    batch_size: int = Field(default=32, ge=1, le=128, description="批处理大小")
    max_length: int = Field(default=512, ge=64, le=1024, description="最大文本长度")
    enable_normalization: bool = Field(default=True, description="启用文本标准化")
    enable_punctuation: bool = Field(default=True, description="启用标点符号处理")
    enable_case_conversion: bool = Field(default=False, description="启用大小写转换")
    custom_vocab: Optional[str] = Field(default=None, description="自定义词汇表路径")
    
    @validator('language')
    def validate_language(cls, v):
        allowed_languages = ['zh', 'en', 'ja', 'ko', 'multi']
        if v not in allowed_languages:
            raise ValueError(f'language must be one of {allowed_languages}')
        return v


class AudioFeaturesConfig(BaseModel):
    """音频特征配置"""
    version: str = Field(default="v2Pro", description="模型版本")
    device: str = Field(default="auto", description="计算设备")
    batch_size: int = Field(default=1, ge=1, le=16, description="批处理大小")
    n_processes: int = Field(default=4, ge=1, le=16, description="进程数")
    
    # 特征提取选项
    extract_cnhubert: bool = Field(default=True, description="提取CNHubert特征")
    extract_wav32k: bool = Field(default=True, description="提取32kHz音频")
    extract_speaker: bool = Field(default=True, description="提取说话人特征")
    extract_f0: bool = Field(default=False, description="提取基频特征")
    extract_energy: bool = Field(default=False, description="提取能量特征")
    
    # 音频处理参数
    sample_rate: int = Field(default=32000, description="目标采样率")
    hop_length: int = Field(default=320, description="跳跃长度")
    win_length: int = Field(default=1280, description="窗口长度")
    n_fft: int = Field(default=1280, description="FFT点数")
    
    @validator('version')
    def validate_version(cls, v):
        allowed_versions = ['v1', 'v2', 'v3', 'v4', 'v2Pro', 'v2ProPlus']
        if v not in allowed_versions:
            raise ValueError(f'version must be one of {allowed_versions}')
        return v


class SemanticEncodingConfig(BaseModel):
    """语义编码配置"""
    version: str = Field(default="v2Pro", description="模型版本")
    device: str = Field(default="auto", description="计算设备")
    batch_size: int = Field(default=1, ge=1, le=8, description="批处理大小")
    max_sec: int = Field(default=30, ge=5, le=60, description="最大音频长度(秒)")
    
    # 采样参数
    top_k: int = Field(default=20, ge=1, le=100, description="Top-K采样")
    top_p: float = Field(default=0.6, ge=0.1, le=1.0, description="Top-P采样")
    temperature: float = Field(default=1.0, ge=0.1, le=2.0, description="温度参数")
    
    # 编码选项
    use_repetition_penalty: bool = Field(default=True, description="使用重复惩罚")
    repetition_penalty: float = Field(default=1.35, ge=1.0, le=2.0, description="重复惩罚系数")
    seed: int = Field(default=-1, description="随机种子")
    
    @validator('version')
    def validate_version(cls, v):
        allowed_versions = ['v1', 'v2', 'v3', 'v4', 'v2Pro', 'v2ProPlus']
        if v not in allowed_versions:
            raise ValueError(f'version must be one of {allowed_versions}')
        return v


# ==================== 请求模型 ====================

class TextProcessingRequest(BaseRequest):
    """文本处理请求"""
    list_file: str = Field(description="标注文件路径")
    output_dir: str = Field(description="输出目录")
    config: TextProcessingConfig = Field(default_factory=TextProcessingConfig, description="处理配置")
    
    # 输出选项
    output_formats: List[str] = Field(default=["bert", "text"], description="输出格式")
    save_intermediate: bool = Field(default=False, description="保存中间结果")
    overwrite: bool = Field(default=False, description="覆盖已存在文件")
    
    # 质量控制
    min_text_length: int = Field(default=1, ge=1, description="最小文本长度")
    max_text_length: int = Field(default=200, ge=10, description="最大文本长度")
    filter_invalid: bool = Field(default=True, description="过滤无效文本")
    
    @validator('list_file')
    def validate_list_file(cls, v):
        if not v or not v.strip():
            raise ValueError('list_file cannot be empty')
        return v.strip()
    
    @validator('output_formats')
    def validate_output_formats(cls, v):
        allowed_formats = ['bert', 'text', 'tokens', 'embeddings']
        for fmt in v:
            if fmt not in allowed_formats:
                raise ValueError(f'Invalid output format: {fmt}')
        return v


class AudioFeaturesRequest(BaseRequest):
    """音频特征请求"""
    list_file: str = Field(description="标注文件路径")
    output_dir: str = Field(description="输出目录")
    config: AudioFeaturesConfig = Field(default_factory=AudioFeaturesConfig, description="特征配置")
    
    # 处理选项
    resume_from_checkpoint: bool = Field(default=True, description="从检查点恢复")
    checkpoint_interval: int = Field(default=100, ge=10, description="检查点间隔")
    overwrite: bool = Field(default=False, description="覆盖已存在文件")
    
    # 质量控制
    min_audio_length: float = Field(default=0.5, ge=0.1, description="最小音频长度(秒)")
    max_audio_length: float = Field(default=30.0, le=60.0, description="最大音频长度(秒)")
    filter_invalid: bool = Field(default=True, description="过滤无效音频")
    
    @validator('list_file')
    def validate_list_file(cls, v):
        if not v or not v.strip():
            raise ValueError('list_file cannot be empty')
        return v.strip()


class SemanticEncodingRequest(BaseRequest):
    """语义编码请求"""
    list_file: str = Field(description="标注文件路径")
    output_dir: str = Field(description="输出目录")
    config: SemanticEncodingConfig = Field(default_factory=SemanticEncodingConfig, description="编码配置")
    
    # 处理选项
    resume_from_checkpoint: bool = Field(default=True, description="从检查点恢复")
    checkpoint_interval: int = Field(default=50, ge=10, description="检查点间隔")
    overwrite: bool = Field(default=False, description="覆盖已存在文件")
    
    # 输出选项
    output_format: str = Field(default="npy", regex="^(npy|pt|json)$", description="输出格式")
    compress_output: bool = Field(default=True, description="压缩输出文件")
    
    @validator('list_file')
    def validate_list_file(cls, v):
        if not v or not v.strip():
            raise ValueError('list_file cannot be empty')
        return v.strip()


class DatasetValidationRequest(BaseRequest):
    """数据集验证请求"""
    dataset_dir: str = Field(description="数据集目录")
    validation_rules: List[str] = Field(default=["completeness", "consistency", "quality"], description="验证规则")
    output_report: str = Field(description="验证报告输出路径")
    
    # 验证选项
    check_audio_quality: bool = Field(default=True, description="检查音频质量")
    check_text_quality: bool = Field(default=True, description="检查文本质量")
    check_alignment: bool = Field(default=True, description="检查音频文本对齐")
    check_duplicates: bool = Field(default=True, description="检查重复数据")
    
    # 质量阈值
    min_audio_snr: float = Field(default=10.0, description="最小信噪比(dB)")
    max_silence_ratio: float = Field(default=0.3, description="最大静音比例")
    min_text_similarity: float = Field(default=0.8, description="最小文本相似度")
    
    @validator('validation_rules')
    def validate_rules(cls, v):
        allowed_rules = ['completeness', 'consistency', 'quality', 'alignment', 'duplicates']
        for rule in v:
            if rule not in allowed_rules:
                raise ValueError(f'Invalid validation rule: {rule}')
        return v


class BatchDatasetFormattingRequest(BaseRequest):
    """批量数据集格式化请求"""
    projects: List[Dict[str, Any]] = Field(description="项目列表")
    stages: List[str] = Field(default=["text", "audio", "semantic"], description="处理阶段")
    max_concurrent: int = Field(default=2, ge=1, le=5, description="最大并发数")
    global_config: Optional[Dict[str, Any]] = Field(default=None, description="全局配置")
    
    # 输出选项
    merge_outputs: bool = Field(default=False, description="合并输出")
    create_manifest: bool = Field(default=True, description="创建清单文件")
    validate_outputs: bool = Field(default=True, description="验证输出")
    
    @validator('projects')
    def validate_projects(cls, v):
        if not v:
            raise ValueError('projects cannot be empty')
        return v
    
    @validator('stages')
    def validate_stages(cls, v):
        allowed_stages = ['text', 'audio', 'semantic', 'validation']
        for stage in v:
            if stage not in allowed_stages:
                raise ValueError(f'Invalid stage: {stage}')
        return v
