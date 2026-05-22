#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据准备阶段的请求数据模型

包含音频切分、ASR识别等功能的请求模型
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator
from .base import BaseRequest


# ==================== 配置模型 ====================

class AudioSliceConfig(BaseModel):
    """音频切分配置"""
    threshold: float = Field(default=-34.0, ge=-60.0, le=-10.0, description="静音阈值(dB)")
    min_length: int = Field(default=4000, ge=1000, le=30000, description="最短时长(ms)")
    min_interval: int = Field(default=300, ge=100, le=2000, description="最短间隔(ms)")
    max_sil_kept: int = Field(default=500, ge=0, le=2000, description="保留静音长度(ms)")
    hop_size: int = Field(default=10, ge=1, le=100, description="跳跃大小(ms)")
    max_silence_per_file: float = Field(default=1.0, ge=0.1, le=5.0, description="每文件最大静音时长(s)")


class ASRConfig(BaseModel):
    """ASR识别配置"""
    model_type: str = Field(default="funasr", description="ASR模型类型")
    language: str = Field(default="zh", description="识别语言")
    precision: str = Field(default="float32", regex="^(float16|float32|int8)$", description="模型精度")
    batch_size: int = Field(default=1, ge=1, le=32, description="批处理大小")
    enable_vad: bool = Field(default=True, description="启用语音活动检测")
    vad_threshold: float = Field(default=0.5, ge=0.1, le=0.9, description="VAD阈值")
    beam_size: int = Field(default=5, ge=1, le=20, description="束搜索大小")
    length_penalty: float = Field(default=1.0, ge=0.1, le=2.0, description="长度惩罚")
    
    @validator('model_type')
    def validate_model_type(cls, v):
        allowed_models = ['funasr', 'whisper', 'wav2vec2', 'conformer']
        if v not in allowed_models:
            raise ValueError(f'model_type must be one of {allowed_models}')
        return v
    
    @validator('language')
    def validate_language(cls, v):
        allowed_languages = ['zh', 'en', 'ja', 'ko', 'auto']
        if v not in allowed_languages:
            raise ValueError(f'language must be one of {allowed_languages}')
        return v


# ==================== 请求模型 ====================

class AudioSliceRequest(BaseRequest):
    """音频切分请求"""
    input_path: str = Field(description="输入音频路径或目录")
    output_dir: str = Field(description="输出目录")
    config: AudioSliceConfig = Field(default_factory=AudioSliceConfig, description="切分配置")
    recursive: bool = Field(default=True, description="是否递归处理子目录")
    file_patterns: List[str] = Field(default=["*.wav", "*.mp3", "*.flac"], description="文件匹配模式")
    overwrite: bool = Field(default=False, description="是否覆盖已存在文件")
    
    @validator('input_path')
    def validate_input_path(cls, v):
        if not v or not v.strip():
            raise ValueError('input_path cannot be empty')
        return v.strip()
    
    @validator('output_dir')
    def validate_output_dir(cls, v):
        if not v or not v.strip():
            raise ValueError('output_dir cannot be empty')
        return v.strip()


class ASRRequest(BaseRequest):
    """ASR识别请求"""
    input_path: str = Field(description="输入音频路径或目录")
    output_file: str = Field(description="输出标注文件路径")
    config: ASRConfig = Field(default_factory=ASRConfig, description="识别配置")
    recursive: bool = Field(default=True, description="是否递归处理子目录")
    file_patterns: List[str] = Field(default=["*.wav", "*.mp3", "*.flac"], description="文件匹配模式")
    output_format: str = Field(default="list", regex="^(list|json|csv|srt)$", description="输出格式")
    include_confidence: bool = Field(default=True, description="是否包含置信度")
    include_timestamps: bool = Field(default=False, description="是否包含时间戳")
    
    @validator('input_path')
    def validate_input_path(cls, v):
        if not v or not v.strip():
            raise ValueError('input_path cannot be empty')
        return v.strip()
    
    @validator('output_file')
    def validate_output_file(cls, v):
        if not v or not v.strip():
            raise ValueError('output_file cannot be empty')
        return v.strip()


class VoiceSeparationRequest(BaseRequest):
    """人声分离请求"""
    input_path: str = Field(description="输入音频路径或目录")
    output_dir: str = Field(description="输出目录")
    model_name: str = Field(default="UVR-MDX-NET-Inst_HQ_3", description="分离模型名称")
    device: str = Field(default="auto", description="计算设备")
    output_format: str = Field(default="wav", regex="^(wav|flac|mp3)$", description="输出格式")
    normalize: bool = Field(default=True, description="是否标准化音量")
    denoise: bool = Field(default=False, description="是否降噪")
    
    @validator('input_path')
    def validate_input_path(cls, v):
        if not v or not v.strip():
            raise ValueError('input_path cannot be empty')
        return v.strip()


class TextAnnotationRequest(BaseRequest):
    """文本校对标注请求"""
    list_file: str = Field(description="ASR识别结果文件")
    output_file: str = Field(description="校对后输出文件")
    correction_mode: str = Field(default="manual", regex="^(manual|auto|hybrid)$", description="校对模式")
    language: str = Field(default="zh", description="文本语言")
    enable_spell_check: bool = Field(default=True, description="启用拼写检查")
    enable_grammar_check: bool = Field(default=True, description="启用语法检查")
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="置信度阈值")
    
    @validator('list_file')
    def validate_list_file(cls, v):
        if not v or not v.strip():
            raise ValueError('list_file cannot be empty')
        return v.strip()


class BatchDataPrepRequest(BaseRequest):
    """批量数据准备请求"""
    projects: List[dict] = Field(description="项目列表")
    stages: List[str] = Field(default=["slice", "asr"], description="处理阶段")
    max_concurrent: int = Field(default=3, ge=1, le=10, description="最大并发数")
    global_config: Optional[dict] = Field(default=None, description="全局配置")
    notification_config: Optional[dict] = Field(default=None, description="通知配置")
    
    @validator('projects')
    def validate_projects(cls, v):
        if not v:
            raise ValueError('projects cannot be empty')
        return v
    
    @validator('stages')
    def validate_stages(cls, v):
        allowed_stages = ['slice', 'asr', 'separation', 'annotation']
        for stage in v:
            if stage not in allowed_stages:
                raise ValueError(f'Invalid stage: {stage}. Allowed: {allowed_stages}')
        return v