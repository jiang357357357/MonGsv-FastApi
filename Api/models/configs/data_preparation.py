#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据准备阶段的配置模型
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator


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
        allowed_languages = ['zh', 'yue', 'en', 'ja', 'ko', 'auto']
        if v not in allowed_languages:
            raise ValueError(f'language must be one of {allowed_languages}')
        return v


class VoiceSeparationConfig(BaseModel):
    """人声分离配置"""
    model_name: str = Field(default="UVR-MDX-NET-Inst_HQ_3", description="分离模型名称")
    device: str = Field(default="auto", description="计算设备")
    output_format: str = Field(default="wav", regex="^(wav|flac|mp3)$", description="输出格式")
    normalize: bool = Field(default=True, description="是否标准化音量")
    denoise: bool = Field(default=False, description="是否降噪")
    segment_size: int = Field(default=256, ge=64, le=1024, description="分段大小")
    overlap: float = Field(default=0.25, ge=0.0, le=0.5, description="重叠比例")
    
    @validator('model_name')
    def validate_model_name(cls, v):
        allowed_models = [
            'UVR-MDX-NET-Inst_HQ_3',
            'UVR5_HP-2',
            'UVR5_HP-3',
            'UVR5_HP-5'
        ]
        if v not in allowed_models:
            raise ValueError(f'model_name must be one of {allowed_models}')
        return v
