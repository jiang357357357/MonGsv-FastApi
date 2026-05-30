#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理阶段的数据模型

包含语音合成推理的请求和响应模型
"""

from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field
from .base import BaseRequest, BaseResponse


# ==================== 推理配置 ====================

class InferenceConfig(BaseModel):
    """推理配置"""
    top_k: int = Field(default=20, description="Top-K采样")
    top_p: float = Field(default=0.6, description="Top-P采样")
    temperature: float = Field(default=0.6, description="温度参数")
    how_to_cut: str = Field(default="凑四句一切", description="文本切分方式")
    speed: float = Field(default=1.0, description="语速调节")
    seed: int = Field(default=-1, description="随机种子")
    parallel_infer: bool = Field(default=True, description="并行推理")
    split_bucket: bool = Field(default=True, description="分桶处理")
    inference_mode: str = Field(default="normal", description="推理模式: normal 或 accelerated")


class ModelConfig(BaseModel):
    """模型配置"""
    gpt_model_path: Optional[str] = Field(default=None, description="GPT模型路径")
    sovits_model_path: Optional[str] = Field(default=None, description="SoVITS模型路径")
    model_version: str = Field(default="v2Pro", description="模型版本")
    device: str = Field(default="auto", description="推理设备")
    half_precision: bool = Field(default=True, description="半精度推理")


# ==================== 推理请求 ====================

class InferenceRequest(BaseRequest):
    """推理请求"""
    text: str = Field(description="要合成的文本")
    text_language: str = Field(default="zh", description="文本语言")
    ref_audio_path: Optional[str] = Field(default=None, description="参考音频路径")
    ref_audio_base64: Optional[str] = Field(default=None, description="参考音频Base64")
    prompt_text: Optional[str] = Field(default=None, description="参考文本")
    prompt_language: str = Field(default="zh", description="参考文本语言")
    config: InferenceConfig = Field(default_factory=InferenceConfig)
    model_config: Optional[ModelConfig] = Field(default=None, description="模型配置")
    output_format: str = Field(default="wav", description="输出格式")
    sample_rate: int = Field(default=32000, description="采样率")
    return_base64: bool = Field(default=False, description="返回Base64")
    save_path: Optional[str] = Field(default=None, description="保存路径")


class BatchInferenceRequest(BaseRequest):
    """批量推理请求"""
    texts: List[str] = Field(description="要合成的文本列表")
    text_language: str = Field(default="zh", description="文本语言")
    ref_audio_path: Optional[str] = Field(default=None, description="参考音频路径")
    ref_audio_base64: Optional[str] = Field(default=None, description="参考音频Base64")
    prompt_text: Optional[str] = Field(default=None, description="参考文本")
    prompt_language: str = Field(default="zh", description="参考文本语言")
    config: InferenceConfig = Field(default_factory=InferenceConfig)
    output_dir: str = Field(description="输出目录")
    filename_template: str = Field(default="output_{index}.wav", description="文件名模板")
    max_concurrent: int = Field(default=3, description="最大并发数")


# ==================== 推理响应 ====================

class InferenceResult(BaseModel):
    """单个推理结果"""
    text: str = Field(description="合成文本")
    audio_data: Optional[str] = Field(default=None, description="音频数据(Base64)")
    audio_path: Optional[str] = Field(default=None, description="音频文件路径")
    sample_rate: Optional[int] = Field(default=None, description="采样率")
    duration: Optional[float] = Field(default=None, description="音频时长")
    file_size: Optional[int] = Field(default=None, description="文件大小")


class InferenceResponse(BaseResponse):
    """推理响应"""
    audio_data: Optional[str] = None  # Base64编码
    audio_path: Optional[str] = None  # 文件路径
    sample_rate: Optional[int] = None
    duration: Optional[float] = None
    text_segments: Optional[List[str]] = None
    inference_time: Optional[float] = None
    model_info: Optional[Dict[str, str]] = None


class BatchInferenceResponse(BaseResponse):
    """批量推理响应"""
    results: List[InferenceResult] = []
    output_dir: Optional[str] = None
    total_count: int = 0
    success_count: int = 0
    failed_count: int = 0
    total_duration: Optional[float] = None
    average_inference_time: Optional[float] = None


# ==================== 模型管理 ====================

class ModelInfo(BaseModel):
    """模型信息"""
    model_id: str = Field(description="模型ID")
    model_name: str = Field(description="模型名称")
    model_type: str = Field(description="模型类型")  # gpt, sovits
    model_path: str = Field(description="模型路径")
    version: str = Field(description="模型版本")
    language: str = Field(description="支持语言")
    speaker: Optional[str] = Field(default=None, description="说话人")
    description: Optional[str] = Field(default=None, description="模型描述")
    created_at: Optional[str] = Field(default=None, description="创建时间")
    file_size: Optional[int] = Field(default=None, description="文件大小")
    is_loaded: bool = Field(default=False, description="是否已加载")


class ModelListResponse(BaseResponse):
    """模型列表响应"""
    models: List[ModelInfo] = []
    total: int = 0
    loaded_count: int = 0


class ModelLoadRequest(BaseRequest):
    """模型加载请求"""
    gpt_model_path: str = Field(description="GPT模型路径")
    sovits_model_path: str = Field(description="SoVITS模型路径")
    device: str = Field(default="auto", description="加载设备")
    half_precision: bool = Field(default=True, description="半精度加载")


class ModelLoadResponse(BaseResponse):
    """模型加载响应"""
    model_id: Optional[str] = None
    gpt_model_info: Optional[Dict] = None
    sovits_model_info: Optional[Dict] = None
    load_time: Optional[float] = None
    memory_usage: Optional[float] = None
