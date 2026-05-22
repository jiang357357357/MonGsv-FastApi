#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API请求数据模型

包含所有API请求的数据模型定义
"""

from .base import BaseRequest
from .data_preparation import *
from .dataset_formatting import *
from .training import *
from .inference import *
from .workflow import *

__all__ = [
    # 基础请求模型
    "BaseRequest",
    
    # 数据准备请求
    "AudioSliceRequest",
    "ASRRequest",
    
    # 数据格式化请求
    "TextProcessingRequest",
    "AudioFeaturesRequest", 
    "SemanticEncodingRequest",
    
    # 训练请求
    "GPTTrainingRequest",
    "SoVITSTrainingRequest",
    
    # 推理请求
    "InferenceRequest",
    "BatchInferenceRequest",
    "ModelLoadRequest",
    
    # 工作流请求
    "WorkflowRequest",
    "CustomWorkflowRequest",
    "BatchRequest",
    "CreateWorkflowFromTemplateRequest",
]