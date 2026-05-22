#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API响应数据模型

包含所有API响应的数据模型定义
"""

from .base import BaseResponse, PaginatedResponse, StatusResponse
from .data_preparation import *
from .dataset_formatting import *
from .training import *
from .inference import *
from .workflow import *

__all__ = [
    # 基础响应模型
    "BaseResponse",
    "PaginatedResponse", 
    "StatusResponse",
    
    # 数据准备响应
    "AudioSliceResponse",
    "ASRResponse",
    
    # 数据格式化响应
    "TextProcessingResponse",
    "AudioFeaturesResponse",
    "SemanticEncodingResponse",
    
    # 训练响应
    "GPTTrainingResponse",
    "SoVITSTrainingResponse",
    "TrainingListResponse",
    
    # 推理响应
    "InferenceResponse",
    "BatchInferenceResponse",
    "ModelListResponse",
    "ModelLoadResponse",
    
    # 工作流响应
    "WorkflowResponse",
    "WorkflowStatusResponse",
    "WorkflowListResponse",
    "BatchResponse",
    "WorkflowTemplateListResponse",
]