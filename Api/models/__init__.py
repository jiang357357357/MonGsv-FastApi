#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS API数据模型

包含所有API请求和响应的数据模型定义，按请求和响应分类组织
"""

# 导入基础模型
from .base import BaseRequest, BaseResponse

# 导入请求模型
from .requests import *

# 导入响应模型  
from .responses import *

# 导入配置模型
from .configs import *

__all__ = [
    # 基础模型
    "BaseRequest", "BaseResponse",
    
    # 请求模型 - 数据准备
    "AudioSliceRequest", "ASRRequest", "VoiceSeparationRequest", 
    "TextAnnotationRequest", "BatchDataPrepRequest",
    
    # 请求模型 - 数据格式化
    "TextProcessingRequest", "AudioFeaturesRequest", "SemanticEncodingRequest",
    "DatasetValidationRequest", "BatchDatasetFormattingRequest",
    
    # 请求模型 - 训练
    "GPTTrainingRequest", "SoVITSTrainingRequest", "TrainingManagementRequest",
    
    # 请求模型 - 推理
    "InferenceRequest", "BatchInferenceRequest", "ModelLoadRequest",
    
    # 请求模型 - 工作流
    "WorkflowRequest", "CustomWorkflowRequest", "BatchRequest",
    "CreateWorkflowFromTemplateRequest",
    
    # 响应模型 - 数据准备
    "AudioSliceResponse", "ASRResponse", "VoiceSeparationResponse",
    "TextAnnotationResponse", "BatchDataPrepResponse",
    
    # 响应模型 - 数据格式化
    "TextProcessingResponse", "AudioFeaturesResponse", "SemanticEncodingResponse",
    "DatasetValidationResponse", "BatchDatasetFormattingResponse",
    
    # 响应模型 - 训练
    "GPTTrainingResponse", "SoVITSTrainingResponse", "TrainingListResponse",
    
    # 响应模型 - 推理
    "InferenceResponse", "BatchInferenceResponse", "ModelListResponse", "ModelLoadResponse",
    
    # 响应模型 - 工作流
    "WorkflowResponse", "WorkflowStatusResponse", "WorkflowListResponse",
    "BatchResponse", "WorkflowTemplateListResponse",
    
    # 配置模型
    "AudioSliceConfig", "ASRConfig", "TextProcessingConfig",
    "AudioFeaturesConfig", "SemanticEncodingConfig", "GPTTrainingConfig",
    "SoVITSTrainingConfig", "InferenceConfig", "WorkflowConfig",
    
    # 状态和结果模型
    "TrainingStatus", "WorkflowStep", "TrainingMetrics",
]