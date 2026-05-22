#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS API调用模块

提供对FastApi/Base各个功能模块的统一调用接口
"""

from .client import GPTSoVITSClient, SyncGPTSoVITSClient
from .models import *
from .exceptions import *
from .utils import (
    APIConfig, RequestBuilder, AudioUtils, FileUtils, 
    ProgressTracker, RetryHelper, ResponseValidator, ConfigValidator,
    get_default_config, get_request_builder, create_temp_audio_file,
    format_duration, format_file_size
)

__version__ = "1.0.0"
__author__ = "MonGsv Team"
__description__ = "GPT-SoVITS统一API客户端"

__all__ = [
    # 客户端类
    "GPTSoVITSClient",
    "SyncGPTSoVITSClient",
    
    # 请求/响应模型
    "BaseRequest", "BaseResponse",
    "AudioSliceRequest", "AudioSliceResponse", "AudioSliceConfig",
    "ASRRequest", "ASRResponse", "ASRConfig",
    "TextProcessingRequest", "TextProcessingResponse", "TextProcessingConfig",
    "AudioFeaturesRequest", "AudioFeaturesResponse", "AudioFeaturesConfig",
    "SemanticEncodingRequest", "SemanticEncodingResponse", "SemanticEncodingConfig",
    "GPTTrainingRequest", "GPTTrainingResponse", "GPTTrainingConfig",
    "SoVITSTrainingRequest", "SoVITSTrainingResponse", "SoVITSTrainingConfig",
    "TrainingStatus",
    "InferenceRequest", "InferenceResponse", "InferenceConfig",
    "WorkflowRequest", "WorkflowResponse", "WorkflowConfig", "WorkflowStep",
    "BatchRequest", "BatchResponse", "BatchProject",
    
    # 异常类
    "GPTSoVITSAPIError",
    "ServiceUnavailableError", 
    "ValidationError",
    "ProcessingError",
    "AuthenticationError",
    "RateLimitError",
    
    # 工具类
    "APIConfig",
    "RequestBuilder", 
    "AudioUtils",
    "FileUtils",
    "ProgressTracker",
    "RetryHelper",
    "ResponseValidator",
    "ConfigValidator",
    
    # 便捷函数
    "get_default_config",
    "get_request_builder",
    "create_temp_audio_file",
    "format_duration",
    "format_file_size",
]