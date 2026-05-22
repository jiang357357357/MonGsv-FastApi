#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置数据模型

包含所有功能模块的配置模型定义
"""

from .data_preparation import AudioSliceConfig, ASRConfig, VoiceSeparationConfig
from .dataset_formatting import TextProcessingConfig, AudioFeaturesConfig, SemanticEncodingConfig
from .training import GPTTrainingConfig, SoVITSTrainingConfig, TrainingMetrics
from .inference import InferenceConfig, ModelConfig
from .workflow import WorkflowConfig, StageConfig, WorkflowStages

__all__ = [
    # 数据准备配置
    "AudioSliceConfig",
    "ASRConfig", 
    "VoiceSeparationConfig",
    
    # 数据格式化配置
    "TextProcessingConfig",
    "AudioFeaturesConfig",
    "SemanticEncodingConfig",
    
    # 训练配置
    "GPTTrainingConfig",
    "SoVITSTrainingConfig",
    "TrainingMetrics",
    
    # 推理配置
    "InferenceConfig",
    "ModelConfig",
    
    # 工作流配置
    "WorkflowConfig",
    "StageConfig",
    "WorkflowStages",
]