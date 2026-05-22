#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 训练模块

提供完整的GPT-SoVITS模型训练API接口，包括：
- SoVITS模型训练 (阶段2)
- GPT模型训练 (阶段1)
"""

from .sovits_training import SoVITSTrainingService, SoVITSTrainingRequest, SoVITSTrainingResponse, SoVITSTrainingConfig
from .gpt_training import GPTTrainingService, GPTTrainingRequest, GPTTrainingResponse, GPTTrainingConfig

__all__ = [
    # SoVITS训练
    "SoVITSTrainingService",
    "SoVITSTrainingRequest", 
    "SoVITSTrainingResponse",
    "SoVITSTrainingConfig",
    
    # GPT训练
    "GPTTrainingService",
    "GPTTrainingRequest", 
    "GPTTrainingResponse",
    "GPTTrainingConfig"
]