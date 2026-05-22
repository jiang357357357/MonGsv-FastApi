#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 推理模块。
"""

from .service import InferenceService, InferenceRequest, InferenceResponse, InferenceConfig
from .model_manager import ModelManager, ModelInfo
from .audio_processor import AudioProcessor
from .residency import ModelResidencyManager, ResidencyConfig, ResidencyRecord

__all__ = [
    "InferenceService",
    "InferenceRequest",
    "InferenceResponse",
    "InferenceConfig",
    "ModelManager",
    "ModelInfo",
    "AudioProcessor",
    "ModelResidencyManager",
    "ResidencyConfig",
    "ResidencyRecord",
]
