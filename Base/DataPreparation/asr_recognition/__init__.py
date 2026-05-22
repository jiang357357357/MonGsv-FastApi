#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS ASR 识别模块。
"""

from .service import ASRRecognitionService, ASRConfig, ASRRequest, ASRResponse
from .server import app

__version__ = "1.0.0"
__author__ = "MonGsv Team"

__all__ = [
    "ASRRecognitionService",
    "ASRConfig",
    "ASRRequest",
    "ASRResponse",
    "app",
]
