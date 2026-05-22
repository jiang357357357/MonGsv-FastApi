#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 音频切分模块。
"""

from .service import AudioSliceService, SliceConfig, SliceRequest, SliceResponse
from .server import app

__version__ = "1.0.0"
__author__ = "MonGsv Team"

__all__ = [
    "AudioSliceService",
    "SliceConfig",
    "SliceRequest",
    "SliceResponse",
    "app",
]
