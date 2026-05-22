"""
GPT-SoVITS 音频特征提取模块。
"""

from .service import AudioFeaturesService, AudioFeaturesConfig, AudioFeaturesRequest, AudioFeaturesResponse
from .utils import AudioFeaturesUtils

__all__ = [
    "AudioFeaturesService",
    "AudioFeaturesConfig",
    "AudioFeaturesRequest",
    "AudioFeaturesResponse",
    "AudioFeaturesUtils",
]

__version__ = "1.0.0"
