"""
GPT-SoVITS 语义编码模块。
"""

from .service import (
    SemanticEncodingService,
    SemanticEncodingConfig,
    SemanticEncodingRequest,
    SemanticEncodingResponse,
)
from .utils import SemanticEncodingUtils

__all__ = [
    "SemanticEncodingService",
    "SemanticEncodingConfig",
    "SemanticEncodingRequest",
    "SemanticEncodingResponse",
    "SemanticEncodingUtils",
]

__version__ = "1.0.0"
