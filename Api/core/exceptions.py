#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS API异常定义

定义API调用过程中可能出现的各种异常
"""


class GPTSoVITSAPIError(Exception):
    """GPT-SoVITS API基础异常类"""
    
    def __init__(self, message: str, status_code: int = None, details: dict = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self):
        return f"GPTSoVITSAPIError: {self.message}"


class ServiceUnavailableError(GPTSoVITSAPIError):
    """服务不可用异常"""
    
    def __init__(self, service_name: str, message: str = None):
        self.service_name = service_name
        message = message or f"服务 {service_name} 不可用"
        super().__init__(message, status_code=503)


class ValidationError(GPTSoVITSAPIError):
    """参数验证异常"""
    
    def __init__(self, message: str, field: str = None):
        self.field = field
        super().__init__(message, status_code=400)


class ProcessingError(GPTSoVITSAPIError):
    """处理过程异常"""
    
    def __init__(self, message: str, stage: str = None):
        self.stage = stage
        super().__init__(message, status_code=500)


class TimeoutError(GPTSoVITSAPIError):
    """超时异常"""
    
    def __init__(self, message: str = "请求超时"):
        super().__init__(message, status_code=408)


class AuthenticationError(GPTSoVITSAPIError):
    """认证异常"""
    
    def __init__(self, message: str = "认证失败"):
        super().__init__(message, status_code=401)


class RateLimitError(GPTSoVITSAPIError):
    """频率限制异常"""
    
    def __init__(self, message: str = "请求频率过高"):
        super().__init__(message, status_code=429)