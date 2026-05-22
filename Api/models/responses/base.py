#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础响应数据模型

定义所有API响应的基础类
"""

from typing import Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


class BaseResponse(BaseModel):
    """基础响应模型"""
    success: bool = Field(description="请求是否成功")
    message: str = Field(description="响应消息")
    request_id: Optional[str] = Field(default=None, description="请求ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间")
    processing_time: Optional[float] = Field(default=None, description="处理时间(秒)")
    server_info: Optional[dict] = Field(default=None, description="服务器信息")
    
    class Config:
        """Pydantic配置"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PaginatedResponse(BaseResponse):
    """分页响应模型"""
    total: int = Field(description="总记录数")
    page: int = Field(description="当前页码")
    page_size: int = Field(description="每页大小")
    total_pages: int = Field(description="总页数")
    has_next: bool = Field(description="是否有下一页")
    has_prev: bool = Field(description="是否有上一页")


class StatusResponse(BaseResponse):
    """状态响应模型"""
    status: str = Field(description="状态")
    progress: Optional[float] = Field(default=None, ge=0, le=100, description="进度百分比")
    details: Optional[dict] = Field(default=None, description="详细信息")
    estimated_completion: Optional[datetime] = Field(default=None, description="预计完成时间")


class ErrorResponse(BaseResponse):
    """错误响应模型"""
    error_code: str = Field(description="错误代码")
    error_type: str = Field(description="错误类型")
    error_details: Optional[dict] = Field(default=None, description="错误详情")
    traceback: Optional[str] = Field(default=None, description="错误堆栈")
    suggestions: Optional[list] = Field(default=None, description="解决建议")


class HealthResponse(BaseResponse):
    """健康检查响应模型"""
    status: str = Field(description="服务状态")
    version: str = Field(description="服务版本")
    uptime: float = Field(description="运行时间(秒)")
    system_info: Optional[dict] = Field(default=None, description="系统信息")
    dependencies: Optional[dict] = Field(default=None, description="依赖服务状态")