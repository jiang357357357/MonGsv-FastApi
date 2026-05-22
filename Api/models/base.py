#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础数据模型

定义所有API模型的基础类
"""

from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime


class BaseRequest(BaseModel):
    """基础请求模型"""
    request_id: Optional[str] = Field(default=None, description="请求ID")
    timeout: Optional[int] = Field(default=300, description="超时时间(秒)")


class BaseResponse(BaseModel):
    """基础响应模型"""
    success: bool = Field(description="是否成功")
    message: str = Field(description="响应消息")
    request_id: Optional[str] = Field(default=None, description="请求ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间")
    processing_time: Optional[float] = Field(default=None, description="处理时间(秒)")


class PaginatedResponse(BaseResponse):
    """分页响应模型"""
    total: int = Field(description="总数量")
    page: int = Field(description="当前页码")
    page_size: int = Field(description="每页大小")
    total_pages: int = Field(description="总页数")


class StatusResponse(BaseResponse):
    """状态响应模型"""
    status: str = Field(description="状态")
    progress: Optional[float] = Field(default=None, description="进度百分比")
    details: Optional[dict] = Field(default=None, description="详细信息")