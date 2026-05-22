#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础请求数据模型

定义所有API请求的基础类
"""

from typing import Optional
from pydantic import BaseModel, Field


class BaseRequest(BaseModel):
    """基础请求模型"""
    request_id: Optional[str] = Field(default=None, description="请求ID，用于追踪请求")
    timeout: Optional[int] = Field(default=300, description="超时时间(秒)")
    priority: Optional[int] = Field(default=0, description="请求优先级，数值越大优先级越高")
    metadata: Optional[dict] = Field(default=None, description="请求元数据")
    
    class Config:
        """Pydantic配置"""
        extra = "forbid"  # 禁止额外字段
        validate_assignment = True  # 赋值时验证
        use_enum_values = True  # 使用枚举值


class PaginatedRequest(BaseRequest):
    """分页请求模型"""
    page: int = Field(default=1, ge=1, description="页码，从1开始")
    page_size: int = Field(default=20, ge=1, le=100, description="每页大小，最大100")
    sort_by: Optional[str] = Field(default=None, description="排序字段")
    sort_order: Optional[str] = Field(default="asc", regex="^(asc|desc)$", description="排序方向")


class FilterRequest(BaseRequest):
    """过滤请求模型"""
    filters: Optional[dict] = Field(default=None, description="过滤条件")
    search: Optional[str] = Field(default=None, description="搜索关键词")
    date_from: Optional[str] = Field(default=None, description="开始日期")
    date_to: Optional[str] = Field(default=None, description="结束日期")