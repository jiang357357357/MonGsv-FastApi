#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工作流相关的数据模型

包含完整工作流、批量处理等功能的请求和响应模型
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from .base import BaseRequest, BaseResponse


# ==================== 工作流配置 ====================

class WorkflowConfig(BaseModel):
    """工作流配置"""
    language: str = Field(default="zh", description="语言")
    version: str = Field(default="v2Pro", description="模型版本")
    skip_existing: bool = Field(default=True, description="跳过已存在文件")
    parallel_processing: bool = Field(default=True, description="并行处理")
    auto_cleanup: bool = Field(default=False, description="自动清理临时文件")
    enable_validation: bool = Field(default=True, description="启用数据验证")
    max_workers: int = Field(default=4, description="最大工作进程数")


class StageConfig(BaseModel):
    """阶段配置"""
    enabled: bool = Field(default=True, description="是否启用此阶段")
    config: Optional[Dict[str, Any]] = Field(default=None, description="阶段特定配置")
    retry_count: int = Field(default=3, description="重试次数")
    timeout: Optional[int] = Field(default=None, description="超时时间")


class WorkflowStages(BaseModel):
    """工作流阶段配置"""
    audio_slice: StageConfig = Field(default_factory=StageConfig)
    asr_recognition: StageConfig = Field(default_factory=StageConfig)
    text_processing: StageConfig = Field(default_factory=StageConfig)
    audio_features: StageConfig = Field(default_factory=StageConfig)
    semantic_encoding: StageConfig = Field(default_factory=StageConfig)
    gpt_training: StageConfig = Field(default_factory=lambda: StageConfig(enabled=False))
    sovits_training: StageConfig = Field(default_factory=lambda: StageConfig(enabled=False))


# ==================== 工作流请求 ====================

class WorkflowRequest(BaseRequest):
    """工作流请求"""
    project_name: str = Field(description="项目名称")
    input_audio_dir: str = Field(description="输入音频目录")
    output_dir: str = Field(description="输出目录")
    config: WorkflowConfig = Field(default_factory=WorkflowConfig)
    stages: WorkflowStages = Field(default_factory=WorkflowStages)
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="项目元数据")


class CustomWorkflowRequest(BaseRequest):
    """自定义工作流请求"""
    workflow_name: str = Field(description="工作流名称")
    steps: List[Dict[str, Any]] = Field(description="工作流步骤")
    input_data: Dict[str, Any] = Field(description="输入数据")
    config: Optional[Dict[str, Any]] = Field(default=None, description="全局配置")


# ==================== 工作流步骤 ====================

class WorkflowStep(BaseModel):
    """工作流步骤"""
    step_id: str = Field(description="步骤ID")
    step_name: str = Field(description="步骤名称")
    step_type: str = Field(description="步骤类型")
    status: str = Field(description="步骤状态")  # pending, running, completed, failed, skipped
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = Field(default=0, description="重试次数")
    dependencies: List[str] = Field(default=[], description="依赖步骤")


class WorkflowExecution(BaseModel):
    """工作流执行状态"""
    workflow_id: str = Field(description="工作流ID")
    project_name: str = Field(description="项目名称")
    status: str = Field(description="整体状态")  # pending, running, completed, failed, cancelled
    steps: List[WorkflowStep] = Field(default=[], description="步骤列表")
    current_step: Optional[str] = None
    overall_progress: Optional[float] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_duration: Optional[float] = None
    error_message: Optional[str] = None
    output_summary: Optional[Dict[str, Any]] = None


# ==================== 工作流响应 ====================

class WorkflowResponse(BaseResponse):
    """工作流响应"""
    project_name: Optional[str] = None
    workflow_id: Optional[str] = None
    steps: List[WorkflowStep] = []
    current_step: Optional[str] = None
    overall_progress: Optional[float] = None
    estimated_completion: Optional[datetime] = None
    output_summary: Optional[Dict[str, Any]] = None


class WorkflowStatusResponse(BaseResponse):
    """工作流状态响应"""
    execution: WorkflowExecution


class WorkflowListResponse(BaseResponse):
    """工作流列表响应"""
    workflows: List[WorkflowExecution] = []
    total: int = 0
    running_count: int = 0
    completed_count: int = 0
    failed_count: int = 0


# ==================== 批量处理 ====================

class BatchProject(BaseModel):
    """批量项目"""
    name: str = Field(description="项目名称")
    input_dir: str = Field(description="输入目录")
    output_dir: str = Field(description="输出目录")
    language: str = Field(default="zh", description="语言")
    version: str = Field(default="v2Pro", description="版本")
    priority: int = Field(default=0, description="优先级")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="项目元数据")


class BatchRequest(BaseRequest):
    """批量请求"""
    batch_name: str = Field(description="批次名称")
    projects: List[BatchProject] = Field(description="项目列表")
    max_concurrent: int = Field(default=3, description="最大并发数")
    config: WorkflowConfig = Field(default_factory=WorkflowConfig)
    notification_config: Optional[Dict[str, Any]] = Field(default=None, description="通知配置")


class BatchProjectResult(BaseModel):
    """批量项目结果"""
    project_name: str
    status: str  # completed, failed, cancelled
    workflow_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    error_message: Optional[str] = None
    output_summary: Optional[Dict[str, Any]] = None


class BatchResponse(BaseResponse):
    """批量响应"""
    batch_id: Optional[str] = None
    batch_name: Optional[str] = None
    total_projects: Optional[int] = None
    completed_projects: Optional[int] = None
    failed_projects: Optional[int] = None
    cancelled_projects: Optional[int] = None
    project_results: List[BatchProjectResult] = []
    overall_progress: Optional[float] = None
    estimated_completion: Optional[datetime] = None


# ==================== 模板和预设 ====================

class WorkflowTemplate(BaseModel):
    """工作流模板"""
    template_id: str = Field(description="模板ID")
    template_name: str = Field(description="模板名称")
    description: str = Field(description="模板描述")
    category: str = Field(description="模板分类")
    stages: WorkflowStages = Field(description="阶段配置")
    config: WorkflowConfig = Field(description="默认配置")
    created_at: datetime = Field(description="创建时间")
    updated_at: datetime = Field(description="更新时间")
    usage_count: int = Field(default=0, description="使用次数")


class WorkflowTemplateListResponse(BaseResponse):
    """工作流模板列表响应"""
    templates: List[WorkflowTemplate] = []
    categories: List[str] = []
    total: int = 0


class CreateWorkflowFromTemplateRequest(BaseRequest):
    """从模板创建工作流请求"""
    template_id: str = Field(description="模板ID")
    project_name: str = Field(description="项目名称")
    input_audio_dir: str = Field(description="输入音频目录")
    output_dir: str = Field(description="输出目录")
    config_overrides: Optional[Dict[str, Any]] = Field(default=None, description="配置覆盖")
    stage_overrides: Optional[Dict[str, Any]] = Field(default=None, description="阶段配置覆盖")