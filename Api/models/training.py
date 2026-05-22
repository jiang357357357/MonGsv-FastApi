#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练阶段的数据模型

包含GPT训练、SoVITS训练等功能的请求和响应模型
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from datetime import datetime
from .base import BaseRequest, BaseResponse


# ==================== GPT训练 ====================

class GPTTrainingConfig(BaseModel):
    """GPT训练配置"""
    batch_size: int = Field(default=8, description="批次大小")
    total_epoch: int = Field(default=15, description="总轮数")
    learning_rate: float = Field(default=0.01, description="学习率")
    save_every_epoch: int = Field(default=5, description="保存间隔")
    gpu_numbers: str = Field(default="0", description="GPU设备")
    warmup_steps: int = Field(default=1000, description="预热步数")
    gradient_clip: float = Field(default=1.0, description="梯度裁剪")
    weight_decay: float = Field(default=0.01, description="权重衰减")


class GPTTrainingRequest(BaseRequest):
    """GPT训练请求"""
    exp_name: str = Field(description="实验名称")
    exp_root: str = Field(description="实验根目录")
    config: GPTTrainingConfig = Field(default_factory=GPTTrainingConfig)


class GPTTrainingResponse(BaseResponse):
    """GPT训练响应"""
    job_id: Optional[str] = None
    exp_name: Optional[str] = None
    status: Optional[str] = None
    log_file: Optional[str] = None
    checkpoint_dir: Optional[str] = None


# ==================== SoVITS训练 ====================

class SoVITSTrainingConfig(BaseModel):
    """SoVITS训练配置"""
    version: str = Field(default="v2Pro", description="模型版本")
    batch_size: int = Field(default=32, description="批次大小")
    total_epoch: int = Field(default=8, description="总轮数")
    learning_rate: float = Field(default=0.0001, description="学习率")
    save_every_epoch: int = Field(default=4, description="保存间隔")
    gpu_numbers: str = Field(default="0", description="GPU设备")
    fp16_run: bool = Field(default=True, description="使用FP16")
    segment_size: int = Field(default=8192, description="音频段长度")


class SoVITSTrainingRequest(BaseRequest):
    """SoVITS训练请求"""
    exp_name: str = Field(description="实验名称")
    exp_root: str = Field(description="实验根目录")
    config: SoVITSTrainingConfig = Field(default_factory=SoVITSTrainingConfig)


class SoVITSTrainingResponse(BaseResponse):
    """SoVITS训练响应"""
    job_id: Optional[str] = None
    exp_name: Optional[str] = None
    status: Optional[str] = None
    log_file: Optional[str] = None
    checkpoint_dir: Optional[str] = None


# ==================== 训练状态 ====================

class TrainingMetrics(BaseModel):
    """训练指标"""
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    grad_norm: Optional[float] = None
    memory_usage: Optional[float] = None
    samples_per_second: Optional[float] = None


class TrainingStatus(BaseModel):
    """训练状态"""
    job_id: str
    status: str  # running, completed, failed, stopped, paused
    progress: Optional[float] = None
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    metrics: Optional[TrainingMetrics] = None
    log_file: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    estimated_remaining: Optional[float] = None
    error_message: Optional[str] = None
    last_checkpoint: Optional[str] = None


# ==================== 训练管理 ====================

class TrainingJobInfo(BaseModel):
    """训练任务信息"""
    job_id: str
    exp_name: str
    training_type: str  # gpt, sovits
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    config: Optional[Dict] = None


class TrainingListResponse(BaseResponse):
    """训练任务列表响应"""
    jobs: List[TrainingJobInfo] = []
    total: int = 0
    running_count: int = 0
    completed_count: int = 0
    failed_count: int = 0