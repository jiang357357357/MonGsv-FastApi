#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据准备阶段的响应数据模型

包含音频切分、ASR识别等功能的响应模型
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from .base import BaseResponse


# ==================== 结果数据模型 ====================

class AudioSliceResult(BaseModel):
    """音频切分结果"""
    original_file: str = Field(description="原始文件路径")
    output_files: List[str] = Field(description="切分后的文件列表")
    segment_count: int = Field(description="切分段数")
    total_duration: float = Field(description="总时长(秒)")
    average_segment_duration: float = Field(description="平均段时长(秒)")
    silence_ratio: float = Field(description="静音比例")


class ASRResult(BaseModel):
    """ASR识别结果"""
    file_path: str = Field(description="音频文件路径")
    text: str = Field(description="识别文本")
    confidence: Optional[float] = Field(default=None, description="置信度")
    duration: Optional[float] = Field(default=None, description="音频时长(秒)")
    language: Optional[str] = Field(default=None, description="检测语言")
    word_count: Optional[int] = Field(default=None, description="词数")
    char_count: Optional[int] = Field(default=None, description="字符数")
    timestamps: Optional[List[dict]] = Field(default=None, description="时间戳信息")


class VoiceSeparationResult(BaseModel):
    """人声分离结果"""
    original_file: str = Field(description="原始文件路径")
    vocal_file: str = Field(description="人声文件路径")
    instrumental_file: str = Field(description="伴奏文件路径")
    separation_quality: Optional[float] = Field(default=None, description="分离质量评分")
    processing_time: float = Field(description="处理时间(秒)")


class TextAnnotationResult(BaseModel):
    """文本校对结果"""
    original_text: str = Field(description="原始文本")
    corrected_text: str = Field(description="校对后文本")
    corrections: List[dict] = Field(description="修正记录")
    confidence_score: float = Field(description="校对置信度")
    correction_count: int = Field(description="修正数量")


# ==================== 响应模型 ====================

class AudioSliceResponse(BaseResponse):
    """音频切分响应"""
    output_dir: Optional[str] = Field(default=None, description="输出目录")
    processed_files: List[str] = Field(default=[], description="已处理文件列表")
    results: List[AudioSliceResult] = Field(default=[], description="切分结果详情")
    error_files: List[str] = Field(default=[], description="处理失败的文件")
    statistics: Optional[Dict[str, Any]] = Field(default=None, description="统计信息")
    
    # 汇总统计
    total_files: int = Field(default=0, description="总文件数")
    success_files: int = Field(default=0, description="成功处理文件数")
    total_segments: int = Field(default=0, description="总切分段数")
    total_duration: float = Field(default=0.0, description="总音频时长(秒)")
    average_segments_per_file: float = Field(default=0.0, description="平均每文件切分段数")


class ASRResponse(BaseResponse):
    """ASR识别响应"""
    output_file: Optional[str] = Field(default=None, description="输出标注文件")
    processed_files: List[str] = Field(default=[], description="已处理文件列表")
    results: List[ASRResult] = Field(default=[], description="识别结果详情")
    error_files: List[str] = Field(default=[], description="处理失败的文件")
    statistics: Optional[Dict[str, Any]] = Field(default=None, description="统计信息")
    
    # 汇总统计
    total_files: int = Field(default=0, description="总文件数")
    success_files: int = Field(default=0, description="成功识别文件数")
    total_duration: float = Field(default=0.0, description="总音频时长(秒)")
    total_text_length: int = Field(default=0, description="总文本长度")
    average_confidence: float = Field(default=0.0, description="平均置信度")
    recognition_speed: float = Field(default=0.0, description="识别速度(倍速)")


class VoiceSeparationResponse(BaseResponse):
    """人声分离响应"""
    output_dir: Optional[str] = Field(default=None, description="输出目录")
    processed_files: List[str] = Field(default=[], description="已处理文件列表")
    results: List[VoiceSeparationResult] = Field(default=[], description="分离结果详情")
    error_files: List[str] = Field(default=[], description="处理失败的文件")
    
    # 汇总统计
    total_files: int = Field(default=0, description="总文件数")
    success_files: int = Field(default=0, description="成功分离文件数")
    total_processing_time: float = Field(default=0.0, description="总处理时间(秒)")
    average_quality_score: float = Field(default=0.0, description="平均分离质量")


class TextAnnotationResponse(BaseResponse):
    """文本校对响应"""
    output_file: Optional[str] = Field(default=None, description="校对后输出文件")
    results: List[TextAnnotationResult] = Field(default=[], description="校对结果详情")
    
    # 汇总统计
    total_entries: int = Field(default=0, description="总条目数")
    corrected_entries: int = Field(default=0, description="已校对条目数")
    total_corrections: int = Field(default=0, description="总修正数量")
    average_confidence: float = Field(default=0.0, description="平均校对置信度")
    correction_rate: float = Field(default=0.0, description="修正率")


class BatchDataPrepResponse(BaseResponse):
    """批量数据准备响应"""
    batch_id: Optional[str] = Field(default=None, description="批次ID")
    total_projects: int = Field(default=0, description="总项目数")
    completed_projects: int = Field(default=0, description="已完成项目数")
    failed_projects: int = Field(default=0, description="失败项目数")
    running_projects: int = Field(default=0, description="运行中项目数")
    
    project_results: List[Dict[str, Any]] = Field(default=[], description="项目结果列表")
    overall_progress: float = Field(default=0.0, description="整体进度")
    estimated_completion: Optional[datetime] = Field(default=None, description="预计完成时间")
    
    # 详细统计
    stage_statistics: Dict[str, Dict[str, Any]] = Field(default={}, description="各阶段统计")
    resource_usage: Optional[Dict[str, Any]] = Field(default=None, description="资源使用情况")


# ==================== 状态和监控响应 ====================

class DataPrepStatusResponse(BaseResponse):
    """数据准备状态响应"""
    stage: str = Field(description="当前阶段")
    status: str = Field(description="状态")
    progress: float = Field(description="进度百分比")
    current_file: Optional[str] = Field(default=None, description="当前处理文件")
    processed_count: int = Field(description="已处理数量")
    total_count: int = Field(description="总数量")
    estimated_remaining: Optional[float] = Field(default=None, description="预计剩余时间(秒)")
    error_count: int = Field(default=0, description="错误数量")
    warnings: List[str] = Field(default=[], description="警告信息")


class DataPrepMetricsResponse(BaseResponse):
    """数据准备指标响应"""
    processing_speed: float = Field(description="处理速度(文件/秒)")
    throughput: float = Field(description="吞吐量(MB/秒)")
    cpu_usage: float = Field(description="CPU使用率")
    memory_usage: float = Field(description="内存使用率")
    disk_usage: float = Field(description="磁盘使用率")
    gpu_usage: Optional[float] = Field(default=None, description="GPU使用率")
    
    # 质量指标
    average_audio_quality: Optional[float] = Field(default=None, description="平均音频质量")
    average_text_quality: Optional[float] = Field(default=None, description="平均文本质量")
    error_rate: float = Field(description="错误率")
    
    # 时间指标
    average_processing_time: float = Field(description="平均处理时间(秒)")
    queue_wait_time: float = Field(description="队列等待时间(秒)")
    
    timestamp: datetime = Field(default_factory=datetime.now, description="指标时间戳")