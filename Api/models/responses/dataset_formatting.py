#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集格式化阶段的响应数据模型

包含文本处理、音频特征提取、语义编码等功能的响应模型
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from .base import BaseResponse


# ==================== 结果数据模型 ====================

class TextProcessingResult(BaseModel):
    """文本处理结果"""
    file_path: str = Field(description="文件路径")
    original_text: str = Field(description="原始文本")
    processed_text: str = Field(description="处理后文本")
    text_length: int = Field(description="文本长度")
    token_count: int = Field(description="词元数量")
    bert_feature_shape: Optional[List[int]] = Field(default=None, description="BERT特征形状")
    processing_time: float = Field(description="处理时间(秒)")
    quality_score: Optional[float] = Field(default=None, description="质量评分")


class AudioFeaturesResult(BaseModel):
    """音频特征提取结果"""
    file_path: str = Field(description="音频文件路径")
    duration: float = Field(description="音频时长(秒)")
    sample_rate: int = Field(description="采样率")
    
    # 特征文件路径
    cnhubert_file: Optional[str] = Field(default=None, description="CNHubert特征文件")
    wav32k_file: Optional[str] = Field(default=None, description="32kHz音频文件")
    speaker_file: Optional[str] = Field(default=None, description="说话人特征文件")
    f0_file: Optional[str] = Field(default=None, description="基频特征文件")
    energy_file: Optional[str] = Field(default=None, description="能量特征文件")
    
    # 特征统计
    feature_dimensions: Dict[str, int] = Field(description="特征维度")
    feature_statistics: Optional[Dict[str, Any]] = Field(default=None, description="特征统计")
    processing_time: float = Field(description="处理时间(秒)")
    quality_metrics: Optional[Dict[str, float]] = Field(default=None, description="质量指标")


class SemanticEncodingResult(BaseModel):
    """语义编码结果"""
    file_path: str = Field(description="音频文件路径")
    semantic_file: str = Field(description="语义特征文件路径")
    token_count: int = Field(description="语义词元数量")
    sequence_length: int = Field(description="序列长度")
    encoding_time: float = Field(description="编码时间(秒)")
    
    # 编码统计
    unique_tokens: int = Field(description="唯一词元数")
    repetition_rate: float = Field(description="重复率")
    entropy: Optional[float] = Field(default=None, description="信息熵")
    compression_ratio: Optional[float] = Field(default=None, description="压缩比")


class DatasetValidationResult(BaseModel):
    """数据集验证结果"""
    rule_name: str = Field(description="验证规则名称")
    status: str = Field(description="验证状态")  # passed, failed, warning
    score: float = Field(description="验证得分")
    details: Dict[str, Any] = Field(description="验证详情")
    issues: List[str] = Field(default=[], description="发现的问题")
    suggestions: List[str] = Field(default=[], description="改进建议")


# ==================== 响应模型 ====================

class TextProcessingResponse(BaseResponse):
    """文本处理响应"""
    output_dir: Optional[str] = Field(default=None, description="输出目录")
    text_file: Optional[str] = Field(default=None, description="文本文件路径")
    bert_dir: Optional[str] = Field(default=None, description="BERT特征目录")
    
    results: List[TextProcessingResult] = Field(default=[], description="处理结果详情")
    error_files: List[str] = Field(default=[], description="处理失败的文件")
    
    # 汇总统计
    total_files: int = Field(default=0, description="总文件数")
    success_files: int = Field(default=0, description="成功处理文件数")
    total_text_length: int = Field(default=0, description="总文本长度")
    total_tokens: int = Field(default=0, description="总词元数")
    average_text_length: float = Field(default=0.0, description="平均文本长度")
    vocabulary_size: int = Field(default=0, description="词汇表大小")
    
    # 质量统计
    average_quality_score: float = Field(default=0.0, description="平均质量评分")
    quality_distribution: Optional[Dict[str, int]] = Field(default=None, description="质量分布")
    processing_speed: float = Field(default=0.0, description="处理速度(文件/秒)")


class AudioFeaturesResponse(BaseResponse):
    """音频特征响应"""
    output_dir: Optional[str] = Field(default=None, description="输出目录")
    cnhubert_dir: Optional[str] = Field(default=None, description="CNHubert特征目录")
    wav32k_dir: Optional[str] = Field(default=None, description="32kHz音频目录")
    sv_dir: Optional[str] = Field(default=None, description="说话人特征目录")
    
    results: List[AudioFeaturesResult] = Field(default=[], description="特征提取结果详情")
    error_files: List[str] = Field(default=[], description="处理失败的文件")
    
    # 汇总统计
    total_files: int = Field(default=0, description="总文件数")
    success_files: int = Field(default=0, description="成功处理文件数")
    total_audio_duration: float = Field(default=0.0, description="总音频时长(秒)")
    total_processing_time: float = Field(default=0.0, description="总处理时间(秒)")
    
    # 特征统计
    feature_dimensions: Dict[str, int] = Field(default={}, description="特征维度统计")
    average_feature_quality: Dict[str, float] = Field(default={}, description="平均特征质量")
    processing_speed: float = Field(default=0.0, description="处理速度(秒音频/秒)")
    
    # 资源使用
    peak_memory_usage: Optional[float] = Field(default=None, description="峰值内存使用(GB)")
    gpu_utilization: Optional[float] = Field(default=None, description="GPU利用率")


class SemanticEncodingResponse(BaseResponse):
    """语义编码响应"""
    output_dir: Optional[str] = Field(default=None, description="输出目录")
    semantic_file: Optional[str] = Field(default=None, description="语义特征文件")
    
    results: List[SemanticEncodingResult] = Field(default=[], description="编码结果详情")
    error_files: List[str] = Field(default=[], description="处理失败的文件")
    
    # 汇总统计
    total_files: int = Field(default=0, description="总文件数")
    success_files: int = Field(default=0, description="成功编码文件数")
    total_tokens: int = Field(default=0, description="总语义词元数")
    total_sequence_length: int = Field(default=0, description="总序列长度")
    
    # 编码统计
    vocabulary_size: int = Field(default=0, description="语义词汇表大小")
    average_tokens_per_second: float = Field(default=0.0, description="平均每秒词元数")
    average_repetition_rate: float = Field(default=0.0, description="平均重复率")
    average_entropy: float = Field(default=0.0, description="平均信息熵")
    
    # 质量指标
    encoding_quality_score: float = Field(default=0.0, description="编码质量评分")
    consistency_score: float = Field(default=0.0, description="一致性评分")
    processing_speed: float = Field(default=0.0, description="处理速度(秒音频/秒)")


class DatasetValidationResponse(BaseResponse):
    """数据集验证响应"""
    output_report: Optional[str] = Field(default=None, description="验证报告文件")
    
    validation_results: List[DatasetValidationResult] = Field(default=[], description="验证结果")
    overall_score: float = Field(description="总体评分")
    overall_status: str = Field(description="总体状态")  # passed, failed, warning
    
    # 统计信息
    total_files: int = Field(description="总文件数")
    valid_files: int = Field(description="有效文件数")
    invalid_files: int = Field(description="无效文件数")
    warning_files: int = Field(description="警告文件数")
    
    # 质量指标
    audio_quality_metrics: Dict[str, float] = Field(default={}, description="音频质量指标")
    text_quality_metrics: Dict[str, float] = Field(default={}, description="文本质量指标")
    alignment_metrics: Dict[str, float] = Field(default={}, description="对齐质量指标")
    
    # 问题汇总
    critical_issues: List[str] = Field(default=[], description="严重问题")
    warnings: List[str] = Field(default=[], description="警告信息")
    recommendations: List[str] = Field(default=[], description="改进建议")


class BatchDatasetFormattingResponse(BaseResponse):
    """批量数据集格式化响应"""
    batch_id: Optional[str] = Field(default=None, description="批次ID")
    total_projects: int = Field(default=0, description="总项目数")
    completed_projects: int = Field(default=0, description="已完成项目数")
    failed_projects: int = Field(default=0, description="失败项目数")
    running_projects: int = Field(default=0, description="运行中项目数")
    
    project_results: List[Dict[str, Any]] = Field(default=[], description="项目结果列表")
    overall_progress: float = Field(default=0.0, description="整体进度")
    estimated_completion: Optional[datetime] = Field(default=None, description="预计完成时间")
    
    # 合并输出信息
    merged_output_dir: Optional[str] = Field(default=None, description="合并输出目录")
    manifest_file: Optional[str] = Field(default=None, description="清单文件路径")
    
    # 详细统计
    stage_statistics: Dict[str, Dict[str, Any]] = Field(default={}, description="各阶段统计")
    quality_summary: Dict[str, float] = Field(default={}, description="质量汇总")
    resource_usage: Optional[Dict[str, Any]] = Field(default=None, description="资源使用情况")


# ==================== 状态和监控响应 ====================

class DatasetFormattingStatusResponse(BaseResponse):
    """数据集格式化状态响应"""
    stage: str = Field(description="当前阶段")
    status: str = Field(description="状态")
    progress: float = Field(description="进度百分比")
    current_file: Optional[str] = Field(default=None, description="当前处理文件")
    processed_count: int = Field(description="已处理数量")
    total_count: int = Field(description="总数量")
    
    # 性能指标
    processing_speed: float = Field(description="处理速度")
    estimated_remaining: Optional[float] = Field(default=None, description="预计剩余时间(秒)")
    
    # 质量指标
    current_quality_score: Optional[float] = Field(default=None, description="当前质量评分")
    average_quality_score: Optional[float] = Field(default=None, description="平均质量评分")
    
    # 错误和警告
    error_count: int = Field(default=0, description="错误数量")
    warning_count: int = Field(default=0, description="警告数量")
    recent_errors: List[str] = Field(default=[], description="最近错误")
    recent_warnings: List[str] = Field(default=[], description="最近警告")


class DatasetFormattingMetricsResponse(BaseResponse):
    """数据集格式化指标响应"""
    # 性能指标
    processing_speed: float = Field(description="处理速度(文件/秒)")
    throughput: float = Field(description="吞吐量(MB/秒)")
    cpu_usage: float = Field(description="CPU使用率")
    memory_usage: float = Field(description="内存使用率")
    disk_usage: float = Field(description="磁盘使用率")
    gpu_usage: Optional[float] = Field(default=None, description="GPU使用率")
    
    # 质量指标
    text_quality_score: float = Field(description="文本质量评分")
    audio_quality_score: float = Field(description="音频质量评分")
    feature_quality_score: float = Field(description="特征质量评分")
    overall_quality_score: float = Field(description="总体质量评分")
    
    # 数据统计
    total_text_length: int = Field(description="总文本长度")
    total_audio_duration: float = Field(description="总音频时长(秒)")
    total_features_size: float = Field(description="总特征大小(MB)")
    vocabulary_size: int = Field(description="词汇表大小")
    
    # 错误统计
    error_rate: float = Field(description="错误率")
    warning_rate: float = Field(description="警告率")
    retry_rate: float = Field(description="重试率")
    
    timestamp: datetime = Field(default_factory=datetime.now, description="指标时间戳")