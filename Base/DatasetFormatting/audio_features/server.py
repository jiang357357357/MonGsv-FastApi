"""
GPT-SoVITS 音频特征提取 FastAPI 服务器

提供HTTP API接口用于音频特征提取
"""

import os
import asyncio
from pathlib import Path
from typing import Dict, Any, Literal

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .service import AudioFeaturesService, AudioFeaturesConfig, AudioFeaturesRequest, AudioFeaturesResponse


class AudioFeaturesConfigModel(BaseModel):
    """音频特征提取配置模型"""
    cnhubert_base_dir: str = Field(
        default="GPT_SoVITS/pretrained_models/chinese-hubert-base",
        description="CNHubert预训练模型目录"
    )
    sv_model_path: str = Field(
        default="GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt",
        description="说话人模型路径"
    )
    version: Literal["v1", "v2", "v2Pro", "v2ProPlus", "v3", "v4"] = Field(
        default="v2", 
        description="GPT-SoVITS版本"
    )
    is_half: bool = Field(default=True, description="是否使用半精度")
    device: str = Field(default="auto", description="计算设备")
    maxx: float = Field(default=0.95, ge=0.1, le=1.0, description="最大归一化值")
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="混合比例")
    max_audio_value: float = Field(default=2.2, ge=1.0, le=10.0, description="音频最大值过滤阈值")
    n_parts: int = Field(default=1, ge=1, le=16, description="并行处理数")
    save_wav32k: bool = Field(default=True, description="是否保存32kHz音频")
    save_cnhubert: bool = Field(default=True, description="是否保存CNHubert特征")
    save_speaker: bool = Field(default=None, description="是否保存说话人特征（None=自动判断）")


class AudioFeaturesRequestModel(BaseModel):
    """音频特征提取请求模型"""
    input_text_file: str = Field(..., description="输入标注文件路径")
    input_wav_dir: str = Field(..., description="音频目录路径")
    experiment_name: str = Field(..., description="实验名称")
    output_dir: str = Field(..., description="输出目录路径")
    config: AudioFeaturesConfigModel = Field(
        default_factory=AudioFeaturesConfigModel,
        description="处理配置"
    )


class AudioFeaturesResponseModel(BaseModel):
    """音频特征提取响应模型"""
    success: bool
    message: str
    output_files: Dict[str, str] = {}
    processed_count: int = 0
    failed_count: int = 0
    nan_failed_count: int = 0
    processing_time: float = 0.0
    details: Dict[str, Any] = {}


# 创建FastAPI应用
app = FastAPI(
    title="GPT-SoVITS 音频特征提取服务",
    description="提供音频特征提取功能，包括CNHubert SSL特征和说话人特征提取",
    version="1.0.0"
)

# 全局API实例
audio_api = AudioFeaturesService()

# 任务状态存储
task_status: Dict[str, Dict] = {}


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "GPT-SoVITS 音频特征提取服务",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "extract": "/extract",
            "extract_async": "/extract_async",
            "status": "/status/{task_id}",
            "analyze": "/analyze",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "audio_features",
        "version": "1.0.0"
    }


@app.post("/extract", response_model=AudioFeaturesResponseModel)
async def extract_features(request: AudioFeaturesRequestModel):
    """
    同步提取音频特征
    
    Args:
        request: 音频特征提取请求
        
    Returns:
        处理结果
    """
    try:
        # 转换请求格式
        config = AudioFeaturesConfig(
            cnhubert_base_dir=request.config.cnhubert_base_dir,
            sv_model_path=request.config.sv_model_path,
            version=request.config.version,
            is_half=request.config.is_half,
            device=request.config.device,
            maxx=request.config.maxx,
            alpha=request.config.alpha,
            max_audio_value=request.config.max_audio_value,
            n_parts=request.config.n_parts,
            save_wav32k=request.config.save_wav32k,
            save_cnhubert=request.config.save_cnhubert,
            save_speaker=request.config.save_speaker
        )
        
        api_request = AudioFeaturesRequest(
            input_text_file=request.input_text_file,
            input_wav_dir=request.input_wav_dir,
            experiment_name=request.experiment_name,
            output_dir=request.output_dir,
            config=config
        )
        
        # 执行处理
        result = await audio_api.extract_features(api_request)
        
        return AudioFeaturesResponseModel(
            success=result.success,
            message=result.message,
            output_files=result.output_files,
            processed_count=result.processed_count,
            failed_count=result.failed_count,
            nan_failed_count=result.nan_failed_count,
            processing_time=result.processing_time,
            details=result.details
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


async def background_extract_features(task_id: str, request: AudioFeaturesRequestModel):
    """后台提取音频特征任务"""
    try:
        task_status[task_id] = {
            "status": "processing",
            "message": "正在提取音频特征...",
            "progress": 0
        }
        
        # 转换请求格式
        config = AudioFeaturesConfig(
            cnhubert_base_dir=request.config.cnhubert_base_dir,
            sv_model_path=request.config.sv_model_path,
            version=request.config.version,
            is_half=request.config.is_half,
            device=request.config.device,
            maxx=request.config.maxx,
            alpha=request.config.alpha,
            max_audio_value=request.config.max_audio_value,
            n_parts=request.config.n_parts,
            save_wav32k=request.config.save_wav32k,
            save_cnhubert=request.config.save_cnhubert,
            save_speaker=request.config.save_speaker
        )
        
        api_request = AudioFeaturesRequest(
            input_text_file=request.input_text_file,
            input_wav_dir=request.input_wav_dir,
            experiment_name=request.experiment_name,
            output_dir=request.output_dir,
            config=config
        )
        
        # 执行处理
        result = await audio_api.extract_features(api_request)
        
        # 更新任务状态
        task_status[task_id] = {
            "status": "completed" if result.success else "failed",
            "message": result.message,
            "progress": 100,
            "result": {
                "success": result.success,
                "output_files": result.output_files,
                "processed_count": result.processed_count,
                "failed_count": result.failed_count,
                "nan_failed_count": result.nan_failed_count,
                "processing_time": result.processing_time,
                "details": result.details
            }
        }
        
    except Exception as e:
        task_status[task_id] = {
            "status": "failed",
            "message": f"处理失败: {str(e)}",
            "progress": 0,
            "error": str(e)
        }


@app.post("/extract_async")
async def extract_features_async(request: AudioFeaturesRequestModel, background_tasks: BackgroundTasks):
    """
    异步提取音频特征
    
    Args:
        request: 音频特征提取请求
        background_tasks: 后台任务管理器
        
    Returns:
        任务ID和状态
    """
    import uuid
    
    # 生成任务ID
    task_id = str(uuid.uuid4())
    
    # 添加后台任务
    background_tasks.add_task(background_extract_features, task_id, request)
    
    return {
        "task_id": task_id,
        "status": "accepted",
        "message": "任务已提交，正在后台处理"
    }


@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """
    获取任务状态
    
    Args:
        task_id: 任务ID
        
    Returns:
        任务状态信息
    """
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return task_status[task_id]


@app.delete("/status/{task_id}")
async def delete_task_status(task_id: str):
    """
    删除任务状态记录
    
    Args:
        task_id: 任务ID
        
    Returns:
        删除结果
    """
    if task_id in task_status:
        del task_status[task_id]
        return {"message": "任务状态已删除"}
    else:
        raise HTTPException(status_code=404, detail="任务不存在")


@app.post("/analyze")
async def analyze_dataset(
    input_text_file: str,
    input_wav_dir: str = None
):
    """
    分析数据集
    
    Args:
        input_text_file: 输入标注文件路径
        input_wav_dir: 音频目录路径
        
    Returns:
        数据集分析结果
    """
    try:
        from .utils import AudioFeaturesUtils
        
        analysis = AudioFeaturesUtils.analyze_dataset_from_list(input_text_file, input_wav_dir)
        
        if "error" in analysis:
            raise HTTPException(status_code=400, detail=analysis["error"])
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


@app.get("/config/suggest")
async def suggest_config(
    input_text_file: str,
    input_wav_dir: str = None,
    version: str = "v2",
    target_processing_time: float = None,
    available_memory_gb: float = None
):
    """
    建议处理配置
    
    Args:
        input_text_file: 输入文件路径
        input_wav_dir: 音频目录路径
        version: GPT-SoVITS版本
        target_processing_time: 目标处理时间（秒）
        available_memory_gb: 可用内存（GB）
        
    Returns:
        建议的配置
    """
    try:
        from .utils import AudioFeaturesUtils
        
        if not os.path.exists(input_text_file):
            raise HTTPException(status_code=404, detail="输入文件不存在")
        
        # 建议配置
        suggested_config = AudioFeaturesUtils.suggest_processing_config(
            input_text_file, input_wav_dir, target_processing_time, available_memory_gb, version
        )
        
        # 分析数据集
        analysis = AudioFeaturesUtils.analyze_dataset_from_list(input_text_file, input_wav_dir)
        
        # 估算处理时间
        time_estimate = AudioFeaturesUtils.estimate_processing_time(
            input_text_file, input_wav_dir, suggested_config
        )
        
        return {
            "suggested_config": {
                "cnhubert_base_dir": suggested_config.cnhubert_base_dir,
                "sv_model_path": suggested_config.sv_model_path,
                "version": suggested_config.version,
                "is_half": suggested_config.is_half,
                "device": suggested_config.device,
                "n_parts": suggested_config.n_parts,
                "save_speaker": suggested_config.save_speaker
            },
            "analysis": {
                "total_files": analysis.get("valid_files", 0),
                "total_duration": analysis.get("total_duration", 0),
                "loud_files_count": analysis.get("loud_files_count", 0),
                "estimated_processing_time": time_estimate.get("estimated_total_time", 0)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


@app.get("/models/info")
async def get_models_info(
    cnhubert_path: str = "GPT_SoVITS/pretrained_models/chinese-hubert-base",
    speaker_path: str = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
):
    """
    获取模型信息
    
    Args:
        cnhubert_path: CNHubert模型路径
        speaker_path: 说话人模型路径
        
    Returns:
        模型信息
    """
    try:
        from .utils import AudioFeaturesUtils
        
        model_info = AudioFeaturesUtils.get_model_info(cnhubert_path, speaker_path)
        
        return model_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取模型信息失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info"
    )