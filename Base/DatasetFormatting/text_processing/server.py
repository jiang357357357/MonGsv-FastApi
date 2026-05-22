"""
GPT-SoVITS 文本处理 FastAPI 服务器

提供HTTP API接口用于文本特征提取
"""

import os
import asyncio
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .service import TextProcessingService, TextProcessingConfig, TextProcessingRequest, TextProcessingResponse


class TextProcessingConfigModel(BaseModel):
    """文本处理配置模型"""
    bert_pretrained_dir: str = Field(
        default="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        description="BERT预训练模型目录"
    )
    version: str = Field(default="v2", description="GPT-SoVITS版本")
    is_half: bool = Field(default=True, description="是否使用半精度")
    device: str = Field(default="auto", description="计算设备")
    n_parts: int = Field(default=1, ge=1, le=16, description="并行处理数")


class TextProcessingRequestModel(BaseModel):
    """文本处理请求模型"""
    input_text_file: str = Field(..., description="输入标注文件路径")
    input_wav_dir: str = Field(..., description="音频目录路径")
    experiment_name: str = Field(..., description="实验名称")
    output_dir: str = Field(..., description="输出目录路径")
    config: TextProcessingConfigModel = Field(
        default_factory=TextProcessingConfigModel,
        description="处理配置"
    )


class TextProcessingResponseModel(BaseModel):
    """文本处理响应模型"""
    success: bool
    message: str
    output_files: Dict[str, str] = {}
    processed_count: int = 0
    failed_count: int = 0
    processing_time: float = 0.0
    details: Dict[str, Any] = {}


# 创建FastAPI应用
app = FastAPI(
    title="GPT-SoVITS 文本处理服务",
    description="提供文本特征提取功能，包括音素转换和BERT特征提取",
    version="1.0.0"
)

# 全局API实例
text_api = TextProcessingService()

# 任务状态存储
task_status: Dict[str, Dict] = {}


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "GPT-SoVITS 文本处理服务",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "process": "/process",
            "process_async": "/process_async",
            "status": "/status/{task_id}",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "text_processing",
        "version": "1.0.0"
    }


@app.post("/process", response_model=TextProcessingResponseModel)
async def process_text(request: TextProcessingRequestModel):
    """
    同步处理文本
    
    Args:
        request: 文本处理请求
        
    Returns:
        处理结果
    """
    try:
        # 转换请求格式
        config = TextProcessingConfig(
            bert_pretrained_dir=request.config.bert_pretrained_dir,
            version=request.config.version,
            is_half=request.config.is_half,
            device=request.config.device,
            n_parts=request.config.n_parts
        )
        
        api_request = TextProcessingRequest(
            input_text_file=request.input_text_file,
            input_wav_dir=request.input_wav_dir,
            experiment_name=request.experiment_name,
            output_dir=request.output_dir,
            config=config
        )
        
        # 执行处理
        result = await text_api.process_text(api_request)
        
        return TextProcessingResponseModel(
            success=result.success,
            message=result.message,
            output_files=result.output_files,
            processed_count=result.processed_count,
            failed_count=result.failed_count,
            processing_time=result.processing_time,
            details=result.details
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


async def background_process_text(task_id: str, request: TextProcessingRequestModel):
    """后台处理文本任务"""
    try:
        task_status[task_id] = {
            "status": "processing",
            "message": "正在处理文本...",
            "progress": 0
        }
        
        # 转换请求格式
        config = TextProcessingConfig(
            bert_pretrained_dir=request.config.bert_pretrained_dir,
            version=request.config.version,
            is_half=request.config.is_half,
            device=request.config.device,
            n_parts=request.config.n_parts
        )
        
        api_request = TextProcessingRequest(
            input_text_file=request.input_text_file,
            input_wav_dir=request.input_wav_dir,
            experiment_name=request.experiment_name,
            output_dir=request.output_dir,
            config=config
        )
        
        # 执行处理
        result = await text_api.process_text(api_request)
        
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


@app.post("/process_async")
async def process_text_async(request: TextProcessingRequestModel, background_tasks: BackgroundTasks):
    """
    异步处理文本
    
    Args:
        request: 文本处理请求
        background_tasks: 后台任务管理器
        
    Returns:
        任务ID和状态
    """
    import uuid
    
    # 生成任务ID
    task_id = str(uuid.uuid4())
    
    # 添加后台任务
    background_tasks.add_task(background_process_text, task_id, request)
    
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


@app.get("/config/suggest")
async def suggest_config(
    input_text_file: str,
    has_chinese: bool = None
):
    """
    建议处理配置
    
    Args:
        input_text_file: 输入文件路径
        has_chinese: 是否包含中文（可选，自动检测）
        
    Returns:
        建议的配置
    """
    try:
        if not os.path.exists(input_text_file):
            raise HTTPException(status_code=404, detail="输入文件不存在")
        
        # 分析文件内容
        with open(input_text_file, "r", encoding="utf8") as f:
            content = f.read()
        
        lines = content.strip().split("\n")
        total_lines = len(lines)
        
        # 检测是否包含中文
        if has_chinese is None:
            has_chinese = any("zh" in line for line in lines)
        
        # 建议配置
        suggested_config = {
            "bert_pretrained_dir": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
            "version": "v2",
            "is_half": True,
            "device": "auto",
            "n_parts": min(4, max(1, total_lines // 100))  # 根据数据量调整并行数
        }
        
        return {
            "suggested_config": suggested_config,
            "analysis": {
                "total_lines": total_lines,
                "has_chinese": has_chinese,
                "estimated_processing_time": f"{total_lines * 0.1:.1f}秒"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )