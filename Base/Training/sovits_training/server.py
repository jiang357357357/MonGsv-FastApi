#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS SoVITS训练 FastAPI 服务器

提供HTTP API接口服务
"""

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse

from .service import SoVITSTrainingService, SoVITSTrainingRequest, SoVITSTrainingResponse, SoVITSTrainingConfig, SoVITSTrainingStatus


# 创建FastAPI应用
app = FastAPI(
    title="GPT-SoVITS SoVITS训练 API",
    description="基于GPT-SoVITS的SoVITS模型训练服务",
    version="1.0.0"
)

# 初始化API实例
try:
    training_api = SoVITSTrainingService()
except Exception as e:
    print(f"初始化失败: {e}")
    training_api = None


@app.get("/")
async def root():
    """根路径"""
    return {"message": "GPT-SoVITS SoVITS训练 API", "status": "running"}


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy" if training_api else "error",
        "gpt_sovits_root": training_api.gpt_sovits_root if training_api else None,
        "s2_train_script_exists": os.path.exists(training_api.s2_train_script) if training_api else False,
        "supported_versions": training_api.get_supported_versions() if training_api else []
    }


@app.post("/train/start", response_model=SoVITSTrainingResponse)
async def start_training_endpoint(request: SoVITSTrainingRequest):
    """
    开始SoVITS训练
    
    Args:
        request: 训练请求参数
        
    Returns:
        SoVITSTrainingResponse: 训练响应
    """
    if not training_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    return await training_api.start_training(request)


@app.get("/train/status/{job_id}", response_model=SoVITSTrainingStatus)
async def get_training_status_endpoint(job_id: str):
    """
    获取训练状态
    
    Args:
        job_id: 训练任务ID
        
    Returns:
        SoVITSTrainingStatus: 训练状态
    """
    if not training_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    status = training_api.get_training_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"训练任务不存在: {job_id}")
    
    return status


@app.post("/train/stop/{job_id}")
async def stop_training_endpoint(job_id: str):
    """
    停止训练
    
    Args:
        job_id: 训练任务ID
        
    Returns:
        操作结果
    """
    if not training_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    success = training_api.stop_training(job_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"无法停止训练任务: {job_id}")
    
    return {"message": f"训练任务 {job_id} 已停止", "success": True}


@app.get("/train/jobs", response_model=List[str])
async def list_training_jobs_endpoint():
    """
    列出所有训练任务
    
    Returns:
        List[str]: 任务ID列表
    """
    if not training_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    return training_api.list_training_jobs()


@app.post("/train/validate")
async def validate_config_endpoint(config: SoVITSTrainingConfig):
    """
    验证训练配置
    
    Args:
        config: 训练配置
        
    Returns:
        验证结果
    """
    if not training_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    return training_api.validate_config(config)


@app.get("/train/versions")
async def get_supported_versions_endpoint():
    """
    获取支持的模型版本
    
    Returns:
        支持的版本列表
    """
    if not training_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    return {
        "versions": training_api.get_supported_versions(),
        "default": "v2Pro"
    }


@app.get("/train/log/{job_id}")
async def get_training_log_endpoint(job_id: str, lines: Optional[int] = 100):
    """
    获取训练日志
    
    Args:
        job_id: 训练任务ID
        lines: 返回的行数
        
    Returns:
        训练日志内容
    """
    if not training_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    if job_id not in training_api.training_jobs:
        raise HTTPException(status_code=404, detail=f"训练任务不存在: {job_id}")
    
    log_file = training_api.training_jobs[job_id]["log_file"]
    
    if not os.path.exists(log_file):
        return {"log": "日志文件不存在", "lines": 0}
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            if lines and lines > 0:
                log_lines = all_lines[-lines:]
            else:
                log_lines = all_lines
        
        return {
            "log": "".join(log_lines),
            "lines": len(log_lines),
            "total_lines": len(all_lines)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取日志失败: {str(e)}")


@app.get("/train/download/log/{job_id}")
async def download_training_log_endpoint(job_id: str):
    """
    下载训练日志文件
    
    Args:
        job_id: 训练任务ID
        
    Returns:
        日志文件下载
    """
    if not training_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    if job_id not in training_api.training_jobs:
        raise HTTPException(status_code=404, detail=f"训练任务不存在: {job_id}")
    
    log_file = training_api.training_jobs[job_id]["log_file"]
    
    if not os.path.exists(log_file):
        raise HTTPException(status_code=404, detail="日志文件不存在")
    
    return FileResponse(
        log_file,
        media_type='text/plain',
        filename=f"training_log_{job_id}.log"
    )


@app.get("/train/config/template/{version}")
async def get_config_template_endpoint(version: str):
    """
    获取配置模板
    
    Args:
        version: 模型版本
        
    Returns:
        配置模板
    """
    if not training_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    if version not in training_api.get_supported_versions():
        raise HTTPException(status_code=400, detail=f"不支持的版本: {version}")
    
    # 返回默认配置
    config = SoVITSTrainingConfig(version=version)
    return config.dict()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)