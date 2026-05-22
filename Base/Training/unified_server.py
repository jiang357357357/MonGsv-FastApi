#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 统一训练服务

提供GPT和SoVITS训练的统一API接口
"""

import os
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict

from Code.FastApi.Base.Training.sovits_training.service import SoVITSTrainingService, SoVITSTrainingRequest, SoVITSTrainingResponse
from Code.FastApi.Base.Training.gpt_training.service import GPTTrainingService, GPTTrainingRequest, GPTTrainingResponse


# 创建FastAPI应用
app = FastAPI(
    title="GPT-SoVITS 统一训练 API",
    description="提供GPT和SoVITS模型训练的统一服务接口",
    version="1.0.0"
)

# 初始化API实例
try:
    sovits_api = SoVITSTrainingService()
    gpt_api = GPTTrainingService()
    print("✓ 训练API初始化成功")
except Exception as e:
    print(f"✗ 训练API初始化失败: {e}")
    sovits_api = None
    gpt_api = None


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "GPT-SoVITS 统一训练 API",
        "status": "running",
        "services": {
            "sovits_training": "available" if sovits_api else "unavailable",
            "gpt_training": "available" if gpt_api else "unavailable"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy" if (sovits_api and gpt_api) else "partial",
        "services": {
            "sovits_training": {
                "status": "healthy" if sovits_api else "error",
                "gpt_sovits_root": sovits_api.gpt_sovits_root if sovits_api else None
            },
            "gpt_training": {
                "status": "healthy" if gpt_api else "error", 
                "gpt_sovits_root": gpt_api.gpt_sovits_root if gpt_api else None
            }
        }
    }


# GPT训练相关接口
@app.post("/gpt/train/start", response_model=GPTTrainingResponse)
async def start_gpt_training(request: GPTTrainingRequest):
    """开始GPT训练"""
    if not gpt_api:
        raise HTTPException(status_code=500, detail="GPT训练API未正确初始化")
    
    return await gpt_api.start_training(request)


@app.get("/gpt/train/status/{job_id}")
async def get_gpt_training_status(job_id: str):
    """获取GPT训练状态"""
    if not gpt_api:
        raise HTTPException(status_code=500, detail="GPT训练API未正确初始化")
    
    status = gpt_api.get_training_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"GPT训练任务不存在: {job_id}")
    
    return status


@app.post("/gpt/train/stop/{job_id}")
async def stop_gpt_training(job_id: str):
    """停止GPT训练"""
    if not gpt_api:
        raise HTTPException(status_code=500, detail="GPT训练API未正确初始化")
    
    success = gpt_api.stop_training(job_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"无法停止GPT训练任务: {job_id}")
    
    return {"message": f"GPT训练任务 {job_id} 已停止", "success": True}


# SoVITS训练相关接口
@app.post("/sovits/train/start", response_model=SoVITSTrainingResponse)
async def start_sovits_training(request: SoVITSTrainingRequest):
    """开始SoVITS训练"""
    if not sovits_api:
        raise HTTPException(status_code=500, detail="SoVITS训练API未正确初始化")
    
    return await sovits_api.start_training(request)


@app.get("/sovits/train/status/{job_id}")
async def get_sovits_training_status(job_id: str):
    """获取SoVITS训练状态"""
    if not sovits_api:
        raise HTTPException(status_code=500, detail="SoVITS训练API未正确初始化")
    
    status = sovits_api.get_training_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"SoVITS训练任务不存在: {job_id}")
    
    return status


@app.post("/sovits/train/stop/{job_id}")
async def stop_sovits_training(job_id: str):
    """停止SoVITS训练"""
    if not sovits_api:
        raise HTTPException(status_code=500, detail="SoVITS训练API未正确初始化")
    
    success = sovits_api.stop_training(job_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"无法停止SoVITS训练任务: {job_id}")
    
    return {"message": f"SoVITS训练任务 {job_id} 已停止", "success": True}


# 统一接口
@app.get("/train/status/{job_id}")
async def get_training_status(job_id: str):
    """获取任意训练任务状态"""
    # 先尝试GPT训练
    if gpt_api:
        gpt_status = gpt_api.get_training_status(job_id)
        if gpt_status:
            return {"type": "gpt", "status": gpt_status}
    
    # 再尝试SoVITS训练
    if sovits_api:
        sovits_status = sovits_api.get_training_status(job_id)
        if sovits_status:
            return {"type": "sovits", "status": sovits_status}
    
    raise HTTPException(status_code=404, detail=f"训练任务不存在: {job_id}")


@app.get("/train/jobs")
async def list_all_training_jobs():
    """列出所有训练任务"""
    jobs = {
        "gpt_jobs": gpt_api.list_training_jobs() if gpt_api else [],
        "sovits_jobs": sovits_api.list_training_jobs() if sovits_api else []
    }
    return jobs


@app.post("/train/stop/{job_id}")
async def stop_any_training(job_id: str):
    """停止任意训练任务"""
    # 先尝试停止GPT训练
    if gpt_api and job_id in gpt_api.list_training_jobs():
        success = gpt_api.stop_training(job_id)
        if success:
            return {"message": f"GPT训练任务 {job_id} 已停止", "type": "gpt", "success": True}
    
    # 再尝试停止SoVITS训练
    if sovits_api and job_id in sovits_api.list_training_jobs():
        success = sovits_api.stop_training(job_id)
        if success:
            return {"message": f"SoVITS训练任务 {job_id} 已停止", "type": "sovits", "success": True}
    
    raise HTTPException(status_code=404, detail=f"无法停止训练任务: {job_id}")


# 训练流水线
class TrainingPipelineRequest(BaseModel):
    """训练流水线请求"""
    exp_name: str
    exp_root: str
    gpt_config: Optional[Dict] = None
    sovits_config: Optional[Dict] = None
    auto_start_sovits: bool = True  # GPT训练完成后自动开始SoVITS训练


@app.post("/train/pipeline")
async def start_training_pipeline(request: TrainingPipelineRequest):
    """
    启动完整训练流水线
    
    1. 先训练GPT模型
    2. GPT训练完成后自动开始SoVITS训练
    """
    if not (gpt_api and sovits_api):
        raise HTTPException(status_code=500, detail="训练API未正确初始化")
    
    # 创建GPT训练请求
    from Code.FastApi.Base.Training.gpt_training.service import GPTTrainingConfig
    gpt_config = GPTTrainingConfig(**(request.gpt_config or {}))
    gpt_request = GPTTrainingRequest(
        exp_name=request.exp_name,
        exp_root=request.exp_root,
        config=gpt_config
    )
    
    # 启动GPT训练
    gpt_response = await gpt_api.start_training(gpt_request)
    
    if not gpt_response.success:
        return {
            "success": False,
            "message": f"GPT训练启动失败: {gpt_response.message}",
            "gpt_response": gpt_response
        }
    
    result = {
        "success": True,
        "message": "训练流水线已启动",
        "gpt_job_id": gpt_response.job_id,
        "gpt_response": gpt_response,
        "pipeline_status": "gpt_training"
    }
    
    # 如果启用自动SoVITS训练，创建SoVITS训练请求
    if request.auto_start_sovits:
        from Code.FastApi.Base.Training.sovits_training.service import SoVITSTrainingConfig
        sovits_config = SoVITSTrainingConfig(**(request.sovits_config or {}))
        sovits_request = SoVITSTrainingRequest(
            exp_name=request.exp_name,
            exp_root=request.exp_root,
            config=sovits_config
        )
        
        result["sovits_request"] = sovits_request
        result["message"] += "，SoVITS训练将在GPT训练完成后自动开始"
    
    return result


@app.get("/train/pipeline/status/{gpt_job_id}")
async def get_pipeline_status(gpt_job_id: str):
    """获取训练流水线状态"""
    if not (gpt_api and sovits_api):
        raise HTTPException(status_code=500, detail="训练API未正确初始化")
    
    # 获取GPT训练状态
    gpt_status = gpt_api.get_training_status(gpt_job_id)
    if not gpt_status:
        raise HTTPException(status_code=404, detail=f"GPT训练任务不存在: {gpt_job_id}")
    
    result = {
        "gpt_job_id": gpt_job_id,
        "gpt_status": gpt_status,
        "pipeline_stage": "gpt_training"
    }
    
    # 如果GPT训练完成，检查是否有对应的SoVITS训练
    if gpt_status.status == "completed":
        # 查找相关的SoVITS训练任务
        sovits_jobs = sovits_api.list_training_jobs()
        for sovits_job_id in sovits_jobs:
            sovits_status = sovits_api.get_training_status(sovits_job_id)
            if sovits_status and sovits_job_id.startswith("sovits_"):
                result["sovits_job_id"] = sovits_job_id
                result["sovits_status"] = sovits_status
                result["pipeline_stage"] = "sovits_training"
                break
        
        if "sovits_job_id" not in result:
            result["pipeline_stage"] = "completed"
    elif gpt_status.status == "failed":
        result["pipeline_stage"] = "failed"
    
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)