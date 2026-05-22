#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS ASR语音识别 FastAPI 服务器

提供HTTP API接口服务
"""

import os
import tempfile
import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse

from .service import ASRRecognitionService, ASRRequest, ASRResponse, ASRConfig


# 创建FastAPI应用
app = FastAPI(
    title="GPT-SoVITS ASR语音识别 API",
    description="基于GPT-SoVITS的语音识别服务",
    version="1.0.0"
)

# 初始化API实例
try:
    asr_api = ASRRecognitionService()
except Exception as e:
    print(f"初始化失败: {e}")
    asr_api = None


@app.get("/")
async def root():
    """根路径"""
    return {"message": "GPT-SoVITS ASR语音识别 API", "status": "running"}


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy" if asr_api else "error",
        "gpt_sovits_root": asr_api.gpt_sovits_root if asr_api else None,
        "asr_tools_exists": os.path.exists(asr_api.asr_tools_dir) if asr_api else False,
        "supported_models": asr_api.get_supported_models() if asr_api else {}
    }


@app.get("/models")
async def get_supported_models():
    """获取支持的模型信息"""
    if not asr_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    return asr_api.get_supported_models()


@app.post("/suggest-config")
async def suggest_config(language: str = "zh"):
    """根据语言建议最佳配置"""
    if not asr_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    config = asr_api.suggest_config(language)
    return config


@app.post("/recognize", response_model=ASRResponse)
async def recognize_audio_endpoint(request: ASRRequest):
    """
    语音识别接口
    
    Args:
        request: 识别请求参数
        
    Returns:
        ASRResponse: 识别结果
    """
    if not asr_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    return await asr_api.recognize_audio(request)


@app.post("/recognize/upload")
async def recognize_upload_endpoint(
    file: UploadFile = File(...),
    model_type: str = Form(default="funasr"),
    model_size: str = Form(default="large"),
    language: str = Form(default="zh"),
    precision: str = Form(default="float32"),
    vad_filter: bool = Form(default=True),
    beam_size: int = Form(default=5)
):
    """
    上传文件并识别
    
    Args:
        file: 上传的音频文件
        其他参数: 识别配置参数
        
    Returns:
        识别结果
    """
    if not asr_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 保存上传文件
        input_path = os.path.join(temp_dir, file.filename)
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 创建输出目录
        output_dir = os.path.join(temp_dir, "output")
        
        # 构建请求
        config = ASRConfig(
            model_type=model_type,
            model_size=model_size,
            language=language,
            precision=precision,
            vad_filter=vad_filter,
            beam_size=beam_size
        )
        
        request = ASRRequest(
            input_path=input_path,
            output_dir=output_dir,
            config=config
        )
        
        # 执行识别
        result = await asr_api.recognize_audio(request)
        
        if result.success and result.output_file:
            # 返回识别结果文件
            return FileResponse(
                result.output_file,
                media_type='text/plain',
                filename=f"{Path(file.filename).stem}_recognition.list"
            )
        else:
            raise HTTPException(status_code=400, detail=result.message)


@app.post("/recognize/batch")
async def batch_recognize_endpoint(
    input_dir: str,
    output_dir: str,
    model_type: str = "funasr",
    model_size: str = "large", 
    language: str = "zh",
    precision: str = "float32"
):
    """
    批量识别接口
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        其他参数: 识别配置参数
        
    Returns:
        批量识别结果
    """
    if not asr_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    config = ASRConfig(
        model_type=model_type,
        model_size=model_size,
        language=language,
        precision=precision
    )
    
    results = await asr_api.batch_recognize(input_dir, output_dir, config)
    
    # 统计结果
    total_files = len(results)
    success_count = sum(1 for r in results if r.success)
    total_processing_time = sum(r.processing_time for r in results)
    
    return {
        "success": True,
        "message": f"批量识别完成",
        "total_files": total_files,
        "success_count": success_count,
        "failure_count": total_files - success_count,
        "total_processing_time": total_processing_time,
        "results": results
    }