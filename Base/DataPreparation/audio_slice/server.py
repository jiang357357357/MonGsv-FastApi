#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 音频切分 FastAPI 服务器

提供HTTP API接口服务
"""

import os
import tempfile
import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse

from .service import AudioSliceService, SliceRequest, SliceResponse, SliceConfig


# 创建FastAPI应用
app = FastAPI(
    title="GPT-SoVITS 音频切分 API",
    description="基于GPT-SoVITS的音频切分服务",
    version="1.0.0"
)

# 初始化API实例
try:
    slice_api = AudioSliceService()
except Exception as e:
    print(f"初始化失败: {e}")
    slice_api = None


@app.get("/")
async def root():
    """根路径"""
    return {"message": "GPT-SoVITS 音频切分 API", "status": "running"}


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy" if slice_api else "error",
        "gpt_sovits_root": slice_api.gpt_sovits_root if slice_api else None,
        "slice_script_exists": os.path.exists(slice_api.slice_script) if slice_api else False
    }


@app.post("/slice", response_model=SliceResponse)
async def slice_audio_endpoint(request: SliceRequest):
    """
    音频切分接口
    
    Args:
        request: 切分请求参数
        
    Returns:
        SliceResponse: 切分结果
    """
    if not slice_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    return await slice_api.slice_audio(request)


@app.post("/slice/upload")
async def slice_upload_endpoint(
    file: UploadFile = File(...),
    threshold: float = Form(default=-34.0),
    min_length: int = Form(default=4000),
    min_interval: int = Form(default=300),
    hop_size: int = Form(default=10),
    max_sil_kept: int = Form(default=500),
    max_volume: float = Form(default=0.9),
    alpha: float = Form(default=0.25)
):
    """
    上传文件并切分
    
    Args:
        file: 上传的音频文件
        其他参数: 切分配置参数
        
    Returns:
        切分结果和下载链接
    """
    if not slice_api:
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
        config = SliceConfig(
            threshold=threshold,
            min_length=min_length,
            min_interval=min_interval,
            hop_size=hop_size,
            max_sil_kept=max_sil_kept,
            max_volume=max_volume,
            alpha=alpha
        )
        
        request = SliceRequest(
            input_path=input_path,
            output_dir=output_dir,
            config=config
        )
        
        # 执行切分
        result = await slice_api.slice_audio(request)
        
        if result.success:
            # 打包输出文件
            import zipfile
            zip_path = os.path.join(temp_dir, f"{Path(file.filename).stem}_sliced.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for output_file in result.output_files:
                    zipf.write(output_file, os.path.basename(output_file))
            
            return FileResponse(
                zip_path,
                media_type='application/zip',
                filename=f"{Path(file.filename).stem}_sliced.zip"
            )
        else:
            raise HTTPException(status_code=400, detail=result.message)