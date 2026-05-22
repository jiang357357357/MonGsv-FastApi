#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 语义编码 FastAPI 服务器

提供HTTP API接口服务
"""

import os
import tempfile
import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse

from .service import SemanticEncodingService, SemanticEncodingRequest, SemanticEncodingResponse, SemanticEncodingConfig
from .utils import SemanticEncodingUtils


# 创建FastAPI应用
app = FastAPI(
    title="GPT-SoVITS 语义编码 API",
    description="基于GPT-SoVITS的语义编码服务，提取语义Token序列",
    version="1.0.0"
)

# 初始化API实例
try:
    semantic_api = SemanticEncodingService()
except Exception as e:
    print(f"初始化失败: {e}")
    semantic_api = None


@app.get("/")
async def root():
    """根路径"""
    return {"message": "GPT-SoVITS 语义编码 API", "status": "running"}


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy" if semantic_api else "error",
        "supported_versions": SemanticEncodingUtils.get_supported_versions(),
        "cuda_available": os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"),
        "torch_available": True
    }


@app.get("/versions")
async def get_supported_versions():
    """获取支持的模型版本信息"""
    return SemanticEncodingUtils.get_supported_versions()


@app.post("/suggest-config")
async def suggest_config(
    input_text_file: str,
    cnhubert_dir: str,
    target_time: float = 300.0,
    memory_gb: float = 8.0
):
    """根据数据集建议最佳配置"""
    if not semantic_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    config = SemanticEncodingUtils.suggest_processing_config(
        input_text_file, cnhubert_dir, target_time, memory_gb
    )
    return config


@app.post("/analyze")
async def analyze_dataset(
    input_text_file: str,
    cnhubert_dir: str
):
    """分析数据集"""
    if not semantic_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    analysis = SemanticEncodingUtils.analyze_input_data(input_text_file, cnhubert_dir)
    
    if "error" in analysis:
        raise HTTPException(status_code=400, detail=analysis["error"])
    
    return analysis


@app.post("/estimate-time")
async def estimate_processing_time(
    input_text_file: str,
    cnhubert_dir: str,
    device: str = "auto",
    is_half: bool = True,
    n_parts: int = 1
):
    """估算处理时间"""
    if not semantic_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    config = SemanticEncodingConfig(
        device=device,
        is_half=is_half,
        n_parts=n_parts
    )
    
    estimate = SemanticEncodingUtils.estimate_processing_time(
        input_text_file, cnhubert_dir, config
    )
    
    if "error" in estimate:
        raise HTTPException(status_code=400, detail=estimate["error"])
    
    return estimate


@app.post("/validate")
async def validate_input_files(
    input_text_file: str,
    cnhubert_dir: str,
    check_models: bool = True
):
    """验证输入文件"""
    if not semantic_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    validation = SemanticEncodingUtils.validate_input_files(
        input_text_file, cnhubert_dir, check_models
    )
    
    return validation


@app.post("/encode", response_model=SemanticEncodingResponse)
async def encode_semantic_endpoint(request: SemanticEncodingRequest):
    """
    语义编码接口
    
    Args:
        request: 编码请求参数
        
    Returns:
        SemanticEncodingResponse: 编码结果
    """
    if not semantic_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    return await semantic_api.encode_semantic(request)


@app.post("/encode/upload")
async def encode_upload_endpoint(
    text_file: UploadFile = File(...),
    cnhubert_archive: UploadFile = File(...),
    experiment_name: str = Form(...),
    pretrained_s2G: str = Form(default="GPT_SoVITS/pretrained_models/s2G2333k.pth"),
    s2config_path: str = Form(default="GPT_SoVITS/configs/s2.json"),
    version: str = Form(default=None),
    device: str = Form(default="auto"),
    is_half: bool = Form(default=True),
    n_parts: int = Form(default=1),
    output_format: str = Form(default="tsv")
):
    """
    上传文件并编码
    
    Args:
        text_file: 标注文件
        cnhubert_archive: CNHubert特征压缩包
        experiment_name: 实验名称
        其他参数: 编码配置参数
        
    Returns:
        编码结果文件
    """
    if not semantic_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 保存标注文件
        text_file_path = os.path.join(temp_dir, text_file.filename)
        with open(text_file_path, "wb") as buffer:
            shutil.copyfileobj(text_file.file, buffer)
        
        # 解压CNHubert特征
        cnhubert_dir = os.path.join(temp_dir, "cnhubert")
        os.makedirs(cnhubert_dir, exist_ok=True)
        
        # 假设是zip或tar格式
        archive_path = os.path.join(temp_dir, cnhubert_archive.filename)
        with open(archive_path, "wb") as buffer:
            shutil.copyfileobj(cnhubert_archive.file, buffer)
        
        # 解压文件
        if archive_path.endswith('.zip'):
            import zipfile
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(cnhubert_dir)
        elif archive_path.endswith(('.tar', '.tar.gz', '.tgz')):
            import tarfile
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(cnhubert_dir)
        else:
            raise HTTPException(status_code=400, detail="不支持的压缩格式")
        
        # 创建输出目录
        output_dir = os.path.join(temp_dir, "output")
        
        # 构建请求
        config = SemanticEncodingConfig(
            pretrained_s2G=pretrained_s2G,
            s2config_path=s2config_path,
            version=version,
            device=device,
            is_half=is_half,
            n_parts=n_parts,
            output_format=output_format
        )
        
        request = SemanticEncodingRequest(
            input_text_file=text_file_path,
            cnhubert_dir=cnhubert_dir,
            experiment_name=experiment_name,
            output_dir=output_dir,
            config=config
        )
        
        # 执行编码
        result = await semantic_api.encode_semantic(request)
        
        if result.success and result.output_file:
            # 返回编码结果文件
            filename = f"{experiment_name}_semantic.{output_format}"
            return FileResponse(
                result.output_file,
                media_type='text/plain' if output_format == 'tsv' else 'application/json',
                filename=filename
            )
        else:
            raise HTTPException(status_code=400, detail=result.message)


@app.post("/encode/batch")
async def batch_encode_endpoint(
    input_dir: str,
    output_dir: str,
    pretrained_s2G: str = "GPT_SoVITS/pretrained_models/s2G2333k.pth",
    s2config_path: str = "GPT_SoVITS/configs/s2.json",
    version: str = None,
    device: str = "auto",
    is_half: bool = True,
    n_parts: int = 1,
    output_format: str = "tsv"
):
    """
    批量编码接口
    
    Args:
        input_dir: 输入目录路径（包含标注文件和CNHubert特征）
        output_dir: 输出目录路径
        其他参数: 编码配置参数
        
    Returns:
        批量编码结果
    """
    if not semantic_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    # 批量分析目录
    batch_analysis = SemanticEncodingUtils.batch_analyze_directory(input_dir)
    
    if "error" in batch_analysis:
        raise HTTPException(status_code=400, detail=batch_analysis["error"])
    
    results = []
    config = SemanticEncodingConfig(
        pretrained_s2G=pretrained_s2G,
        s2config_path=s2config_path,
        version=version,
        device=device,
        is_half=is_half,
        n_parts=n_parts,
        output_format=output_format
    )
    
    # 处理每个有效的标注文件
    for file_path, analysis in batch_analysis["analyses"].items():
        if "error" not in analysis:
            # 推断CNHubert目录
            cnhubert_dir = os.path.join(os.path.dirname(file_path), "4-cnhubert")
            
            # 创建子输出目录
            file_name = Path(file_path).stem
            sub_output_dir = os.path.join(output_dir, file_name)
            
            request = SemanticEncodingRequest(
                input_text_file=file_path,
                cnhubert_dir=cnhubert_dir,
                experiment_name=file_name,
                output_dir=sub_output_dir,
                config=config
            )
            
            result = await semantic_api.encode_semantic(request)
            results.append({
                "input_file": file_path,
                "success": result.success,
                "message": result.message,
                "output_file": result.output_file,
                "processed_count": result.processed_count,
                "processing_time": result.processing_time
            })
    
    # 统计总体结果
    total_files = len(results)
    success_count = sum(1 for r in results if r["success"])
    total_processing_time = sum(r["processing_time"] for r in results)
    total_processed = sum(r["processed_count"] for r in results)
    
    return {
        "success": True,
        "message": f"批量编码完成",
        "total_files": total_files,
        "success_count": success_count,
        "failure_count": total_files - success_count,
        "total_processed_count": total_processed,
        "total_processing_time": total_processing_time,
        "results": results
    }


@app.post("/check-completeness")
async def check_output_completeness(
    output_dir: str,
    input_text_file: str,
    output_format: str = "tsv"
):
    """检查输出完整性"""
    if not semantic_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    # 解析输入文件获取期望的文件列表
    expected_files = []
    try:
        with open(input_text_file, "r", encoding="utf8") as f:
            lines = f.read().strip().split("\n")
        
        for line in lines:
            parts = line.split("|")
            if len(parts) >= 4:
                wav_name = os.path.basename(parts[0])
                wav_name = os.path.splitext(wav_name)[0]
                expected_files.append(wav_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"解析输入文件失败: {str(e)}")
    
    completeness = SemanticEncodingUtils.check_output_completeness(
        output_dir, expected_files, output_format
    )
    
    return completeness