#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 推理 FastAPI 服务器

提供HTTP API接口服务
"""

import os
import tempfile
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from .service import InferenceService, InferenceRequest, InferenceResponse, InferenceConfig
from .model_manager import ModelManager, ModelConfig


# 创建FastAPI应用
app = FastAPI(
    title="GPT-SoVITS 推理 API",
    description="基于GPT-SoVITS的语音合成推理服务",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化API实例
try:
    inference_api = InferenceService()
    model_manager = ModelManager()
except Exception as e:
    print(f"初始化失败: {e}")
    inference_api = None
    model_manager = None


@app.get("/")
async def root():
    """根路径"""
    return {"message": "GPT-SoVITS 推理 API", "status": "running"}


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy" if inference_api else "error",
        "gpt_sovits_root": inference_api.gpt_sovits_root if inference_api else None,
        "device": str(inference_api.device) if inference_api else None,
        "models_loaded": bool(inference_api.current_gpt_path and inference_api.current_sovits_path) if inference_api else False,
        "supported_languages": inference_api.get_supported_languages() if inference_api else [],
        "supported_formats": inference_api.get_supported_formats() if inference_api else []
    }


@app.post("/inference", response_model=InferenceResponse)
async def inference_endpoint(request: InferenceRequest):
    """
    语音合成推理
    
    Args:
        request: 推理请求参数
        
    Returns:
        InferenceResponse: 推理结果
    """
    if not inference_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    return await inference_api.inference(request)


@app.post("/inference/file")
async def inference_with_file_endpoint(
    text: str = Form(..., description="要合成的文本"),
    text_language: str = Form(default="zh", description="文本语言"),
    prompt_text: Optional[str] = Form(default=None, description="参考文本"),
    prompt_language: str = Form(default="zh", description="参考文本语言"),
    ref_audio: UploadFile = File(..., description="参考音频文件"),
    output_format: str = Form(default="wav", description="输出格式"),
    return_base64: bool = Form(default=False, description="是否返回Base64编码"),
    # 推理配置参数
    top_k: int = Form(default=20, description="Top-K采样参数"),
    top_p: float = Form(default=0.6, description="Top-P采样参数"),
    temperature: float = Form(default=0.6, description="温度参数"),
    speed: float = Form(default=1.0, description="语速调节"),
    how_to_cut: str = Form(default="不切", description="文本切分方式"),
    use_cuda_graph: bool = Form(default=False, description="尝试使用 CUDA Graph 加速普通推理"),
    cuda_graph_mode: str = Form(default="graph", description="CUDA Graph模式: graph 或 decoder_only")
):
    """
    使用文件上传的语音合成推理
    
    Returns:
        推理结果
    """
    if not inference_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    # 保存上传的音频文件
    temp_audio_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await ref_audio.read()
            temp_file.write(content)
            temp_audio_path = temp_file.name
        
        # 构建推理请求
        config = InferenceConfig(
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            speed=speed,
            how_to_cut=how_to_cut,
            use_cuda_graph=use_cuda_graph,
            cuda_graph_mode=cuda_graph_mode,
        )
        
        request = InferenceRequest(
            text=text,
            text_language=text_language,
            ref_audio_path=temp_audio_path,
            prompt_text=prompt_text,
            prompt_language=prompt_language,
            config=config,
            output_format=output_format,
            return_base64=return_base64
        )
        
        # 执行推理
        result = await inference_api.inference(request)
        
        return result
        
    finally:
        # 清理临时文件
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)


@app.get("/inference/download/{job_id}")
async def download_audio_endpoint(job_id: str):
    """
    下载生成的音频文件
    
    Args:
        job_id: 任务ID（实际为文件路径）
        
    Returns:
        音频文件下载
    """
    requested_path = Path(job_id).resolve()
    temp_root = Path(tempfile.gettempdir()).resolve()

    if requested_path.parent != temp_root or not requested_path.name.startswith("gpt_sovits_infer_"):
        raise HTTPException(status_code=400, detail="无效的音频文件标识")

    if not requested_path.exists():
        raise HTTPException(status_code=404, detail="音频文件不存在")
    
    return FileResponse(
        str(requested_path),
        media_type='audio/wav',
        filename=f"generated_audio_{requested_path.name}"
    )


# 模型管理相关接口
@app.post("/models/load")
async def load_models_endpoint(gpt_path: str, sovits_path: str):
    """
    加载GPT和SoVITS模型
    
    Args:
        gpt_path: GPT模型路径
        sovits_path: SoVITS模型路径
        
    Returns:
        加载结果
    """
    if not inference_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    success = inference_api.load_models(gpt_path, sovits_path)
    
    if success:
        return {"message": "模型加载成功", "success": True}
    else:
        raise HTTPException(status_code=400, detail="模型加载失败")


@app.get("/models/info")
async def get_model_info_endpoint():
    """
    获取当前模型信息
    
    Returns:
        模型信息
    """
    if not inference_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    return inference_api.get_model_info()


@app.post("/models/unload")
async def unload_models_endpoint():
    """
    卸载当前模型

    Returns:
        卸载结果
    """
    if not inference_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")

    unloaded = inference_api.unload_models(reason="manual")
    return {
        "message": "模型已卸载" if unloaded else "当前没有已加载模型",
        "success": True,
        "unloaded": unloaded,
    }


@app.post("/models/cleanup")
async def cleanup_models_endpoint(force: bool = False):
    """
    执行一次模型驻留清理

    Args:
        force: 是否强制执行

    Returns:
        清理结果
    """
    if not inference_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")

    return inference_api.cleanup_resident_models(force=force)


@app.post("/models/register")
async def register_model_endpoint(config: ModelConfig):
    """
    注册新模型
    
    Args:
        config: 模型配置
        
    Returns:
        注册结果
    """
    if not model_manager:
        raise HTTPException(status_code=500, detail="模型管理器未正确初始化")
    
    success = model_manager.register_model(config)
    
    if success:
        return {"message": f"模型 {config.name} 注册成功", "success": True}
    else:
        raise HTTPException(status_code=400, detail="模型注册失败")


@app.get("/models/list")
async def list_models_endpoint():
    """
    列出所有注册的模型
    
    Returns:
        模型列表
    """
    if not model_manager:
        raise HTTPException(status_code=500, detail="模型管理器未正确初始化")
    
    models = model_manager.list_models()
    return {
        "models": [
            {
                "name": model.name,
                "version": model.version,
                "description": model.description,
                "is_loaded": model.is_loaded,
                "file_size": model.file_size,
                "created_time": model.created_time.isoformat() if model.created_time else None
            }
            for model in models
        ],
        "current_model": model_manager.current_model
    }


@app.post("/models/switch/{model_name}")
async def switch_model_endpoint(model_name: str):
    """
    切换当前模型
    
    Args:
        model_name: 模型名称
        
    Returns:
        切换结果
    """
    if not model_manager or not inference_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    # 获取模型信息
    model_info = model_manager.get_model_info(model_name)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"模型不存在: {model_name}")
    
    # 加载模型
    success = inference_api.load_models(model_info.gpt_path, model_info.sovits_path)
    if not success:
        raise HTTPException(status_code=400, detail="模型加载失败")
    
    # 设置为当前模型
    model_manager.set_current_model(model_name)
    
    return {"message": f"已切换到模型: {model_name}", "success": True}


@app.delete("/models/{model_name}")
async def unregister_model_endpoint(model_name: str):
    """
    注销模型
    
    Args:
        model_name: 模型名称
        
    Returns:
        注销结果
    """
    if not model_manager:
        raise HTTPException(status_code=500, detail="模型管理器未正确初始化")
    
    success = model_manager.unregister_model(model_name)
    
    if success:
        return {"message": f"模型 {model_name} 注销成功", "success": True}
    else:
        raise HTTPException(status_code=404, detail=f"模型不存在: {model_name}")


@app.post("/models/validate/{model_name}")
async def validate_model_endpoint(model_name: str):
    """
    验证模型完整性
    
    Args:
        model_name: 模型名称
        
    Returns:
        验证结果
    """
    if not model_manager:
        raise HTTPException(status_code=500, detail="模型管理器未正确初始化")
    
    return model_manager.validate_model(model_name)


@app.get("/models/search")
async def search_models_endpoint(search_dir: str):
    """
    搜索目录中的模型文件
    
    Args:
        search_dir: 搜索目录
        
    Returns:
        找到的模型文件
    """
    if not model_manager:
        raise HTTPException(status_code=500, detail="模型管理器未正确初始化")
    
    return {
        "found_models": model_manager.search_models(search_dir),
        "search_dir": search_dir
    }


@app.get("/models/statistics")
async def get_model_statistics_endpoint():
    """
    获取模型统计信息
    
    Returns:
        统计信息
    """
    if not model_manager:
        raise HTTPException(status_code=500, detail="模型管理器未正确初始化")
    
    return model_manager.get_model_statistics()


# 工具接口
@app.post("/utils/clear_cache")
async def clear_cache_endpoint():
    """
    清理推理缓存
    
    Returns:
        操作结果
    """
    if not inference_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    inference_api.clear_cache()
    return {"message": "缓存已清理", "success": True}


@app.get("/utils/languages")
async def get_supported_languages_endpoint():
    """
    获取支持的语言列表
    
    Returns:
        支持的语言
    """
    if not inference_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    return {
        "languages": inference_api.get_supported_languages(),
        "default": "zh"
    }


@app.get("/utils/formats")
async def get_supported_formats_endpoint():
    """
    获取支持的音频格式
    
    Returns:
        支持的格式
    """
    if not inference_api:
        raise HTTPException(status_code=500, detail="API未正确初始化")
    
    return {
        "formats": inference_api.get_supported_formats(),
        "default": "wav"
    }


@app.get("/utils/config/template")
async def get_config_template_endpoint():
    """
    获取推理配置模板
    
    Returns:
        配置模板
    """
    config = InferenceConfig()
    return config.dict()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)
