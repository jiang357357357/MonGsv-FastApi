#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS API客户端

提供对GPT-SoVITS各个功能模块的统一调用接口
"""

import asyncio
import aiohttp
import json
import uuid
import time
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import base64

from ..models import *
from .exceptions import *


class GPTSoVITSClient:
    """GPT-SoVITS API客户端"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:8000",
                 api_key: Optional[str] = None,
                 timeout: int = 300,
                 max_retries: int = 3):
        """
        初始化客户端
        
        Args:
            base_url: API服务基础URL
            api_key: API密钥（可选）
            timeout: 默认超时时间
            max_retries: 最大重试次数
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
    
    async def _ensure_session(self):
        """确保会话存在"""
        if self.session is None or self.session.closed:
            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout
            )
    
    async def close(self):
        """关闭会话"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def _request(self, method: str, endpoint: str, 
                      data: Optional[Dict] = None,
                      files: Optional[Dict] = None,
                      timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        发送HTTP请求
        
        Args:
            method: HTTP方法
            endpoint: API端点
            data: 请求数据
            files: 文件数据
            timeout: 超时时间
            
        Returns:
            响应数据
        """
        await self._ensure_session()
        
        url = f"{self.base_url}{endpoint}"
        request_timeout = timeout or self.timeout
        
        for attempt in range(self.max_retries + 1):
            try:
                if files:
                    # 文件上传请求
                    form_data = aiohttp.FormData()
                    
                    # 添加表单字段
                    if data:
                        for key, value in data.items():
                            if isinstance(value, (dict, list)):
                                form_data.add_field(key, json.dumps(value))
                            else:
                                form_data.add_field(key, str(value))
                    
                    # 添加文件
                    for key, file_path in files.items():
                        if isinstance(file_path, str) and Path(file_path).exists():
                            with open(file_path, 'rb') as f:
                                form_data.add_field(key, f.read(), 
                                                  filename=Path(file_path).name)
                    
                    async with self.session.request(
                        method, url, data=form_data,
                        timeout=aiohttp.ClientTimeout(total=request_timeout)
                    ) as response:
                        return await self._handle_response(response)
                
                else:
                    # JSON请求
                    async with self.session.request(
                        method, url, json=data,
                        timeout=aiohttp.ClientTimeout(total=request_timeout)
                    ) as response:
                        return await self._handle_response(response)
            
            except asyncio.TimeoutError:
                if attempt == self.max_retries:
                    raise TimeoutError(f"请求超时: {url}")
                await asyncio.sleep(2 ** attempt)  # 指数退避
            
            except aiohttp.ClientError as e:
                if attempt == self.max_retries:
                    raise GPTSoVITSAPIError(f"网络请求失败: {str(e)}")
                await asyncio.sleep(2 ** attempt)
    
    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """处理HTTP响应"""
        try:
            response_data = await response.json()
        except:
            response_data = {"message": await response.text()}
        
        if response.status == 200:
            return response_data
        elif response.status == 400:
            raise ValidationError(response_data.get("message", "参数验证失败"))
        elif response.status == 401:
            raise AuthenticationError(response_data.get("message", "认证失败"))
        elif response.status == 404:
            raise GPTSoVITSAPIError(response_data.get("message", "资源不存在"), 404)
        elif response.status == 429:
            raise RateLimitError(response_data.get("message", "请求频率过高"))
        elif response.status == 503:
            raise ServiceUnavailableError("unknown", response_data.get("message", "服务不可用"))
        else:
            raise GPTSoVITSAPIError(
                response_data.get("message", f"HTTP {response.status}"),
                response.status
            )
    
    def _add_request_id(self, request: BaseRequest) -> BaseRequest:
        """添加请求ID"""
        if not request.request_id:
            request.request_id = str(uuid.uuid4())
        return request
    
    # ==================== 健康检查 ====================
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return await self._request("GET", "/health")
    
    async def get_services_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return await self._request("GET", "/services/status")
    
    # ==================== 数据准备阶段 ====================
    
    async def audio_slice(self, request: AudioSliceRequest) -> AudioSliceResponse:
        """音频切分"""
        request = self._add_request_id(request)
        
        data = {
            "input_path": request.input_path,
            "output_dir": request.output_dir,
            "threshold": request.config.threshold,
            "min_length": request.config.min_length,
            "min_interval": request.config.min_interval,
            "max_sil_kept": request.config.max_sil_kept
        }
        
        response_data = await self._request(
            "POST", "/data-prep/audio-slice/process", 
            data=data, timeout=request.timeout
        )
        
        return AudioSliceResponse(**response_data, request_id=request.request_id)
    
    async def asr_recognize(self, request: ASRRequest) -> ASRResponse:
        """ASR语音识别"""
        request = self._add_request_id(request)
        
        data = {
            "audio_dir": request.input_path,
            "output_file": request.output_file,
            "model_type": request.config.model_type,
            "language": request.config.language,
            "precision": request.config.precision,
            "batch_size": request.config.batch_size
        }
        
        response_data = await self._request(
            "POST", "/data-prep/asr/recognize",
            data=data, timeout=request.timeout
        )
        
        return ASRResponse(**response_data, request_id=request.request_id)
    
    # ==================== 数据格式化阶段 ====================
    
    async def text_processing(self, request: TextProcessingRequest) -> TextProcessingResponse:
        """文本特征提取"""
        request = self._add_request_id(request)
        
        data = {
            "list_file": request.list_file,
            "output_dir": request.output_dir,
            "language": request.config.language,
            "bert_model": request.config.bert_model,
            "batch_size": request.config.batch_size
        }
        
        response_data = await self._request(
            "POST", "/dataset/text/extract",
            data=data, timeout=request.timeout
        )
        
        return TextProcessingResponse(**response_data, request_id=request.request_id)
    
    async def audio_features(self, request: AudioFeaturesRequest) -> AudioFeaturesResponse:
        """音频特征提取"""
        request = self._add_request_id(request)
        
        data = {
            "list_file": request.list_file,
            "output_dir": request.output_dir,
            "version": request.config.version,
            "device": request.config.device,
            "batch_size": request.config.batch_size,
            "n_processes": request.config.n_processes
        }
        
        response_data = await self._request(
            "POST", "/dataset/audio/extract",
            data=data, timeout=request.timeout
        )
        
        return AudioFeaturesResponse(**response_data, request_id=request.request_id)
    
    async def semantic_encoding(self, request: SemanticEncodingRequest) -> SemanticEncodingResponse:
        """语义编码"""
        request = self._add_request_id(request)
        
        data = {
            "list_file": request.list_file,
            "output_dir": request.output_dir,
            "version": request.config.version,
            "device": request.config.device,
            "batch_size": request.config.batch_size
        }
        
        response_data = await self._request(
            "POST", "/dataset/semantic/encode",
            data=data, timeout=request.timeout
        )
        
        return SemanticEncodingResponse(**response_data, request_id=request.request_id)
    
    # ==================== 训练阶段 ====================
    
    async def start_gpt_training(self, request: GPTTrainingRequest) -> GPTTrainingResponse:
        """开始GPT训练"""
        request = self._add_request_id(request)
        
        data = {
            "exp_name": request.exp_name,
            "exp_root": request.exp_root,
            "batch_size": request.config.batch_size,
            "total_epoch": request.config.total_epoch,
            "learning_rate": request.config.learning_rate,
            "save_every_epoch": request.config.save_every_epoch,
            "gpu_numbers": request.config.gpu_numbers
        }
        
        response_data = await self._request(
            "POST", "/training/gpt/start",
            data=data, timeout=request.timeout
        )
        
        return GPTTrainingResponse(**response_data, request_id=request.request_id)
    
    async def start_sovits_training(self, request: SoVITSTrainingRequest) -> SoVITSTrainingResponse:
        """开始SoVITS训练"""
        request = self._add_request_id(request)
        
        data = {
            "exp_name": request.exp_name,
            "exp_root": request.exp_root,
            "version": request.config.version,
            "batch_size": request.config.batch_size,
            "total_epoch": request.config.total_epoch,
            "learning_rate": request.config.learning_rate,
            "save_every_epoch": request.config.save_every_epoch,
            "gpu_numbers": request.config.gpu_numbers
        }
        
        response_data = await self._request(
            "POST", "/training/sovits/start",
            data=data, timeout=request.timeout
        )
        
        return SoVITSTrainingResponse(**response_data, request_id=request.request_id)
    
    async def get_training_status(self, job_id: str) -> TrainingStatus:
        """获取训练状态"""
        response_data = await self._request("GET", f"/training/status/{job_id}")
        return TrainingStatus(**response_data["status"])
    
    async def stop_training(self, job_id: str) -> Dict[str, Any]:
        """停止训练"""
        return await self._request("POST", f"/training/stop/{job_id}")
    
    # ==================== 推理阶段 ====================
    
    async def inference(self, request: InferenceRequest) -> InferenceResponse:
        """文本转语音推理"""
        request = self._add_request_id(request)
        
        data = {
            "text": request.text,
            "text_language": request.text_language,
            "prompt_text": request.prompt_text,
            "prompt_language": request.prompt_language,
            "top_k": request.config.top_k,
            "top_p": request.config.top_p,
            "temperature": request.config.temperature,
            "how_to_cut": request.config.how_to_cut,
            "speed": request.config.speed,
            "output_format": request.output_format,
            "return_base64": request.return_base64
        }
        
        files = {}
        if request.ref_audio_path:
            files["ref_audio"] = request.ref_audio_path
        
        response_data = await self._request(
            "POST", "/inference/tts",
            data=data, files=files, timeout=request.timeout
        )
        
        return InferenceResponse(**response_data, request_id=request.request_id)
    
    async def inference_with_base64_audio(self, request: InferenceRequest) -> InferenceResponse:
        """使用Base64音频进行推理"""
        if not request.ref_audio_base64:
            raise ValidationError("ref_audio_base64 is required")
        
        request = self._add_request_id(request)
        
        data = {
            "text": request.text,
            "text_language": request.text_language,
            "ref_audio_base64": request.ref_audio_base64,
            "prompt_text": request.prompt_text,
            "prompt_language": request.prompt_language,
            "top_k": request.config.top_k,
            "top_p": request.config.top_p,
            "temperature": request.config.temperature,
            "how_to_cut": request.config.how_to_cut,
            "speed": request.config.speed,
            "output_format": request.output_format,
            "return_base64": request.return_base64
        }
        
        response_data = await self._request(
            "POST", "/inference/tts",
            data=data, timeout=request.timeout
        )
        
        return InferenceResponse(**response_data, request_id=request.request_id)
    
    # ==================== 工作流 ====================
    
    async def complete_workflow(self, request: WorkflowRequest) -> WorkflowResponse:
        """执行完整工作流"""
        request = self._add_request_id(request)
        
        data = {
            "project_name": request.project_name,
            "input_audio_dir": request.input_audio_dir,
            "output_dir": request.output_dir,
            "language": request.config.language,
            "version": request.config.version,
            "skip_existing": request.config.skip_existing,
            "parallel_processing": request.config.parallel_processing
        }
        
        response_data = await self._request(
            "POST", "/workflow/complete",
            data=data, timeout=request.timeout
        )
        
        return WorkflowResponse(**response_data, request_id=request.request_id)
    
    async def batch_process(self, request: BatchRequest) -> BatchResponse:
        """批量处理项目"""
        request = self._add_request_id(request)
        
        data = {
            "projects": [project.dict() for project in request.projects],
            "max_concurrent": request.max_concurrent
        }
        
        response_data = await self._request(
            "POST", "/batch/projects",
            data=data, timeout=request.timeout
        )
        
        return BatchResponse(**response_data, request_id=request.request_id)
    
    # ==================== 工具方法 ====================
    
    def encode_audio_file(self, file_path: str) -> str:
        """将音频文件编码为Base64"""
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def save_base64_audio(self, base64_data: str, output_path: str):
        """保存Base64音频到文件"""
        audio_data = base64.b64decode(base64_data)
        with open(output_path, 'wb') as f:
            f.write(audio_data)
    
    async def wait_for_training_completion(self, job_id: str, 
                                         check_interval: int = 30,
                                         max_wait_time: int = 3600) -> TrainingStatus:
        """等待训练完成"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status = await self.get_training_status(job_id)
            
            if status.status in ["completed", "failed", "stopped"]:
                return status
            
            await asyncio.sleep(check_interval)
        
        raise TimeoutError(f"训练任务 {job_id} 等待超时")


# ==================== 同步客户端 ====================

class SyncGPTSoVITSClient:
    """同步版本的GPT-SoVITS客户端"""
    
    def __init__(self, **kwargs):
        self.async_client = GPTSoVITSClient(**kwargs)
    
    def _run_async(self, coro):
        """运行异步方法"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        async def _impl():
            async with self.async_client as client:
                return await client.health_check()
        return self._run_async(_impl())
    
    def audio_slice(self, request: AudioSliceRequest) -> AudioSliceResponse:
        """音频切分"""
        async def _impl():
            async with self.async_client as client:
                return await client.audio_slice(request)
        return self._run_async(_impl())
    
    def asr_recognize(self, request: ASRRequest) -> ASRResponse:
        """ASR识别"""
        async def _impl():
            async with self.async_client as client:
                return await client.asr_recognize(request)
        return self._run_async(_impl())
    
    def text_processing(self, request: TextProcessingRequest) -> TextProcessingResponse:
        """文本处理"""
        async def _impl():
            async with self.async_client as client:
                return await client.text_processing(request)
        return self._run_async(_impl())
    
    def audio_features(self, request: AudioFeaturesRequest) -> AudioFeaturesResponse:
        """音频特征"""
        async def _impl():
            async with self.async_client as client:
                return await client.audio_features(request)
        return self._run_async(_impl())
    
    def semantic_encoding(self, request: SemanticEncodingRequest) -> SemanticEncodingResponse:
        """语义编码"""
        async def _impl():
            async with self.async_client as client:
                return await client.semantic_encoding(request)
        return self._run_async(_impl())
    
    def start_gpt_training(self, request: GPTTrainingRequest) -> GPTTrainingResponse:
        """GPT训练"""
        async def _impl():
            async with self.async_client as client:
                return await client.start_gpt_training(request)
        return self._run_async(_impl())
    
    def start_sovits_training(self, request: SoVITSTrainingRequest) -> SoVITSTrainingResponse:
        """SoVITS训练"""
        async def _impl():
            async with self.async_client as client:
                return await client.start_sovits_training(request)
        return self._run_async(_impl())
    
    def get_training_status(self, job_id: str) -> TrainingStatus:
        """训练状态"""
        async def _impl():
            async with self.async_client as client:
                return await client.get_training_status(job_id)
        return self._run_async(_impl())
    
    def inference(self, request: InferenceRequest) -> InferenceResponse:
        """推理"""
        async def _impl():
            async with self.async_client as client:
                return await client.inference(request)
        return self._run_async(_impl())
    
    def complete_workflow(self, request: WorkflowRequest) -> WorkflowResponse:
        """完整工作流"""
        async def _impl():
            async with self.async_client as client:
                return await client.complete_workflow(request)
        return self._run_async(_impl())