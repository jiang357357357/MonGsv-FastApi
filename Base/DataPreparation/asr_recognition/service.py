#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS ASR语音识别 API 核心模块

提供语音识别的核心功能和数据模型
"""

import asyncio
import gc
import os
import time
import traceback
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Literal, Optional

import torch
from pydantic import BaseModel, Field

from Code.FastApi.Base.ASR.engines.asr import ASRManager as UnifiedASREngine, write_output_file
from Code.FastApi.Base.Inference.residency import ModelResidencyManager, ResidencyConfig


class ASRConfig(BaseModel):
    """ASR识别配置参数"""

    model_type: Literal["funasr", "faster_whisper"] = Field(default="funasr", description="ASR模型类型")
    model_size: str = Field(default="large", description="模型大小")
    language: str = Field(default="zh", description="识别语言")
    precision: Literal["float16", "float32", "int8"] = Field(default="float32", description="模型精度")
    batch_size: int = Field(default=1, description="批处理大小")
    vad_filter: bool = Field(default=True, description="是否启用VAD过滤")
    beam_size: int = Field(default=5, description="束搜索大小")


class ASRRequest(BaseModel):
    """ASR识别请求"""

    input_path: str = Field(description="输入音频文件或目录路径")
    output_dir: str = Field(description="输出目录路径")
    config: ASRConfig = Field(default_factory=ASRConfig)


class ASRResponse(BaseModel):
    """ASR识别响应"""

    success: bool
    message: str
    output_file: str = ""
    processed_files: List[str] = []
    recognition_results: List[Dict[str, str]] = []
    error_files: List[str] = []
    total_duration: float = 0.0
    processing_time: float = 0.0


class ASRRecognitionService:
    """ASR语音识别API类"""

    def __init__(self, gpt_sovits_root: str = None):
        self.gpt_sovits_root = gpt_sovits_root or self._find_gpt_sovits_root()
        self._model_lock = RLock()
        self._request_lock = RLock()
        self.current_model: Optional[UnifiedASREngine] = None
        self.current_model_key: Optional[str] = None
        self.current_model_descriptor: Optional[str] = None
        self.current_model_config: Optional[ASRConfig] = None
        self.current_model_type: Optional[str] = None
        self._whisper_available = True

        self.asr_models = {
            "funasr": {
                "languages": ["zh", "yue"],
                "sizes": ["large"],
                "precisions": ["float32"],
            },
            "faster_whisper": {
                "languages": ["auto", "en", "ja", "ko", "zh", "yue"],
                "sizes": ["medium", "medium.en", "large-v2", "large-v3", "large-v3-turbo"],
                "precisions": ["float32", "float16", "int8"],
            },
        }

        self.residency_manager = ModelResidencyManager(
            ResidencyConfig(
                idle_ttl_seconds=int(os.environ.get("ASR_IDLE_TTL_SECONDS", "1200")),
                max_loaded_models=int(os.environ.get("ASR_MAX_LOADED_MODELS", "1")),
                cleanup_interval_seconds=int(os.environ.get("ASR_CLEANUP_INTERVAL_SECONDS", "60")),
            )
        )

    def _find_gpt_sovits_root(self) -> str:
        """自动查找GPT-SoVITS项目根目录"""
        current_dir = Path(__file__).parent

        for parent in current_dir.parents:
            gpt_sovits_dir = parent / "GPT_SoVITS"
            if gpt_sovits_dir.exists():
                return str(parent)

        possible_paths = [
            "../../../../文档/GPT-SoVITS-main",
            "../../../GPT-SoVITS-main",
            "../../GPT-SoVITS-main",
        ]

        for path in possible_paths:
            abs_path = Path(__file__).parent / path
            if abs_path.exists() and (abs_path / "GPT_SoVITS").exists():
                return str(abs_path.resolve())

        raise FileNotFoundError("无法找到GPT-SoVITS项目根目录")

    def _clean_path(self, path_str: str) -> str:
        if path_str.endswith(("\\", "/")):
            return self._clean_path(path_str[0:-1])
        path_str = path_str.replace("/", os.sep).replace("\\", os.sep)
        return path_str.strip(" '\n\"\u202a")

    def _validate_input(self, input_path: str) -> bool:
        clean_path = self._clean_path(input_path)
        return os.path.exists(clean_path)

    def _validate_config(self, config: ASRConfig) -> bool:
        model_info = self.asr_models.get(config.model_type)
        if not model_info:
            return False
        if config.language not in model_info["languages"]:
            return False
        if config.model_size not in model_info["sizes"]:
            return False
        if config.precision not in model_info["precisions"]:
            return False
        return True

    def _normalize_config(self, config: ASRConfig) -> ASRConfig:
        return config

    def _build_model_descriptor(self, config: ASRConfig) -> str:
        if config.model_type == "funasr":
            return f"funasr-large/{config.language}"
        return f"whisper/{config.model_size}/{config.precision}"

    def _build_model_key(self, descriptor: str) -> str:
        return self.residency_manager.build_model_key("asr", descriptor)

    def _load_primary_model(self, config: ASRConfig):
        if config.model_type == "funasr":
            return UnifiedASREngine(engine=UnifiedASREngine.ENGINE_FUNASR)
        if config.model_type == "faster_whisper":
            return UnifiedASREngine(engine=UnifiedASREngine.ENGINE_WHISPER)
        raise ValueError(f"不支持的ASR模型类型: {config.model_type}")

    def _cleanup_runtime(self):
        self.current_model = None
        self.current_model_key = None
        self.current_model_descriptor = None
        self.current_model_config = None
        self.current_model_type = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_supported_models(self) -> Dict[str, Dict]:
        """获取支持的模型信息"""
        return self.asr_models.copy()

    def suggest_config(self, language: str = "zh") -> ASRConfig:
        """根据语言建议最佳配置"""
        if language in ["zh", "yue"]:
            return ASRConfig(
                model_type="funasr",
                model_size="large",
                language=language,
                precision="float32",
            )
        return ASRConfig(
            model_type="faster_whisper",
            model_size="large-v3",
            language=language if language != "auto" else "auto",
            precision="float16",
        )

    def load_models(self, config: ASRConfig) -> bool:
        normalized_config = self._normalize_config(config)
        descriptor = self._build_model_descriptor(normalized_config)
        requested_key = self._build_model_key(descriptor)

        with self._model_lock:
            try:
                if self.current_model is not None and self.current_model_key == requested_key:
                    print(f"[asr-residency] 命中已驻留模型: {descriptor}")
                    self.current_model_config = normalized_config
                    self.residency_manager.register_loaded("asr", descriptor)
                    return True

                if self.current_model is not None:
                    print("[asr-residency] 切换模型，卸载当前驻留模型")
                    self.unload_models(reason="switch")

                print(f"[asr-residency] 加载模型: {descriptor}")
                self.current_model = self._load_primary_model(normalized_config)
                self.current_model_key = requested_key
                self.current_model_descriptor = descriptor
                self.current_model_config = normalized_config
                self.current_model_type = normalized_config.model_type
                self.residency_manager.register_loaded("asr", descriptor)
                cleanup_result = self.cleanup_resident_models(force=True)
                if cleanup_result.get("unloaded"):
                    print(f"[asr-residency] 清理驻留模型: {cleanup_result['unloaded']}")
                print("[asr-residency] 模型已驻留")
                return True
            except Exception as exc:
                print(f"ASR模型加载失败: {exc}")
                traceback.print_exc()
                self._cleanup_runtime()
                return False

    def unload_models(self, reason: str = "manual") -> bool:
        with self._model_lock:
            current_key = self.current_model_key
            if self.current_model is None:
                if current_key:
                    self.residency_manager.mark_unloaded(current_key, reason=reason)
                return False
            print(f"[asr-residency] 卸载模型: reason={reason}")
            self._cleanup_runtime()
            if current_key:
                self.residency_manager.mark_unloaded(current_key, reason=reason)
            return True

    def cleanup_resident_models(self, force: bool = False) -> Dict[str, Any]:
        """按驻留策略清理空闲模型。"""
        with self._model_lock:
            if not force and not self.residency_manager.should_run_cleanup():
                return {
                    "success": True,
                    "message": "cleanup skipped",
                    "unloaded": [],
                }

            unloaded: List[Dict[str, str]] = []
            current_key = self.current_model_key
            for record in self.residency_manager.select_eviction_candidates():
                if record.active_requests > 0:
                    continue
                reason = "idle_timeout" if record.is_idle_expired(self.residency_manager.config) else "evicted"
                if current_key and record.model_key == current_key:
                    self.unload_models(reason=reason)
                else:
                    self.residency_manager.mark_unloaded(record.model_key, reason=reason)
                print(f"[asr-residency] 标记清理: reason={reason}, model_key={record.model_key}")
                unloaded.append(
                    {
                        "model_key": record.model_key,
                        "reason": reason,
                    }
                )

            self.residency_manager.mark_cleanup_run()
            return {
                "success": True,
                "message": "cleanup completed",
                "unloaded": unloaded,
            }

    def _execute_recognition(self, input_path: str, output_dir: str, config: ASRConfig) -> tuple[str, List[Dict[str, str]]]:
        if self.current_model is None:
            raise RuntimeError("ASR模型未加载")

        engine_type = ""
        if config.model_type == "funasr":
            engine_type = UnifiedASREngine.ENGINE_FUNASR
        if config.model_type == "faster_whisper":
            engine_type = UnifiedASREngine.ENGINE_WHISPER

        output_file_name, recognition_results = self.current_model.batch_transcribe(
            input_path, config.language, engine=engine_type,
        )
        output_file = write_output_file(output_dir, output_file_name, recognition_results)
        return output_file, recognition_results

    def _recognize_audio_internal(self, request: ASRRequest) -> ASRResponse:
        start_time = time.time()
        model_key: Optional[str] = None
        request_started = False

        try:
            input_path = self._clean_path(request.input_path)
            output_dir = self._clean_path(request.output_dir)

            if not self._validate_input(input_path):
                return ASRResponse(
                    success=False,
                    message=f"输入路径不存在: {input_path}",
                    processing_time=time.time() - start_time,
                )

            normalized_config = self._normalize_config(request.config)
            if not self._validate_config(normalized_config):
                return ASRResponse(
                    success=False,
                    message=f"不支持的配置: {normalized_config.model_type}/{normalized_config.language}",
                    processing_time=time.time() - start_time,
                )

            os.makedirs(output_dir, exist_ok=True)

            with self._request_lock:
                if not self.load_models(normalized_config):
                    return ASRResponse(
                        success=False,
                        message="ASR模型加载失败",
                        error_files=[input_path],
                        processing_time=time.time() - start_time,
                    )

                model_key = self.current_model_key
                self.residency_manager.begin_request(model_key)
                request_started = True
                if model_key:
                    print(f"[asr-residency] 开始识别: model_key={model_key}")

                output_file, recognition_results = self._execute_recognition(input_path, output_dir, normalized_config)
                processed_files = [item["audio_path"] for item in recognition_results]

                return ASRResponse(
                    success=True,
                    message="识别完成",
                    output_file=output_file,
                    processed_files=processed_files,
                    recognition_results=recognition_results,
                    processing_time=time.time() - start_time,
                )
        except Exception as exc:
            return ASRResponse(
                success=False,
                message=f"识别过程出错: {exc}",
                error_files=[request.input_path],
                processing_time=time.time() - start_time,
            )
        finally:
            if request_started:
                self.residency_manager.end_request(model_key)
                if model_key:
                    print(f"[asr-residency] 结束识别: model_key={model_key}")
            self.cleanup_resident_models()

    async def recognize_audio(self, request: ASRRequest) -> ASRResponse:
        """执行语音识别"""
        return await asyncio.to_thread(self._recognize_audio_internal, request)

    async def process(self, request: ASRRequest) -> ASRResponse:
        """基础层统一入口。"""
        return await self.recognize_audio(request)

    def recognize_audio_sync(self, request: ASRRequest) -> ASRResponse:
        """
        同步版本的语音识别

        Args:
            request: 识别请求参数

        Returns:
            ASRResponse: 识别结果
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.recognize_audio(request))
        finally:
            loop.close()

    async def batch_recognize(self, input_dir: str, output_dir: str, config: ASRConfig = None) -> List[ASRResponse]:
        """
        批量识别目录中的音频文件

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            config: 识别配置

        Returns:
            List[ASRResponse]: 批量识别结果
        """
        if config is None:
            config = ASRConfig()

        audio_extensions = [".wav", ".mp3", ".flac", ".m4a", ".aac"]
        audio_files = []

        for ext in audio_extensions:
            audio_files.extend(Path(input_dir).glob(f"*{ext}"))
            audio_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))

        results = []
        for audio_file in audio_files:
            file_output_dir = os.path.join(output_dir, audio_file.stem)
            request = ASRRequest(
                input_path=str(audio_file),
                output_dir=file_output_dir,
                config=config,
            )
            result = await self.recognize_audio(request)
            results.append(result)

        return results

    def _get_residency_status(self) -> Dict[str, Any]:
        status = self.residency_manager.get_status()
        records = []
        for record in self.residency_manager.list_records():
            records.append(
                {
                    "model_key": record.model_key,
                    "descriptor": record.sovits_path,
                    "loaded_at": record.loaded_at.isoformat(),
                    "last_used_at": record.last_used_at.isoformat(),
                    "active_requests": record.active_requests,
                    "status": record.status,
                    "unload_reason": record.unload_reason,
                }
            )
        status["records"] = records
        status["current_model_key"] = self.current_model_key
        return status

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_descriptor": self.current_model_descriptor,
            "model_config": self.current_model_config.model_dump() if self.current_model_config else None,
            "models_loaded": self.current_model is not None,
            "supported_models": self.get_supported_models(),
            "residency": self._get_residency_status(),
        }
