#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 推理基础服务。
"""

import asyncio
import base64
import gc
import io
import os
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from pydantic import BaseModel, Field
from Code.FastApi.Base.gpt_sovits_env import setup_gpt_sovits_paths
from .residency import ModelResidencyManager, ResidencyConfig


class InferenceConfig(BaseModel):
    """推理配置参数。"""

    top_k: int = Field(default=20, description="Top-K 采样参数")
    top_p: float = Field(default=0.6, description="Top-P 采样参数")
    temperature: float = Field(default=0.6, description="温度参数")
    how_to_cut: str = Field(default="凑四句一切", description="文本切分方式")
    speed: float = Field(default=1.0, description="语速调节")
    pause_second: float = Field(default=0.3, description="句间停顿时长(秒)")
    ref_free: bool = Field(default=False, description="是否启用无参考模式")
    if_freeze: bool = Field(default=False, description="是否冻结缓存")
    if_sr: bool = Field(default=False, description="是否启用音频超分辨率")
    batch_size: int = Field(default=1, description="批处理大小")
    split_bucket: bool = Field(default=True, description="是否分桶处理")
    fragment_interval: float = Field(default=0.3, description="片段间隔")
    seed: int = Field(default=-1, description="随机种子，-1 为随机")
    parallel_infer: bool = Field(default=True, description="是否并行推理")
    repetition_penalty: float = Field(default=1.35, description="重复惩罚")
    sample_steps: int = Field(default=8, description="采样步数")
    batch_threshold: float = Field(default=0.75, description="分桶阈值")
    super_sampling: bool = Field(default=False, description="是否启用超分")
    streaming_mode: bool = Field(default=False, description="是否流式推理")
    return_fragment: bool = Field(default=False, description="是否返回分段音频")


class InferenceRequest(BaseModel):
    """推理请求。"""

    text: str = Field(description="要合成的文本")
    text_language: str = Field(default="zh", description="文本语言")
    ref_audio_path: Optional[str] = Field(default=None, description="参考音频文件路径")
    ref_audio_base64: Optional[str] = Field(default=None, description="参考音频 Base64 编码")
    prompt_text: Optional[str] = Field(default=None, description="参考文本")
    prompt_language: str = Field(default="zh", description="参考文本语言")
    aux_ref_audio_paths: Optional[List[str]] = Field(default=None, description="辅助参考音频路径列表")
    config: InferenceConfig = Field(default_factory=InferenceConfig)
    output_format: str = Field(default="wav", description="输出格式")
    return_base64: bool = Field(default=False, description="是否返回 Base64 编码的音频")


class InferenceResponse(BaseModel):
    """推理响应。"""

    success: bool
    message: str
    audio_data: Optional[str] = None
    audio_path: Optional[str] = None
    sample_rate: Optional[int] = None
    duration: Optional[float] = None
    processing_time: Optional[float] = None
    text_segments: Optional[List[str]] = None
    error_details: Optional[str] = None


class InferenceService:
    """GPT-SoVITS 推理服务包装。"""

    CUT_METHODS = {
        "不切": "cut0",
        "凑四句一切": "cut1",
        "凑50字一切": "cut2",
        "按中文句号。切": "cut3",
        "按英文句号.切": "cut4",
        "按标点符号切": "cut5",
        "cut0": "cut0",
        "cut1": "cut1",
        "cut2": "cut2",
        "cut3": "cut3",
        "cut4": "cut4",
        "cut5": "cut5",
    }

    DEFAULT_LANGUAGES = [
        "auto",
        "auto_yue",
        "zh",
        "en",
        "ja",
        "yue",
        "ko",
        "all_zh",
        "all_ja",
        "all_yue",
        "all_ko",
    ]

    def __init__(self, gpt_sovits_root: str = None):
        self.gpt_sovits_root = gpt_sovits_root or self._find_gpt_sovits_root()
        self.device = self._get_device()
        self.is_half = torch.cuda.is_available()
        self.current_gpt_path: Optional[str] = None
        self.current_sovits_path: Optional[str] = None
        self.model_version: Optional[str] = None
        self.tts_pipeline = None
        self.tts_config = None
        self._TTS = None
        self._TTS_Config = None
        self._detect_sovits_version = None
        self._get_text_split_method = None
        self.residency_manager = ModelResidencyManager(
            ResidencyConfig(
                idle_ttl_seconds=int(os.environ.get("INFERENCE_IDLE_TTL_SECONDS", "1200")),
                max_loaded_models=int(os.environ.get("INFERENCE_MAX_LOADED_MODELS", "1")),
                cleanup_interval_seconds=int(os.environ.get("INFERENCE_CLEANUP_INTERVAL_SECONDS", "60")),
            )
        )
        self._setup_inference_environment()

    def _find_gpt_sovits_root(self) -> str:
        """自动查找 GPT-SoVITS 项目根目录。"""
        return str(setup_gpt_sovits_paths(Path(__file__).resolve().parent))

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _setup_inference_environment(self):
        """设置导入环境。"""
        setup_gpt_sovits_paths(Path(self.gpt_sovits_root))
        self._import_modules()

    def _import_modules(self):
        """延迟导入官方推理管线。"""
        from TTS_infer_pack.TTS import TTS, TTS_Config
        from TTS_infer_pack.text_segmentation_method import get_method
        from process_ckpt import get_sovits_version_from_path_fast

        self._TTS = TTS
        self._TTS_Config = TTS_Config
        self._get_text_split_method = get_method
        self._detect_sovits_version = get_sovits_version_from_path_fast

    def _build_tts_config(self, gpt_path: str, sovits_path: str):
        """构建官方 TTS 配置对象。"""
        version, _, _ = self._detect_sovits_version(sovits_path)
        config_path = os.path.join(self.gpt_sovits_root, "GPT_SoVITS", "configs", "tts_infer.yaml")
        tts_config = self._TTS_Config(config_path)
        tts_config.device = str(self.device)
        tts_config.is_half = self.is_half and str(self.device) != "cpu"
        tts_config.update_version(version)
        tts_config.t2s_weights_path = gpt_path
        tts_config.vits_weights_path = sovits_path
        tts_config.bert_base_path = os.path.join(
            self.gpt_sovits_root, "GPT_SoVITS", "pretrained_models", "chinese-roberta-wwm-ext-large"
        )
        tts_config.cnhuhbert_base_path = os.path.join(
            self.gpt_sovits_root, "GPT_SoVITS", "pretrained_models", "chinese-hubert-base"
        )
        tts_config.update_configs()
        return tts_config

    def load_models(self, gpt_path: str, sovits_path: str) -> bool:
        """加载 GPT 与 SoVITS 权重。"""
        try:
            if not os.path.exists(gpt_path):
                raise FileNotFoundError(f"GPT 模型文件不存在: {gpt_path}")
            if not os.path.exists(sovits_path):
                raise FileNotFoundError(f"SoVITS 模型文件不存在: {sovits_path}")

            current_key = self.get_current_model_key()
            requested_key = self.residency_manager.build_model_key(gpt_path, sovits_path)
            if self.tts_pipeline is not None and current_key == requested_key:
                print(f"[inference-residency] 命中已驻留模型: {os.path.basename(gpt_path)} / {os.path.basename(sovits_path)}")
                self.residency_manager.register_loaded(gpt_path, sovits_path)
                return True

            if self.tts_pipeline is not None:
                print("[inference-residency] 切换模型，卸载当前驻留模型")
                self.unload_models(reason="switch")

            print(f"[inference-residency] 加载模型: {os.path.basename(gpt_path)} / {os.path.basename(sovits_path)}")
            self._cleanup_runtime()
            self.tts_config = self._build_tts_config(gpt_path, sovits_path)
            self.tts_pipeline = self._TTS(self.tts_config)
            self.current_gpt_path = gpt_path
            self.current_sovits_path = sovits_path
            self.model_version = self.tts_pipeline.configs.version
            self.residency_manager.register_loaded(gpt_path, sovits_path)
            cleanup_result = self.cleanup_resident_models(force=True)
            if cleanup_result.get("unloaded"):
                print(f"[inference-residency] 清理驻留模型: {cleanup_result['unloaded']}")
            print("[inference-residency] 模型已驻留")
            return True
        except Exception as exc:
            print(f"模型加载失败: {exc}")
            traceback.print_exc()
            self._cleanup_runtime()
            return False

    def _cleanup_runtime(self):
        """清理运行时对象。"""
        if self.tts_pipeline is not None and hasattr(self.tts_pipeline, "stop"):
            try:
                self.tts_pipeline.stop()
            except Exception:
                pass
        self.tts_pipeline = None
        self.tts_config = None
        self.current_gpt_path = None
        self.current_sovits_path = None
        self.model_version = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_current_model_key(self) -> Optional[str]:
        if not self.current_gpt_path or not self.current_sovits_path:
            return None
        return self.residency_manager.build_model_key(self.current_gpt_path, self.current_sovits_path)

    def unload_models(self, reason: str = "manual") -> bool:
        """卸载当前模型并释放内存。"""
        if self.tts_pipeline is None:
            current_key = self.get_current_model_key()
            if current_key:
                self.residency_manager.mark_unloaded(current_key, reason=reason)
            return False
        current_key = self.get_current_model_key()
        print(f"[inference-residency] 卸载模型: reason={reason}")
        self._cleanup_runtime()
        if current_key:
            self.residency_manager.mark_unloaded(current_key, reason=reason)
        return True

    def cleanup_resident_models(self, force: bool = False) -> Dict[str, Any]:
        """按驻留策略清理空闲模型。"""
        if not force and not self.residency_manager.should_run_cleanup():
            return {
                "success": True,
                "message": "cleanup skipped",
                "unloaded": [],
            }

        unloaded: List[Dict[str, str]] = []
        current_key = self.get_current_model_key()
        for record in self.residency_manager.select_eviction_candidates():
            if record.active_requests > 0:
                continue
            reason = "idle_timeout" if record.is_idle_expired(self.residency_manager.config) else "evicted"
            if current_key and record.model_key == current_key:
                self.unload_models(reason=reason)
            else:
                self.residency_manager.mark_unloaded(record.model_key, reason=reason)
            print(f"[inference-residency] 标记清理: reason={reason}, model_key={record.model_key}")
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

    def _process_audio_input(self, request: InferenceRequest) -> tuple[str, bool]:
        """处理参考音频输入。"""
        if request.ref_audio_base64:
            try:
                audio_data = base64.b64decode(request.ref_audio_base64)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_file.write(audio_data)
                    return temp_file.name, True
            except Exception as exc:
                raise ValueError(f"Base64 音频解码失败: {exc}")

        if request.ref_audio_path:
            if not os.path.exists(request.ref_audio_path):
                raise FileNotFoundError(f"参考音频文件不存在: {request.ref_audio_path}")
            return request.ref_audio_path, False

        raise ValueError("必须提供参考音频(ref_audio_path 或 ref_audio_base64)")

    def _preprocess_text(self, text: str, how_to_cut: str) -> List[str]:
        """使用 GPT-SoVITS 官方切分方法生成响应展示分段。"""
        text = text.strip()
        if not text:
            return []

        method_name = self.CUT_METHODS.get(how_to_cut, "cut0")
        if not self._get_text_split_method:
            return [text]

        try:
            method = self._get_text_split_method(method_name)
            split_text = method(text)
        except Exception as exc:
            print(f"文本切分失败，回退为不切分: {exc}")
            return [text]

        return [item.strip() for item in str(split_text).splitlines() if item.strip()]

    def _build_tts_inputs(self, request: InferenceRequest, ref_audio_path: str) -> Dict[str, Any]:
        """将服务请求映射为官方 TTS 输入。"""
        prompt_text = request.prompt_text or ""
        prompt_lang = request.prompt_language
        if request.config.ref_free:
            prompt_text = ""

        return {
            "text": request.text,
            "text_lang": request.text_language,
            "ref_audio_path": ref_audio_path,
            "aux_ref_audio_paths": request.aux_ref_audio_paths or [],
            "prompt_text": prompt_text,
            "prompt_lang": prompt_lang,
            "top_k": request.config.top_k,
            "top_p": request.config.top_p,
            "temperature": request.config.temperature,
            "text_split_method": self.CUT_METHODS.get(request.config.how_to_cut, "cut0"),
            "batch_size": int(request.config.batch_size),
            "batch_threshold": float(request.config.batch_threshold),
            "speed_factor": float(request.config.speed),
            "split_bucket": request.config.split_bucket,
            "return_fragment": request.config.return_fragment,
            "fragment_interval": float(request.config.fragment_interval),
            "seed": int(request.config.seed),
            "parallel_infer": request.config.parallel_infer,
            "repetition_penalty": float(request.config.repetition_penalty),
            "sample_steps": int(request.config.sample_steps),
            "super_sampling": request.config.if_sr or request.config.super_sampling,
            "streaming_mode": request.config.streaming_mode,
        }

    def _run_tts(self, inputs: Dict[str, Any]) -> tuple[int, np.ndarray]:
        """执行官方 TTS 管线并收集输出。"""
        audio_fragments: List[np.ndarray] = []
        sample_rate = 16000

        for current_sr, audio in self.tts_pipeline.run(inputs):
            sample_rate = current_sr
            if audio is None or len(audio) == 0:
                continue
            audio_fragments.append(np.asarray(audio))

        if not audio_fragments:
            raise RuntimeError("推理未生成音频数据")

        if len(audio_fragments) == 1:
            return sample_rate, audio_fragments[0]

        return sample_rate, np.concatenate(audio_fragments, axis=0)

    def _encode_audio_to_base64(self, audio_data: np.ndarray, sample_rate: int, format: str = "wav") -> str:
        """将音频编码为 Base64。"""
        buffer = io.BytesIO()
        if format.lower() == "wav":
            sf.write(buffer, audio_data, sample_rate, format="WAV")
        else:
            sf.write(buffer, audio_data, sample_rate, format="WAV")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    def _create_temp_output_path(self, format: str = "wav") -> str:
        fd, path = tempfile.mkstemp(prefix="gpt_sovits_infer_", suffix=f".{format}")
        os.close(fd)
        return path

    def _save_audio_file(self, audio_data: np.ndarray, sample_rate: int, output_path: str, format: str = "wav") -> str:
        """保存音频文件。"""
        if format.lower() == "wav":
            sf.write(output_path, audio_data, sample_rate, format="WAV")
        else:
            sf.write(output_path, audio_data, sample_rate, format="WAV")
        return output_path

    async def inference(self, request: InferenceRequest) -> InferenceResponse:
        """执行推理。"""
        start_time = datetime.now()
        temp_audio_path = None
        temp_created = False
        model_key = self.get_current_model_key()

        try:
            if self.tts_pipeline is None:
                return InferenceResponse(success=False, message="模型未加载，请先加载 GPT 和 SoVITS 模型")

            self.residency_manager.begin_request(model_key)
            print(f"[inference-residency] 开始推理: model_key={model_key}")
            temp_audio_path, temp_created = self._process_audio_input(request)
            text_segments = self._preprocess_text(request.text, request.config.how_to_cut)
            inputs = self._build_tts_inputs(request, temp_audio_path)
            sample_rate, audio_data = self._run_tts(inputs)

            processing_time = (datetime.now() - start_time).total_seconds()
            duration = len(audio_data) / sample_rate if sample_rate else 0.0
            response = InferenceResponse(
                success=True,
                message="推理完成",
                sample_rate=sample_rate,
                duration=duration,
                processing_time=processing_time,
                text_segments=text_segments,
            )

            if request.return_base64:
                response.audio_data = self._encode_audio_to_base64(audio_data, sample_rate, request.output_format)
            else:
                output_path = self._create_temp_output_path(request.output_format)
                response.audio_path = self._save_audio_file(audio_data, sample_rate, output_path, request.output_format)

            return response
        except Exception as exc:
            return InferenceResponse(
                success=False,
                message=f"推理失败: {exc}",
                error_details=traceback.format_exc(),
                processing_time=(datetime.now() - start_time).total_seconds(),
            )
        finally:
            self.residency_manager.end_request(model_key)
            if model_key:
                print(f"[inference-residency] 结束推理: model_key={model_key}")
            self.cleanup_resident_models()
            if temp_created and temp_audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

    def inference_sync(self, request: InferenceRequest) -> InferenceResponse:
        """同步推理入口。"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.inference(request))
        finally:
            loop.close()

    def get_model_info(self) -> Dict[str, Any]:
        """获取当前模型信息。"""
        languages = self.DEFAULT_LANGUAGES
        if self.tts_pipeline is not None:
            languages = list(self.tts_pipeline.configs.languages)
        return {
            "gpt_path": self.current_gpt_path,
            "sovits_path": self.current_sovits_path,
            "model_version": self.model_version,
            "device": str(self.device),
            "is_half": self.is_half,
            "models_loaded": self.tts_pipeline is not None,
            "supported_languages": languages,
            "residency": self.residency_manager.get_status(),
        }

    def clear_cache(self):
        """清理推理缓存。"""
        if self.tts_pipeline is not None:
            self.tts_pipeline.prompt_cache = {
                "ref_audio_path": None,
                "prompt_semantic": None,
                "refer_spec": [],
                "prompt_text": None,
                "prompt_lang": None,
                "phones": None,
                "bert_features": None,
                "norm_text": None,
                "aux_ref_audio_paths": [],
            }
            self.tts_pipeline.stop_flag = False
        self.residency_manager.mark_used(self.get_current_model_key())
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_supported_languages(self) -> List[str]:
        """获取支持的语言列表。"""
        if self.tts_pipeline is not None:
            return list(self.tts_pipeline.configs.languages)
        return self.DEFAULT_LANGUAGES

    def get_supported_formats(self) -> List[str]:
        """获取支持的音频格式。"""
        return ["wav", "mp3", "ogg", "aac"]
