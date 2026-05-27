#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 统一网关服务

将基础封装层聚合到一个端口，对外提供统一入口。
"""

import os
import sys
import asyncio
import importlib.util
import math
import mimetypes
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Body, WebSocket
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from Code.FastApi.Base.Hub.monhub_bridge import create_monhub_bridge_from_env
from Code.FastApi.Base.monconfig import MonConfig
from Code.FastApi.Domain.Role.Services import RoleService
from Code.FastApi.Domain.Routers import build_domain_router

gateway_dir = Path(__file__).resolve().parent
base_dir = gateway_dir.parent
sys.path.insert(0, str(base_dir))


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"true", "yes", "1", "on", "enabled"}


class UnifiedGatewayConfig(BaseModel):
    """统一网关配置。"""
    enable_auth: bool = False
    api_key: Optional[str] = None
    max_concurrent_jobs: int = 10
    temp_dir: str = "temp"
    log_level: str = "INFO"


def _build_runtime_config() -> UnifiedGatewayConfig:
    return UnifiedGatewayConfig(
        enable_auth=_env_bool("ENABLE_AUTH", False),
        api_key=os.environ.get("API_KEY"),
        max_concurrent_jobs=int(os.environ.get("MAX_CONCURRENT_JOBS", "10")),
        temp_dir=os.environ.get("TEMP_DIR", "temp"),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
    )


def _save_uploaded_audio_files(input_audio_dir: str, audio_files: Optional[List[UploadFile]]) -> list[str]:
    """将训练请求中携带的音频文件保存到输入目录。"""
    if not audio_files:
        return []

    target_dir = Path(input_audio_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[str] = []
    for upload in audio_files:
        if upload is None:
            continue
        filename = Path(upload.filename or "uploaded_audio.bin").name
        if not filename:
            filename = "uploaded_audio.bin"
        target_path = target_dir / filename
        with target_path.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle)
        saved_files.append(str(target_path))
    return saved_files


def _build_asr_config_for_language(language: str):
    """训练/数据准备 ASR 根据语言自动选择识别引擎。"""
    from Code.FastApi.Base.DataPreparation.asr_recognition.service import ASRConfig

    normalized_language = (language or "zh").strip().lower()
    if normalized_language in {"zh", "yue"}:
        return ASRConfig(
            model_type="funasr",
            model_size="large",
            language=normalized_language,
            precision="float32",
        )
    return ASRConfig(
        model_type="faster_whisper",
        model_size="large-v3",
        language=normalized_language if normalized_language else "auto",
        precision="float16",
    )


class BatchProjectItem(BaseModel):
    """批量项目定义。"""
    name: str
    input_dir: str
    output_dir: str
    language: str = "zh"
    version: str = "v2Pro"


class BatchProjectsRequest(BaseModel):
    """批量处理请求。"""
    projects: List[BatchProjectItem]


class TrainingLaunchSummary(BaseModel):
    training_type: str
    success: bool
    message: str
    job_id: Optional[str] = None
    config_file: Optional[str] = None
    log_dir: Optional[str] = None
    model_dir: Optional[str] = None


class TrainingWorkflowOptions(BaseModel):
    start_training: bool = False
    train_gpt: bool = True
    train_sovits: bool = True
    gpt_batch_size: int = 8
    gpt_total_epoch: int = 15
    sovits_batch_size: int = 32
    sovits_total_epoch: int = 8
    training_order: str = "sovits_first"


class RoleEmotionSynthesisRequest(BaseModel):
    """业务合成请求：前端只提交角色、情感和文本。"""

    role_id: int
    emotion: str
    text: str
    text_language: str = "zh"
    world_id: Optional[int] = None
    version: Optional[str] = None
    how_to_cut: str = "凑四句一切"
    top_k: int = 20
    top_p: float = 0.6
    temperature: float = 0.6
    speed: float = 1.0
    sample_steps: int = 8
    if_sr: bool = False
    ref_free: bool = False
    if_freeze: bool = False
    pause_second: float = 0.3
    use_cuda_graph: bool = False
    return_base64: bool = True


def _resources_root() -> Path:
    config = MonConfig(start_path=Path(__file__).resolve())
    workspace_root = config.workspace_root() or Path.cwd()
    root = (workspace_root / "Resources").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _normalize_path_segment(value: str) -> str:
    normalized = (value or "").strip()
    if not normalized:
        raise ValueError("路径段不能为空")
    invalid = set('\\/:*?"<>|')
    if any(ch in invalid for ch in normalized) or normalized in {".", ".."}:
        raise ValueError(f"路径段包含非法字符: {value}")
    return normalized


def _sync_directory_files(source_dir: Path, target_dir: Path) -> list[str]:
    if not source_dir.exists() or not source_dir.is_dir():
        return []

    target_dir.mkdir(parents=True, exist_ok=True)
    for child in target_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    copied_files: list[str] = []
    for source_path in sorted(source_dir.iterdir(), key=lambda item: item.name.lower()):
        if not source_path.is_file():
            continue
        target_path = target_dir / source_path.name
        shutil.copy2(source_path, target_path)
        copied_files.append(str(target_path))
    return copied_files


def _resource_layout(
    project_name: str,
    version: str,
    world_name: str = "Standalone",
    experiment_name: str = "",
) -> Dict[str, str]:
    role_name = _normalize_path_segment(project_name)
    world = _normalize_path_segment(world_name or "Standalone")
    base_version = _normalize_path_segment(version or "v2Pro")

    resources_root = _resources_root()
    model_root = resources_root / "Model" / world / role_name / base_version
    train_root = resources_root / "Train" / "Projects" / world / role_name / base_version
    dataset_root = train_root / "dataset"

    return {
        "role_name": role_name,
        "world_name": world,
        "base_version": base_version,
        "experiment_name": "",
        "model_root": str(model_root),
        "train_root": str(train_root),
        "train_root_parent": str(train_root.parent),
        "dataset_root": str(dataset_root),
        "model_sliced_dir": str(model_root / "sliced"),
        "gpt_model_dir": str(model_root / "GPT"),
        "sovits_model_dir": str(model_root / "SoVITS"),
    }


class ServiceManager:
    """动态加载基础封装服务。"""

    def __init__(self):
        self.services: Dict[str, Dict[str, Any]] = {}
        self.service_configs: Dict[str, Dict[str, str]] = {}
        self.load_errors: Dict[str, str] = {}
        self.load_all_services()

    def load_all_services(self):
        service_configs = [
            {
                "name": "audio_slice",
                "path": "DataPreparation/audio_slice/service.py",
                "class": "AudioSliceService",
                "prefix": "/data-prep/audio-slice",
            },
            {
                "name": "asr_recognition",
                "path": "DataPreparation/asr_recognition/service.py",
                "class": "ASRRecognitionService",
                "prefix": "/data-prep/asr",
            },
            {
                "name": "text_processing",
                "path": "DatasetFormatting/text_processing/service.py",
                "class": "TextProcessingService",
                "prefix": "/dataset/text",
            },
            {
                "name": "audio_features",
                "path": "DatasetFormatting/audio_features/service.py",
                "class": "AudioFeaturesService",
                "prefix": "/dataset/audio",
            },
            {
                "name": "semantic_encoding",
                "path": "DatasetFormatting/semantic_encoding/service.py",
                "class": "SemanticEncodingService",
                "prefix": "/dataset/semantic",
            },
            {
                "name": "gpt_training",
                "path": "Training/gpt_training/service.py",
                "class": "GPTTrainingService",
                "prefix": "/training/gpt",
            },
            {
                "name": "sovits_training",
                "path": "Training/sovits_training/service.py",
                "class": "SoVITSTrainingService",
                "prefix": "/training/sovits",
            },
            {
                "name": "inference",
                "path": "Inference/service.py",
                "class": "InferenceService",
                "prefix": "/inference",
            },
        ]

        for config in service_configs:
            self.service_configs[config["name"]] = config
            try:
                self.load_service(config)
                self.load_errors.pop(config["name"], None)
                print(f"成功加载服务: {config['name']}")
            except Exception as exc:
                self.load_errors[config["name"]] = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                print(f"加载服务失败 {config['name']}: {exc}")

    def load_service(self, config: Dict[str, str]):
        module_path = base_dir / config["path"]
        if not module_path.exists():
            raise FileNotFoundError(f"模块文件不存在: {module_path}")

        relative_module = config["path"][:-3].replace("/", ".").replace("\\", ".")
        module_name = f"Code.FastApi.Base.{relative_module}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        service_class = getattr(module, config["class"])
        instance = service_class()

        self.services[config["name"]] = {
            "instance": instance,
            "prefix": config["prefix"],
            "module": module,
        }
        self.load_errors.pop(config["name"], None)

    def get_service(self, name: str):
        return self.services.get(name, {}).get("instance")

    def reload_service(self, name: str):
        config = self.service_configs.get(name)
        if not config:
            raise KeyError(f"服务不存在: {name}")
        self.load_service(config)


app = FastAPI(
    title="GPT-SoVITS 统一网关 API",
    description="GPT-SoVITS 完整流程的统一 API 服务",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(build_domain_router())

service_manager = ServiceManager()
config = _build_runtime_config()
security = HTTPBearer(auto_error=False)
monhub_bridge = create_monhub_bridge_from_env()


@app.on_event("startup")
async def start_monhub_bridge():
    """Register the gateway in MonHub when enabled."""
    monhub_bridge.start()


@app.on_event("shutdown")
async def stop_monhub_bridge():
    """Unregister the gateway from MonHub on shutdown."""
    monhub_bridge.stop()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """可选的 API 认证。"""
    if config.enable_auth:
        if not credentials or credentials.credentials != config.api_key:
            raise HTTPException(status_code=401, detail="无效的API密钥")
    return credentials


def _ensure_service(name: str):
    service = service_manager.get_service(name)
    if not service:
        raise HTTPException(status_code=503, detail=f"服务不可用: {name}")
    return service


class WorkflowAbortError(RuntimeError):
    """Raised when a workflow must stop before launching training."""


def _service_unavailable_message(name: str) -> str:
    load_error = getattr(service_manager, "load_errors", {}).get(name)
    if load_error:
        return f"服务不可用: {name}，加载失败: {load_error}"
    return f"服务不可用: {name}"


def _stop_workflow(message: str) -> None:
    print(f"[workflow] 停止执行: {message}")
    raise WorkflowAbortError(message)


def _require_workflow_service(name: str):
    service = service_manager.get_service(name)
    if not service:
        _stop_workflow(_service_unavailable_message(name))
    return service


def _step_result_success(result: Any) -> bool:
    if result is None:
        return False
    if isinstance(result, dict):
        return bool(result.get("success", False))
    if hasattr(result, "success"):
        return bool(getattr(result, "success"))
    return False


def _step_result_message(result: Any) -> str:
    payload = _to_payload(result)
    if isinstance(payload, dict):
        for key in ("message", "error", "detail", "Exception"):
            value = payload.get(key)
            if value:
                return str(value)
    return str(payload)


def _require_step_success(step_name: str, result: Any) -> None:
    if _step_result_success(result):
        return
    _stop_workflow(f"{step_name} 失败: {_step_result_message(result)}")


def _validate_training_dataset(project_root: str, version: str, require_gpt: bool, require_sovits: bool) -> dict[str, Any]:
    dataset_root = Path(project_root) / "dataset"
    required_paths: list[Path] = []

    if require_gpt:
        required_paths.extend([
            dataset_root / "2-name2text.txt",
            dataset_root / "6-name2semantic.tsv",
        ])

    if require_sovits:
        required_paths.extend([
            dataset_root / "2-name2text.txt",
            dataset_root / "4-cnhubert",
            dataset_root / "5-wav32k",
        ])
        if version in {"v2Pro", "v2ProPlus"}:
            required_paths.append(dataset_root / "7-sv_cn")

    missing = [str(path) for path in dict.fromkeys(required_paths) if not path.exists()]
    if missing:
        _stop_workflow("训练前置数据缺失: " + "; ".join(missing))

    stats: dict[str, Any] = {"dataset_root": str(dataset_root)}
    text_file = dataset_root / "2-name2text.txt"
    semantic_file = dataset_root / "6-name2semantic.tsv"
    if text_file.exists():
        stats["text_lines"] = len(text_file.read_text(encoding="utf-8", errors="ignore").splitlines())
    if semantic_file.exists():
        stats["semantic_lines"] = max(0, len(semantic_file.read_text(encoding="utf-8", errors="ignore").splitlines()) - 1)
    for dirname in ("4-cnhubert", "5-wav32k", "7-sv_cn"):
        directory = dataset_root / dirname
        if directory.exists():
            stats[f"{dirname}_files"] = sum(1 for item in directory.iterdir() if item.is_file())

    print(f"[workflow] 训练前置数据检查通过: {stats}")
    return stats


def _project_root(output_dir: str, project_name: str) -> str:
    return os.path.join(output_dir, project_name)


def _to_payload(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if hasattr(value, "model_dump"):
        return _to_payload(value.model_dump())
    if isinstance(value, list):
        return [_to_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_payload(item) for key, item in value.items()}
    return value


@app.get("/")
async def root():
    """根路径。"""
    return {
        "service": "GPT-SoVITS 统一网关",
        "version": "2.1.0",
        "status": "running",
        "available_services": list(service_manager.services.keys()),
        "documentation": "/docs",
    }


@app.get("/health")
async def health_check():
    """健康检查。"""
    service_status = {}
    for name, service_info in service_manager.services.items():
        try:
            instance = service_info["instance"]
            if hasattr(instance, "health_check"):
                status = await instance.health_check()
            else:
                status = {"status": "available"}
            service_status[name] = status
        except Exception as exc:
            service_status[name] = {"status": "error", "error": str(exc)}
    for name, error in service_manager.load_errors.items():
        service_status[name] = {
            "status": "load_error",
            "error": error,
            "prefix": service_manager.service_configs.get(name, {}).get("prefix"),
        }

    return {
        "gateway_status": "healthy",
        "services": service_status,
        "total_services": len(service_manager.service_configs),
        "healthy_services": sum(1 for item in service_status.values() if item.get("status") not in {"error", "load_error"}),
        "monhub": monhub_bridge.status(),
    }


@app.get("/monhub/status")
async def get_monhub_status():
    """获取 MonHub 注册桥接状态。"""
    return monhub_bridge.status()


@app.post("/data-prep/audio-slice/process")
async def audio_slice_process(
    input_path: str = Form(...),
    output_dir: str = Form(...),
    threshold: float = Form(default=-34.0),
    min_length: int = Form(default=4000),
    user: Any = Depends(get_current_user),
):
    """音频切分处理。"""
    service = _ensure_service("audio_slice")

    try:
        from Code.FastApi.Base.DataPreparation.audio_slice.service import SliceRequest, SliceConfig

        request = SliceRequest(
            input_path=input_path,
            output_dir=output_dir,
            config=SliceConfig(threshold=threshold, min_length=min_length),
        )
        return await service.process(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"音频切分失败: {exc}")


@app.post("/data-prep/asr/recognize")
async def asr_recognize(
    audio_dir: str = Form(default=""),
    output_file: str = Form(...),
    language: str = Form(default="zh"),
    audio_file: Optional[UploadFile] = File(default=None),
    user: Any = Depends(get_current_user),
):
    """ASR 语音识别。"""
    service = _ensure_service("asr_recognition")

    temp_audio_path: Optional[str] = None
    temp_audio_dir: Optional[str] = None
    try:
        from Code.FastApi.Base.DataPreparation.asr_recognition.service import ASRRequest

        input_path = audio_dir.strip() if audio_dir else ""
        if audio_file is not None:
            suffix = Path(audio_file.filename or "uploaded_audio.wav").suffix or ".wav"
            temp_audio_dir = tempfile.mkdtemp(prefix="asr_upload_")
            original_name = Path(audio_file.filename or f"uploaded_audio{suffix}").name
            temp_audio_path = os.path.join(temp_audio_dir, original_name)
            with open(temp_audio_path, "wb") as temp_file:
                temp_file.write(await audio_file.read())
            input_path = temp_audio_dir

        if not input_path:
            raise ValueError("请提供音频路径或上传音频文件")

        output_path = Path(output_file)
        output_dir = str(output_path.parent) if output_path.suffix else output_file
        request = ASRRequest(
            input_path=input_path,
            output_dir=output_dir,
            config=_build_asr_config_for_language(language),
        )
        return await service.process(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ASR识别失败: {exc}")
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except OSError:
                pass
        if temp_audio_dir and os.path.exists(temp_audio_dir):
            try:
                os.rmdir(temp_audio_dir)
            except OSError:
                pass


@app.post("/inference/transcribe")
async def inference_transcribe(
    audio_file: UploadFile | None = File(default=None),
    audio_path: str = Form(default=""),
    language: str = Form(default="zh"),
    model_type: str = Form(default="funasr"),
    model_size: str = Form(default="large"),
    precision: str = Form(default="float32"),
    user: Any = Depends(get_current_user),
):
    """面向前端单文件转录的轻量接口。"""
    service = _ensure_service("asr_recognition")

    upload_temp_dir: Optional[str] = None
    output_temp_dir: Optional[str] = None
    try:
        from Code.FastApi.Base.DataPreparation.asr_recognition.service import ASRRequest, ASRConfig

        output_temp_dir = tempfile.mkdtemp(prefix="transcribe_output_")
        input_path = ""

        if audio_file is not None:
            suffix = Path(audio_file.filename or "uploaded_audio.wav").suffix or ".wav"
            upload_temp_dir = tempfile.mkdtemp(prefix="transcribe_upload_")
            original_name = Path(audio_file.filename or f"uploaded_audio{suffix}").name
            upload_path = os.path.join(upload_temp_dir, original_name)
            with open(upload_path, "wb") as temp_file:
                temp_file.write(await audio_file.read())
            input_path = upload_path
        elif audio_path.strip():
            resources_root = _resources_root()
            model_root = (resources_root / "Model").resolve()
            target_path = Path(audio_path).expanduser().resolve()
            allowed_suffixes = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}

            if not target_path.exists() or not target_path.is_file():
                raise HTTPException(status_code=404, detail="待转录音频不存在")
            if target_path.suffix.lower() not in allowed_suffixes:
                raise HTTPException(status_code=400, detail="待转录音频格式不受支持")
            if not (target_path.parent == model_root or model_root in target_path.parents):
                raise HTTPException(status_code=400, detail="待转录音频路径不在允许的资源目录范围内")

            input_path = str(target_path)
        else:
            raise HTTPException(status_code=400, detail="请上传音频文件或提供音频路径")

        request = ASRRequest(
            input_path=input_path,
            output_dir=output_temp_dir,
            config=ASRConfig(
                language=language,
                model_type=model_type,
                model_size=model_size,
                precision=precision,
            ),
        )
        result = await service.process(request)
        if not result.success:
            raise HTTPException(status_code=500, detail=result.message)

        recognition_results = getattr(result, "recognition_results", []) or []
        text_parts = [item.get("text", "").strip() for item in recognition_results if item.get("text")]
        combined_text = " ".join(part for part in text_parts if part).strip()
        resolved_language = (
            recognition_results[0].get("language")
            if recognition_results and recognition_results[0].get("language")
            else language
        )

        return {
            "success": True,
            "message": result.message,
            "text": combined_text,
            "language": resolved_language,
            "segments": recognition_results,
            "processing_time": getattr(result, "processing_time", 0.0),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"转录失败: {exc}")
    finally:
        if upload_temp_dir and os.path.exists(upload_temp_dir):
            shutil.rmtree(upload_temp_dir, ignore_errors=True)
        if output_temp_dir and os.path.exists(output_temp_dir):
            shutil.rmtree(output_temp_dir, ignore_errors=True)


@app.post("/inference/transcribe/models/load")
async def load_transcribe_models(
    model_type: str = Form(default="funasr"),
    model_size: str = Form(default="large"),
    language: str = Form(default="zh"),
    precision: str = Form(default="float32"),
    user: Any = Depends(get_current_user),
):
    """预加载一套ASR模型。"""
    service = _ensure_service("asr_recognition")

    from Code.FastApi.Base.DataPreparation.asr_recognition.service import ASRConfig

    config = ASRConfig(
        model_type=model_type,
        model_size=model_size,
        language=language,
        precision=precision,
    )
    if not service.load_models(config):
        raise HTTPException(status_code=400, detail="ASR模型加载失败")
    return {
        "success": True,
        "message": "ASR模型加载成功",
        "model_info": service.get_model_info(),
    }


@app.get("/inference/transcribe/models/info")
async def get_transcribe_models_info(
    user: Any = Depends(get_current_user),
):
    """获取当前ASR模型信息。"""
    service = _ensure_service("asr_recognition")
    return service.get_model_info()


@app.post("/inference/transcribe/models/unload")
async def unload_transcribe_models(
    user: Any = Depends(get_current_user),
):
    """手动卸载当前ASR模型。"""
    service = _ensure_service("asr_recognition")
    unloaded = service.unload_models(reason="manual")
    return {
        "success": True,
        "message": "ASR模型已卸载" if unloaded else "当前没有已加载的ASR模型",
        "unloaded": unloaded,
    }


@app.post("/inference/transcribe/models/cleanup")
async def cleanup_transcribe_models(
    force: bool = Form(default=False),
    user: Any = Depends(get_current_user),
):
    """执行一次ASR驻留清理。"""
    service = _ensure_service("asr_recognition")
    return service.cleanup_resident_models(force=force)


@app.post("/dataset/text/extract")
async def text_extract_features(
    list_file: str = Form(...),
    input_wav_dir: str = Form(...),
    experiment_name: str = Form(default="default"),
    output_dir: str = Form(...),
    user: Any = Depends(get_current_user),
):
    """文本特征提取。"""
    service = _ensure_service("text_processing")

    try:
        from Code.FastApi.Base.DatasetFormatting.text_processing.service import TextProcessingRequest, TextProcessingConfig

        request = TextProcessingRequest(
            input_text_file=list_file,
            input_wav_dir=input_wav_dir,
            experiment_name=experiment_name,
            output_dir=output_dir,
            config=TextProcessingConfig(),
        )
        return await service.process(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"文本特征提取失败: {exc}")


@app.post("/dataset/audio/extract")
async def audio_extract_features(
    list_file: str = Form(...),
    input_wav_dir: str = Form(...),
    experiment_name: str = Form(default="default"),
    output_dir: str = Form(...),
    version: str = Form(default="v2Pro"),
    user: Any = Depends(get_current_user),
):
    """音频特征提取。"""
    service = _ensure_service("audio_features")

    try:
        from Code.FastApi.Base.DatasetFormatting.audio_features.service import AudioFeaturesRequest, AudioFeaturesConfig

        request = AudioFeaturesRequest(
            input_text_file=list_file,
            input_wav_dir=input_wav_dir,
            experiment_name=experiment_name,
            output_dir=output_dir,
            config=AudioFeaturesConfig(version=version),
        )
        return await service.process(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"音频特征提取失败: {exc}")


@app.post("/dataset/semantic/encode")
async def semantic_encode(
    list_file: str = Form(...),
    cnhubert_dir: str = Form(default=""),
    experiment_name: str = Form(default="default"),
    output_dir: str = Form(...),
    version: str = Form(default="v2Pro"),
    user: Any = Depends(get_current_user),
):
    """语义编码。"""
    service = _ensure_service("semantic_encoding")

    try:
        from Code.FastApi.Base.DatasetFormatting.semantic_encoding.service import SemanticEncodingRequest, SemanticEncodingConfig

        resolved_cnhubert_dir = cnhubert_dir.strip() if cnhubert_dir else ""
        if not resolved_cnhubert_dir:
            resolved_cnhubert_dir = os.path.join(output_dir, "4-cnhubert")

        request = SemanticEncodingRequest(
            input_text_file=list_file,
            cnhubert_dir=resolved_cnhubert_dir,
            experiment_name=experiment_name,
            output_dir=output_dir,
            config=SemanticEncodingConfig(version=version),
        )
        return await service.process(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"语义编码失败: {exc}")


@app.post("/training/gpt/start")
async def start_gpt_training(
    exp_name: str = Form(...),
    exp_root: str = Form(default=""),
    workspace_dir: str = Form(default=""),
    model_output_dir: str = Form(default=""),
    version: str = Form(default="v2Pro"),
    batch_size: int = Form(default=8),
    total_epoch: int = Form(default=15),
    user: Any = Depends(get_current_user),
):
    """开始 GPT 训练。"""
    service = _ensure_service("gpt_training")

    try:
        from Code.FastApi.Base.Training.gpt_training.service import GPTTrainingRequest, GPTTrainingConfig
        if not workspace_dir.strip() and not exp_root.strip():
            raise ValueError("exp_root 和 workspace_dir 至少提供一个")

        request = GPTTrainingRequest(
            exp_name=exp_name,
            exp_root=exp_root,
            workspace_dir=workspace_dir,
            model_output_dir=model_output_dir,
            config=GPTTrainingConfig(
                version=version,
                batch_size=batch_size,
                total_epoch=total_epoch,
                save_every_epoch=total_epoch,
            ),
        )
        return await service.start_training(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"GPT训练启动失败: {exc}")


@app.post("/training/sovits/start")
async def start_sovits_training(
    exp_name: str = Form(...),
    exp_root: str = Form(default=""),
    workspace_dir: str = Form(default=""),
    model_output_dir: str = Form(default=""),
    version: str = Form(default="v2Pro"),
    batch_size: int = Form(default=32),
    total_epoch: int = Form(default=8),
    user: Any = Depends(get_current_user),
):
    """开始 SoVITS 训练。"""
    service = _ensure_service("sovits_training")

    try:
        from Code.FastApi.Base.Training.sovits_training.service import SoVITSTrainingRequest, SoVITSTrainingConfig
        if not workspace_dir.strip() and not exp_root.strip():
            raise ValueError("exp_root 和 workspace_dir 至少提供一个")

        request = SoVITSTrainingRequest(
            exp_name=exp_name,
            exp_root=exp_root,
            workspace_dir=workspace_dir,
            model_output_dir=model_output_dir,
            config=SoVITSTrainingConfig(
                version=version,
                batch_size=batch_size,
                total_epoch=total_epoch,
                save_every_epoch=total_epoch,
            ),
        )
        return await service.start_training(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SoVITS训练启动失败: {exc}")


@app.get("/training/status/{job_id}")
async def get_training_status(
    job_id: str,
    user: Any = Depends(get_current_user),
):
    """获取训练状态。"""
    gpt_service = service_manager.get_service("gpt_training")
    if gpt_service and hasattr(gpt_service, "get_training_status"):
        status = gpt_service.get_training_status(job_id)
        if status:
            return {"type": "gpt", "status": status}

    sovits_service = service_manager.get_service("sovits_training")
    if sovits_service and hasattr(sovits_service, "get_training_status"):
        status = sovits_service.get_training_status(job_id)
        if status:
            return {"type": "sovits", "status": status}

    raise HTTPException(status_code=404, detail=f"训练任务不存在: {job_id}")


@app.post("/training/stop/{job_id}")
async def stop_training(
    job_id: str,
    user: Any = Depends(get_current_user),
):
    """停止指定训练任务。"""
    gpt_service = service_manager.get_service("gpt_training")
    if gpt_service and hasattr(gpt_service, "stop_training"):
        if gpt_service.stop_training(job_id):
            return {"success": True, "type": "gpt", "job_id": job_id, "message": "GPT训练已停止"}

    sovits_service = service_manager.get_service("sovits_training")
    if sovits_service and hasattr(sovits_service, "stop_training"):
        if sovits_service.stop_training(job_id):
            return {"success": True, "type": "sovits", "job_id": job_id, "message": "SoVITS训练已停止"}

    raise HTTPException(status_code=404, detail=f"训练任务不存在或无法停止: {job_id}")


@app.post("/inference/tts")
async def text_to_speech(
    text: str = Form(...),
    text_language: str = Form(default="zh"),
    ref_audio: UploadFile | None = File(default=None),
    ref_audio_path: str = Form(default=""),
    prompt_text: str = Form(default=""),
    prompt_language: str = Form(default="zh"),
    how_to_cut: str = Form(default="凑四句一切"),
    top_k: int = Form(default=20),
    top_p: float = Form(default=0.6),
    temperature: float = Form(default=0.6),
    speed: float = Form(default=1.0),
    sample_steps: int = Form(default=8),
    if_sr: bool = Form(default=False),
    ref_free: bool = Form(default=False),
    if_freeze: bool = Form(default=False),
    pause_second: float = Form(default=0.3),
    use_cuda_graph: bool = Form(default=False),
    return_base64: bool = Form(default=True),
    user: Any = Depends(get_current_user),
):
    """文本转语音。"""
    service = _ensure_service("inference")

    try:
        import tempfile
        from Code.FastApi.Base.Inference.service import InferenceRequest, InferenceConfig

        temp_audio_path = ""
        created_temp = False
        if ref_audio is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                content = await ref_audio.read()
                temp_file.write(content)
                temp_audio_path = temp_file.name
                created_temp = True
        elif ref_audio_path.strip():
            temp_audio_path = ref_audio_path.strip()
        else:
            raise ValueError("请上传参考音频或提供参考音频路径")

        try:
            request = InferenceRequest(
                text=text,
                text_language=text_language,
                ref_audio_path=temp_audio_path,
                prompt_text=prompt_text,
                prompt_language=prompt_language,
                config=InferenceConfig(
                    how_to_cut=how_to_cut,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    speed=speed,
                    sample_steps=sample_steps,
                    if_sr=if_sr,
                    ref_free=ref_free,
                    if_freeze=if_freeze,
                    pause_second=pause_second,
                    use_cuda_graph=use_cuda_graph,
                ),
                return_base64=return_base64,
            )
            return await service.inference(request)
        finally:
            if created_temp and temp_audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"语音合成失败: {exc}")


@app.post("/api/synthesis/role-emotion")
async def synthesize_role_emotion(
    payload: RoleEmotionSynthesisRequest,
    user: Any = Depends(get_current_user),
):
    """按角色与情感合成语音。

    这是面向前端/业务系统的接口；模型路径、参考音频、参考文本由后端解析。
    """
    service = _ensure_service("inference")
    role_service = RoleService()

    try:
        from Code.FastApi.Base.Inference.service import InferenceRequest, InferenceConfig

        if not payload.text.strip():
            raise ValueError("请输入要合成的文本")
        if not payload.emotion.strip():
            raise ValueError("请选择情感")

        role = role_service.get_role(payload.role_id)
        if payload.world_id is not None and role.world_id != payload.world_id:
            raise ValueError("角色不属于当前世界")
        if payload.version and role.version != payload.version:
            raise ValueError("角色不属于当前版本")
        if not role.gpt_model_path or not role.sov_model_path:
            raise ValueError("当前角色缺少 GPT 或 SoVITS 模型路径")

        emotions = role_service.list_role_emotions(payload.role_id)
        selected_emotion = next(
            (
                item for item in emotions
                if str(item.get("name", "")).strip() == payload.emotion.strip()
            ),
            None,
        )
        if selected_emotion is None:
            raise ValueError(f"当前角色没有情感配置: {payload.emotion}")

        ref_audio_path = str(selected_emotion.get("music_url") or "").strip()
        if not ref_audio_path:
            raise ValueError("当前情感缺少参考音频")
        prompt_text = str(selected_emotion.get("text") or role.prompt_text or "")
        prompt_language = str(selected_emotion.get("text_language") or role.language or payload.text_language)

        model_info = service.get_model_info() if hasattr(service, "get_model_info") else {}
        models_loaded = bool(
            model_info.get("models_loaded")
            and model_info.get("gpt_path") == role.gpt_model_path
            and model_info.get("sovits_path") == role.sov_model_path
        )
        if not models_loaded and not service.load_models(role.gpt_model_path, role.sov_model_path):
            raise ValueError("模型加载失败")

        request = InferenceRequest(
            text=payload.text.strip(),
            text_language=payload.text_language,
            ref_audio_path=ref_audio_path,
            prompt_text=prompt_text,
            prompt_language=prompt_language,
            config=InferenceConfig(
                how_to_cut=payload.how_to_cut,
                top_k=payload.top_k,
                top_p=payload.top_p,
                temperature=payload.temperature,
                speed=payload.speed,
                sample_steps=payload.sample_steps,
                if_sr=payload.if_sr,
                ref_free=payload.ref_free,
                if_freeze=payload.if_freeze,
                pause_second=payload.pause_second,
                use_cuda_graph=payload.use_cuda_graph,
            ),
            return_base64=payload.return_base64,
        )
        return await service.inference(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"按角色情感合成失败: {exc}")


@app.post("/inference/models/load")
async def load_inference_models(
    gpt_path: str = Form(...),
    sovits_path: str = Form(...),
    user: Any = Depends(get_current_user),
):
    """加载推理模型。"""
    service = _ensure_service("inference")
    if not service.load_models(gpt_path, sovits_path):
        raise HTTPException(status_code=400, detail="模型加载失败")
    return {"success": True, "message": "模型加载成功", "gpt_path": gpt_path, "sovits_path": sovits_path}


@app.get("/inference/models/info")
async def get_inference_models_info(
    user: Any = Depends(get_current_user),
):
    """获取当前推理模型信息。"""
    service = _ensure_service("inference")
    return service.get_model_info()


@app.post("/inference/models/unload")
async def unload_inference_models(
    user: Any = Depends(get_current_user),
):
    """手动卸载当前推理模型。"""
    service = _ensure_service("inference")
    unloaded = service.unload_models(reason="manual")
    return {
        "success": True,
        "message": "模型已卸载" if unloaded else "当前没有已加载模型",
        "unloaded": unloaded,
    }


@app.post("/inference/models/cleanup")
async def cleanup_inference_models(
    force: bool = Form(default=False),
    user: Any = Depends(get_current_user),
):
    """执行一次驻留清理。"""
    service = _ensure_service("inference")
    return service.cleanup_resident_models(force=force)


@app.get("/inference/ref-audio")
async def get_reference_audio(
    path: str,
    user: Any = Depends(get_current_user),
):
    """安全返回资源目录中的参考音频，供前端试听。"""
    resources_root = _resources_root()
    model_root = (resources_root / "Model").resolve()
    legacy_music_root = ((MonConfig(start_path=Path(__file__).resolve()).workspace_root() or Path.cwd()) / "music").resolve()
    target_path = Path(path).expanduser().resolve()
    allowed_suffixes = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}

    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="参考音频不存在")
    if target_path.suffix.lower() not in allowed_suffixes:
        raise HTTPException(status_code=400, detail="参考音频格式不受支持")

    allowed_roots = [model_root]
    if legacy_music_root.exists():
        allowed_roots.append(legacy_music_root)

    if not any(root == target_path.parent or root in target_path.parents for root in allowed_roots):
        raise HTTPException(status_code=400, detail="参考音频路径不在允许的资源目录范围内")

    media_type, _ = mimetypes.guess_type(str(target_path))
    return FileResponse(
        str(target_path),
        media_type=media_type or "application/octet-stream",
        filename=target_path.name,
    )


async def execute_format_task(
    task_name: str,
    service: Any,
    list_file: str,
    input_wav_dir: str,
    output_dir: str,
    project_name: str,
    version: str,
):
    """执行格式化任务。"""
    try:
        if task_name == "text_processing":
            from Code.FastApi.Base.DatasetFormatting.text_processing.service import TextProcessingRequest, TextProcessingConfig

            request = TextProcessingRequest(
                input_text_file=list_file,
                input_wav_dir=input_wav_dir,
                experiment_name=project_name,
                output_dir=output_dir,
                config=TextProcessingConfig(),
            )
            return await service.process(request)

        if task_name == "audio_features":
            from Code.FastApi.Base.DatasetFormatting.audio_features.service import AudioFeaturesRequest, AudioFeaturesConfig

            request = AudioFeaturesRequest(
                input_text_file=list_file,
                input_wav_dir=input_wav_dir,
                experiment_name=project_name,
                output_dir=output_dir,
                config=AudioFeaturesConfig(version=version),
            )
            return await service.process(request)

        if task_name == "semantic_encoding":
            from Code.FastApi.Base.DatasetFormatting.semantic_encoding.service import SemanticEncodingRequest, SemanticEncodingConfig

            request = SemanticEncodingRequest(
                input_text_file=list_file,
                cnhubert_dir=os.path.join(output_dir, "4-cnhubert"),
                experiment_name=project_name,
                output_dir=output_dir,
                config=SemanticEncodingConfig(version=version),
            )
            return await service.process(request)

        return {"success": False, "error": f"未知任务: {task_name}"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


async def _run_preprocessing_workflow(
    project_name: str,
    input_audio_dir: str,
    output_dir: str,
    language: str,
    version: str,
    world_name: str = "Standalone",
    experiment_name: str = "",
) -> Dict[str, Any]:
    workflow_steps = []
    layout = _resource_layout(
        project_name=project_name,
        version=version,
        world_name=world_name,
        experiment_name=experiment_name,
    )
    project_root = layout["train_root"]
    dataset_root = layout["dataset_root"]
    slice_output = os.path.join(dataset_root, "sliced")
    model_sliced_dir = layout["model_sliced_dir"]
    asr_output_dir = os.path.join(dataset_root, "asr")
    list_file = os.path.join(asr_output_dir, f"{Path(slice_output).name}.list")
    os.makedirs(dataset_root, exist_ok=True)
    os.makedirs(model_sliced_dir, exist_ok=True)

    slice_service = _require_workflow_service("audio_slice")
    from Code.FastApi.Base.DataPreparation.audio_slice.service import SliceRequest, SliceConfig

    slice_request = SliceRequest(
        input_path=input_audio_dir,
        output_dir=slice_output,
        config=SliceConfig(),
    )
    slice_result = await slice_service.process(slice_request)
    workflow_steps.append({"step": "audio_slice", "result": _to_payload(slice_result)})
    _require_step_success("audio_slice", slice_result)

    copied_slice_files = _sync_directory_files(
        Path(slice_output),
        Path(model_sliced_dir),
    )
    sync_result = {
        "success": bool(copied_slice_files),
        "message": f"已同步 {len(copied_slice_files)} 个切分音频到模型目录" if copied_slice_files else "没有可同步的切分音频",
        "output_dir": model_sliced_dir,
        "output_files": copied_slice_files,
    }
    workflow_steps.append({"step": "model_slice_sync", "result": sync_result})
    _require_step_success("model_slice_sync", sync_result)

    asr_service = _require_workflow_service("asr_recognition")
    from Code.FastApi.Base.DataPreparation.asr_recognition.service import ASRRequest

    asr_request = ASRRequest(
        input_path=slice_output,
        output_dir=asr_output_dir,
        config=_build_asr_config_for_language(language),
    )
    asr_result = await asr_service.process(asr_request)
    workflow_steps.append({"step": "asr_recognition", "result": _to_payload(asr_result)})
    _require_step_success("asr_recognition", asr_result)
    if getattr(asr_result, "output_file", None):
        list_file = asr_result.output_file
    if not Path(list_file).exists():
        _stop_workflow(f"ASR 未生成标注文件: {list_file}")

    for task_name in ("text_processing", "audio_features", "semantic_encoding"):
        service = _require_workflow_service(task_name)
        result = await execute_format_task(
            task_name=task_name,
            service=service,
            list_file=list_file,
            input_wav_dir=slice_output,
            output_dir=dataset_root,
            project_name=project_name,
            version=version,
        )
        workflow_steps.append({"step": task_name, "result": _to_payload(result)})
        _require_step_success(task_name, result)

    return {
        "project_name": project_name,
        "project_root": project_root,
        "dataset_root": dataset_root,
        "model_root": layout["model_root"],
        "model_sliced_dir": model_sliced_dir,
        "gpt_model_dir": layout["gpt_model_dir"],
        "sovits_model_dir": layout["sovits_model_dir"],
        "world_name": layout["world_name"],
        "base_version": layout["base_version"],
        "experiment_name": layout["experiment_name"],
        "slice_output": slice_output,
        "asr_output_dir": asr_output_dir,
        "list_file": list_file,
        "steps": workflow_steps,
    }


async def _start_training_workflow(
    project_name: str,
    project_root: str,
    version: str,
    options: TrainingWorkflowOptions,
    model_root: str = "",
    gpt_model_dir: str = "",
    sovits_model_dir: str = "",
) -> List[Dict[str, Any]]:
    ordered_targets: List[str]
    if options.training_order == "gpt_first":
        ordered_targets = ["gpt", "sovits"]
    else:
        ordered_targets = ["sovits", "gpt"]

    launches: List[Dict[str, Any]] = []
    dataset_check = _validate_training_dataset(
        project_root=project_root,
        version=version,
        require_gpt=options.train_gpt,
        require_sovits=options.train_sovits,
    )
    launches.append({
        "step": "training_dataset_check",
        "result": {
            "success": True,
            "message": "训练前置数据检查通过",
            **dataset_check,
        },
    })

    for target in ordered_targets:
        if target == "gpt" and options.train_gpt:
            service = _ensure_service("gpt_training")
            from Code.FastApi.Base.Training.gpt_training.service import GPTTrainingRequest, GPTTrainingConfig

            request = GPTTrainingRequest(
                exp_name=project_name,
                exp_root=str(Path(project_root).parent),
                workspace_dir=project_root,
                model_output_dir=gpt_model_dir,
                config=GPTTrainingConfig(
                    version=version,
                    batch_size=options.gpt_batch_size,
                    total_epoch=options.gpt_total_epoch,
                    save_every_epoch=options.gpt_total_epoch,
                ),
            )
            result = await service.start_training(request)
            if not _step_result_success(result):
                _stop_workflow(f"GPT训练未启动: {_step_result_message(result)}")
            launches.append({
                "step": "gpt_training",
                "result": _to_payload(result),
            })

        if target == "sovits" and options.train_sovits:
            service = _ensure_service("sovits_training")
            from Code.FastApi.Base.Training.sovits_training.service import SoVITSTrainingRequest, SoVITSTrainingConfig

            request = SoVITSTrainingRequest(
                exp_name=project_name,
                exp_root=str(Path(project_root).parent),
                workspace_dir=project_root,
                model_output_dir=sovits_model_dir,
                config=SoVITSTrainingConfig(
                    version=version,
                    batch_size=options.sovits_batch_size,
                    total_epoch=options.sovits_total_epoch,
                    save_every_epoch=options.sovits_total_epoch,
                ),
            )
            result = await service.start_training(request)
            if not _step_result_success(result):
                _stop_workflow(f"SoVITS训练未启动: {_step_result_message(result)}")
            launches.append({
                "step": "sovits_training",
                "result": _to_payload(result),
            })

    return launches


@app.post("/workflow/complete")
async def complete_workflow(
    project_name: str = Form(...),
    input_audio_dir: str = Form(...),
    output_dir: str = Form(...),
    language: str = Form(default="zh"),
    version: str = Form(default="v2Pro"),
    world_name: str = Form(default="Standalone"),
    experiment_name: str = Form(default=""),
    start_training: bool = Form(default=False),
    train_gpt: bool = Form(default=True),
    train_sovits: bool = Form(default=True),
    gpt_batch_size: int = Form(default=8),
    gpt_total_epoch: int = Form(default=15),
    sovits_batch_size: int = Form(default=32),
    sovits_total_epoch: int = Form(default=8),
    training_order: str = Form(default="sovits_first"),
    user: Any = Depends(get_current_user),
):
    """完整工作流。"""
    try:
        preprocess_result = await _run_preprocessing_workflow(
            project_name=project_name,
            input_audio_dir=input_audio_dir,
            output_dir=output_dir,
            language=language,
            version=version,
            world_name=world_name,
            experiment_name=experiment_name,
        )
        workflow_steps = list(preprocess_result["steps"])
        training_steps: List[Dict[str, Any]] = []

        if start_training:
            training_options = TrainingWorkflowOptions(
                start_training=True,
                train_gpt=train_gpt,
                train_sovits=train_sovits,
                gpt_batch_size=gpt_batch_size,
                gpt_total_epoch=gpt_total_epoch,
                sovits_batch_size=sovits_batch_size,
                sovits_total_epoch=sovits_total_epoch,
                training_order=training_order,
            )
            training_steps = await _start_training_workflow(
                project_name=project_name,
                project_root=preprocess_result["project_root"],
                version=version,
                options=training_options,
                model_root=preprocess_result["model_root"],
                gpt_model_dir=preprocess_result["gpt_model_dir"],
                sovits_model_dir=preprocess_result["sovits_model_dir"],
            )
            workflow_steps.extend(training_steps)

        return {
            "success": True,
            "message": "完整工作流执行完成" if not start_training else "完整工作流与训练引导执行完成",
            "project_name": project_name,
            "project_root": preprocess_result["project_root"],
            "steps": workflow_steps,
            "training_started": start_training,
            "training_steps": training_steps,
            "next_action": "查看训练状态" if start_training else "可以开始训练模型",
        }
    except WorkflowAbortError as exc:
        raise HTTPException(status_code=400, detail=f"工作流已停止: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"工作流执行失败: {exc}")


@app.post("/workflow/training/full")
async def full_training_workflow(
    project_name: str = Form(...),
    input_audio_dir: str = Form(...),
    output_dir: str = Form(...),
    language: str = Form(default="zh"),
    version: str = Form(default="v2Pro"),
    world_name: str = Form(default="Standalone"),
    experiment_name: str = Form(default=""),
    train_gpt: bool = Form(default=True),
    train_sovits: bool = Form(default=True),
    gpt_batch_size: int = Form(default=8),
    gpt_total_epoch: int = Form(default=15),
    sovits_batch_size: int = Form(default=32),
    sovits_total_epoch: int = Form(default=8),
    training_order: str = Form(default="sovits_first"),
    audio_files: list[UploadFile] | None = File(default=None),
    user: Any = Depends(get_current_user),
):
    """训练引导工作流: 预处理 -> 特征 -> 训练。"""
    try:
        saved_audio_files = _save_uploaded_audio_files(input_audio_dir, audio_files)
        training_options = TrainingWorkflowOptions(
            start_training=True,
            train_gpt=train_gpt,
            train_sovits=train_sovits,
            gpt_batch_size=gpt_batch_size,
            gpt_total_epoch=gpt_total_epoch,
            sovits_batch_size=sovits_batch_size,
            sovits_total_epoch=sovits_total_epoch,
            training_order=training_order,
        )

        preprocess_result = await _run_preprocessing_workflow(
            project_name=project_name,
            input_audio_dir=input_audio_dir,
            output_dir=output_dir,
            language=language,
            version=version,
            world_name=world_name,
            experiment_name=experiment_name,
        )
        training_steps = await _start_training_workflow(
            project_name=project_name,
            project_root=preprocess_result["project_root"],
            version=version,
            options=training_options,
            model_root=preprocess_result["model_root"],
            gpt_model_dir=preprocess_result["gpt_model_dir"],
            sovits_model_dir=preprocess_result["sovits_model_dir"],
        )

        return {
            "success": True,
            "message": "训练引导工作流执行完成",
            "workflow_type": "training_full",
            "project_name": project_name,
            "project_root": preprocess_result["project_root"],
            "received_audio_files": saved_audio_files,
            "preprocess_steps": preprocess_result["steps"],
            "training_steps": training_steps,
            "steps": [*preprocess_result["steps"], *training_steps],
            "next_action": "使用 /training/status/{job_id} 查看训练状态",
        }
    except WorkflowAbortError as exc:
        raise HTTPException(status_code=400, detail=f"训练引导工作流已停止: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"训练引导工作流执行失败: {exc}")


@app.post("/batch/projects")
async def batch_process_projects(
    payload: BatchProjectsRequest = Body(...),
    user: Any = Depends(get_current_user),
):
    """批量处理多个项目。"""
    results = []

    for project in payload.projects:
        try:
            result = await complete_workflow(
                project_name=project.name,
                input_audio_dir=project.input_dir,
                output_dir=project.output_dir,
                language=project.language,
                version=project.version,
                user=user,
            )
            results.append({"project": project.name, "result": result})
        except Exception as exc:
            results.append({"project": project.name, "error": str(exc)})

    return {
        "success": True,
        "message": f"批量处理完成，共处理 {len(payload.projects)} 个项目",
        "results": results,
    }


@app.get("/services/status")
async def get_services_status():
    """获取所有服务状态。"""
    status = {}
    for name, service_info in service_manager.services.items():
        try:
            instance = service_info["instance"]
            if hasattr(instance, "get_status"):
                status[name] = await instance.get_status()
            else:
                status[name] = {"status": "available", "prefix": service_info["prefix"]}
        except Exception as exc:
            status[name] = {"status": "error", "error": str(exc)}
    for name, error in service_manager.load_errors.items():
        status[name] = {
            "status": "load_error",
            "error": error,
            "prefix": service_manager.service_configs.get(name, {}).get("prefix"),
        }
    return status


@app.post("/services/reload/{service_name}")
async def reload_service(service_name: str):
    """重新加载指定服务。"""
    if service_name not in service_manager.service_configs:
        raise HTTPException(status_code=404, detail=f"服务不存在: {service_name}")

    try:
        service_manager.reload_service(service_name)
        return {"success": True, "message": f"服务 {service_name} 重新加载成功"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"服务重新加载失败: {exc}")


# ==================== ASR 模块 ====================

from Code.FastApi.Base.ASR import voice_service as asr_voice_service


@app.websocket("/ws/asr/transcribe")
async def ws_asr_transcribe(websocket: WebSocket):
    from Code.FastApi.Base.ASR.consumers.asr import ASRWebSocketHandler
    handler = ASRWebSocketHandler()
    await websocket.accept()
    await handler.handle_connect(websocket)
    try:
        while True:
            raw = await websocket.receive()
            if raw.get("type") == "websocket.receive":
                if "bytes" in raw:
                    payload = raw["bytes"]
                    await handler.handle_audio(websocket, payload)
                elif "text" in raw:
                    await handler.handle_text(websocket, raw["text"])
    except Exception as exc:
        print(f"[WS-ASR] WebSocket异常: {exc}")


@app.websocket("/ws/asr/final")
async def ws_asr_final(websocket: WebSocket):
    from Code.FastApi.Base.ASR.consumers.final import ASRFinalWebSocketHandler
    handler = ASRFinalWebSocketHandler()
    await websocket.accept()
    await handler.handle_connect(websocket)
    try:
        while True:
            raw = await websocket.receive()
            if raw.get("type") == "websocket.receive":
                if "bytes" in raw:
                    payload = raw["bytes"]
                    await handler.handle_audio(websocket, payload)
                elif "text" in raw:
                    await handler.handle_text(websocket, raw["text"])
            elif raw.get("type") == "websocket.disconnect":
                break
    except Exception as exc:
        print(f"[WS-ASR-FINAL] WebSocket异常: {exc}")


@app.websocket("/ws/asr/vad")
async def ws_asr_vad(websocket: WebSocket):
    from Code.FastApi.Base.ASR.consumers.vad import VADWebSocketHandler
    handler = VADWebSocketHandler()
    await websocket.accept()
    await handler.handle_connect(websocket)
    try:
        while True:
            raw = await websocket.receive()
            if raw.get("type") == "websocket.receive":
                if "bytes" in raw:
                    payload = raw["bytes"]
                    await handler.handle_audio(websocket, payload)
                elif "text" in raw:
                    await handler.handle_text(websocket, raw["text"])
    except Exception:
        pass


@app.websocket("/ws/tts/stream")
async def ws_tts_stream(websocket: WebSocket):
    from Code.FastApi.Base.TTS.consumers.stream import TTSStreamWebSocketHandler

    service = _ensure_service("inference")
    handler = TTSStreamWebSocketHandler(service)
    await websocket.accept()
    await handler.handle_connect(websocket)
    try:
        while True:
            raw = await websocket.receive()
            if raw.get("type") == "websocket.receive":
                if "text" in raw:
                    await handler.handle_text(websocket, raw["text"])
            elif raw.get("type") == "websocket.disconnect":
                break
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        await handler.handle_disconnect()


@app.post("/asr/speaker/register/")
async def asr_speaker_register(
    audio_file: UploadFile = File(...),
    speaker_id: str = Form(...),
    name: str = Form(...),
    user: Any = Depends(get_current_user),
):
    """注册说话人声纹。"""
    cleanup_paths: list[str] = []
    try:
        import tempfile

        suffix = Path(audio_file.filename or "upload.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await audio_file.read()
            tmp.write(content)
            tmp_path = tmp.name
        cleanup_paths.append(tmp_path)

        import subprocess
        wav_path = tmp_path.replace(suffix, ".wav")
        cleanup_paths.append(wav_path)
        subprocess.run([
            os.environ.get("FFMPEG_PATH", "ffmpeg"),
            "-i", tmp_path, "-ar", "16000", "-ac", "1", "-y", wav_path,
        ], capture_output=True, check=True)

        embedding = asr_voice_service.speaker.get_embedding(wav_path)
        success = asr_voice_service.speaker_db.register(speaker_id, name, embedding)
        return {"success": success, "message": f"说话人 {name} 注册成功", "speaker_id": speaker_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"注册失败: {exc}")
    finally:
        for p in cleanup_paths:
            if os.path.exists(p):
                os.unlink(p)


@app.post("/asr/speaker/unregister/")
async def asr_speaker_unregister(
    speaker_id: str = Form(...),
    user: Any = Depends(get_current_user),
):
    """注销说话人。"""
    try:
        success = asr_voice_service.speaker_db.unregister(speaker_id)
        if success:
            return {"success": True, "message": f"说话人 {speaker_id} 已注销"}
        raise HTTPException(status_code=404, detail=f"说话人 {speaker_id} 不存在")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"注销失败: {exc}")


@app.get("/asr/speaker/list/")
async def asr_speaker_list(
    user: Any = Depends(get_current_user),
):
    """列出已注册的说话人。"""
    try:
        speakers = asr_voice_service.speaker_db.list_speakers()
        return {"success": True, "speakers": speakers, "count": len(speakers)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"获取列表失败: {exc}")


@app.post("/asr/speaker/identify/")
async def asr_speaker_identify(
    audio_file: UploadFile = File(...),
    threshold: float = Form(default=0.75),
    user: Any = Depends(get_current_user),
):
    """识别音频中的说话人。"""
    cleanup_paths: list[str] = []
    try:
        import tempfile
        suffix = Path(audio_file.filename or "upload.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await audio_file.read()
            tmp.write(content)
            tmp_path = tmp.name
        cleanup_paths.append(tmp_path)

        import subprocess
        wav_path = tmp_path.replace(suffix, ".wav")
        cleanup_paths.append(wav_path)
        subprocess.run([
            os.environ.get("FFMPEG_PATH", "ffmpeg"),
            "-i", tmp_path, "-ar", "16000", "-ac", "1", "-y", wav_path,
        ], capture_output=True, check=True)

        result = asr_voice_service.identify_speaker_from_audio(wav_path, threshold)
        return {"success": True, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"识别失败: {exc}")
    finally:
        for p in cleanup_paths:
            if os.path.exists(p):
                os.unlink(p)


@app.post("/asr/speaker/verify/")
async def asr_speaker_verify(
    audio1: UploadFile = File(...),
    audio2: UploadFile = File(...),
    user: Any = Depends(get_current_user),
):
    """验证两段音频是否同一人。"""
    cleanup_paths: list[str] = []
    try:
        import tempfile
        wav_paths: list[str] = []
        for i, af in enumerate([audio1, audio2]):
            suffix = Path(af.filename or f"audio{i}.wav").suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await af.read()
                tmp.write(content)
                raw_path = tmp.name
            cleanup_paths.append(raw_path)
            wav_path = raw_path.replace(suffix, ".wav")
            cleanup_paths.append(wav_path)
            import subprocess
            subprocess.run([
                os.environ.get("FFMPEG_PATH", "ffmpeg"),
                "-i", raw_path, "-ar", "16000", "-ac", "1", "-y", wav_path,
            ], capture_output=True, check=True)
            wav_paths.append(wav_path)

        result = asr_voice_service.verify_speaker(wav_paths[0], wav_paths[1])
        return {"success": True, "similarity": result["similarity"], "is_same": result["is_same"]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"验证失败: {exc}")
    finally:
        for p in cleanup_paths:
            if os.path.exists(p):
                os.unlink(p)


@app.post("/asr/diarize/")
async def asr_diarize(
    audio_file: UploadFile = File(...),
    language: str = Form(default="auto"),
    threshold: float = Form(default=0.75),
    user: Any = Depends(get_current_user),
):
    """说话人日志：VAD 分段 + ASR + 说话人聚类。"""
    cleanup_paths: list[str] = []
    try:
        import tempfile
        suffix = Path(audio_file.filename or "upload.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await audio_file.read()
            tmp.write(content)
            tmp_path = tmp.name
        cleanup_paths.append(tmp_path)

        import subprocess
        wav_path = tmp_path.replace(suffix, ".wav")
        cleanup_paths.append(wav_path)
        subprocess.run([
            os.environ.get("FFMPEG_PATH", "ffmpeg"),
            "-i", tmp_path, "-ar", "16000", "-ac", "1", "-y", wav_path,
        ], capture_output=True, check=True)

        result = asr_voice_service.process_audio_with_diarization(wav_path, language, threshold)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"说话人日志失败: {exc}")
    finally:
        for p in cleanup_paths:
            if os.path.exists(p):
                os.unlink(p)


@app.post("/asr/transcribe/")
async def asr_transcribe(
    audio_file: UploadFile = File(...),
    language: str = Form(default="auto"),
    user: Any = Depends(get_current_user),
):
    """统一 ASR 转写接口（流式 Paraformer，单文件）。"""
    cleanup_paths: list[str] = []
    try:
        import tempfile
        suffix = Path(audio_file.filename or "upload.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await audio_file.read()
            tmp.write(content)
            tmp_path = tmp.name
        cleanup_paths.append(tmp_path)

        import subprocess
        wav_path = tmp_path.replace(suffix, ".wav")
        cleanup_paths.append(wav_path)
        subprocess.run([
            os.environ.get("FFMPEG_PATH", "ffmpeg"),
            "-i", tmp_path, "-ar", "16000", "-ac", "1", "-y", wav_path,
        ], capture_output=True, check=True)

        result = asr_voice_service.process_audio(wav_path)
        return {
            "success": result["status"] == "success",
            "text": result.get("text", ""),
            "status": result["status"],
            "segments": result.get("segments", []),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ASR 转写失败: {exc}")
    finally:
        for p in cleanup_paths:
            if os.path.exists(p):
                os.unlink(p)


if __name__ == "__main__":
    import uvicorn

    print("启动 GPT-SoVITS 统一网关服务")
    print("服务地址: http://localhost:8000")
    print("API 文档: http://localhost:8000/docs")
    print("服务数量:", len(service_manager.services))

    uvicorn.run(app, host="0.0.0.0", port=8000)
