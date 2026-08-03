"""网关基础服务的注册、动态加载与重载。"""

from __future__ import annotations

import importlib.util
import traceback
from pathlib import Path
from typing import Any, Dict


BASE_DIR = Path(__file__).resolve().parent.parent

SERVICE_CONFIGS = (
    ("audio_slice", "DataPreparation/audio_slice/service.py", "AudioSliceService", "/data-prep/audio-slice"),
    ("asr_recognition", "DataPreparation/asr_recognition/service.py", "ASRRecognitionService", "/data-prep/asr"),
    ("text_processing", "DatasetFormatting/text_processing/service.py", "TextProcessingService", "/dataset/text"),
    ("audio_features", "DatasetFormatting/audio_features/service.py", "AudioFeaturesService", "/dataset/audio"),
    ("semantic_encoding", "DatasetFormatting/semantic_encoding/service.py", "SemanticEncodingService", "/dataset/semantic"),
    ("gpt_training", "Training/gpt_training/service.py", "GPTTrainingService", "/training/gpt"),
    ("sovits_training", "Training/sovits_training/service.py", "SoVITSTrainingService", "/training/sovits"),
    ("inference", "Inference/service.py", "InferenceService", "/inference"),
)


class ServiceManager:
    """动态加载基础封装服务，并隔离单个服务的加载失败。"""

    def __init__(self) -> None:
        self.services: Dict[str, Dict[str, Any]] = {}
        self.service_configs: Dict[str, Dict[str, str]] = {}
        self.load_errors: Dict[str, str] = {}
        self.load_all_services()

    def load_all_services(self) -> None:
        for name, path, class_name, prefix in SERVICE_CONFIGS:
            config = {"name": name, "path": path, "class": class_name, "prefix": prefix}
            self.service_configs[name] = config
            try:
                self.load_service(config)
                print(f"成功加载服务: {name}")
            except Exception as exc:
                self.load_errors[name] = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                print(f"加载服务失败 {name}: {exc}")

    def load_service(self, config: Dict[str, str]) -> None:
        module_path = BASE_DIR / config["path"]
        if not module_path.exists():
            raise FileNotFoundError(f"模块文件不存在: {module_path}")

        relative_module = config["path"][:-3].replace("/", ".").replace("\\", ".")
        module_name = f"Code.FastApi.Base.{relative_module}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法创建模块加载器: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        service_class = getattr(module, config["class"])
        self.services[config["name"]] = {
            "instance": service_class(),
            "prefix": config["prefix"],
            "module": module,
        }
        self.load_errors.pop(config["name"], None)

    def get_service(self, name: str):
        return self.services.get(name, {}).get("instance")

    def reload_service(self, name: str) -> None:
        config = self.service_configs.get(name)
        if not config:
            raise KeyError(f"服务不存在: {name}")
        self.load_service(config)
