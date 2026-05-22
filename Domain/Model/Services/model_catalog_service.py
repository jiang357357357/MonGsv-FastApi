"""
Model catalog service.
"""

from __future__ import annotations

from pathlib import Path

from Code.FastApi.Base.monconfig import MonConfig
from Code.FastApi.Domain.Model.Models.model_info import ModelInfo


class ModelCatalogService:
    """Read-only catalog for GPT and SoVITS model files."""

    GPT_EXTENSIONS = {".ckpt"}
    SOVITS_EXTENSIONS = {".pth"}

    def __init__(self):
        self._config = MonConfig(start_path=Path(__file__).resolve())

    def list_gpt_models(self) -> list[ModelInfo]:
        return self._scan_models("gpt", self.GPT_EXTENSIONS)

    def list_sovits_models(self) -> list[ModelInfo]:
        return self._scan_models("sovits", self.SOVITS_EXTENSIONS)

    def get_gpt_model(self, model_id: int) -> ModelInfo:
        return self._get_model_by_id(model_id, self.list_gpt_models(), "GPT")

    def get_sovits_model(self, model_id: int) -> ModelInfo:
        return self._get_model_by_id(model_id, self.list_sovits_models(), "SoVITS")

    def _scan_models(self, model_type: str, extensions: set[str]) -> list[ModelInfo]:
        models_dir = self._config.resolve_path("paths", "MODELS_DIR")
        if models_dir is None or not models_dir.exists():
            return []

        discovered: list[Path] = []
        for path in models_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in extensions:
                continue
            if self._should_skip_path(path):
                continue
            discovered.append(path)

        discovered.sort(key=lambda item: str(item).lower())
        return [
            ModelInfo(
                id=index,
                name=path.stem,
                path=str(path),
                version=self._extract_version(path),
            )
            for index, path in enumerate(discovered, start=1)
        ]

    @staticmethod
    def _get_model_by_id(model_id: int, models: list[ModelInfo], label: str) -> ModelInfo:
        for model in models:
            if model.id == model_id:
                return model
        raise ValueError(f"{label} 模型不存在: {model_id}")

    @staticmethod
    def _extract_version(path: Path) -> str | None:
        for part in path.parts:
            text = part.strip()
            if text.lower().startswith("v") and len(text) > 1:
                return text
        return None

    @staticmethod
    def _should_skip_path(path: Path) -> bool:
        lowered = {part.lower() for part in path.parts}
        return "__pycache__" in lowered

