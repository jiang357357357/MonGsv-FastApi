"""
Version domain service.
"""

from __future__ import annotations

from pathlib import Path

from Code.FastApi.Base.monconfig import MonConfig
from Code.FastApi.Domain.Version.Models.responses import DirectoryVersionResponse, EnumVersionResponse


class VersionService:
    """Resolve available GPT-SoVITS versions."""

    ENUM_VERSIONS = ["v1", "v2", "v3", "v4", "v2Pro", "v2ProPlus"]

    def __init__(self):
        self._config = MonConfig(start_path=Path(__file__).resolve())

    def get_enum_versions(self) -> EnumVersionResponse:
        default_version = self._config.get("model", "DEFAULT_VERSION", "v2Pro")
        current_version = self._config.get("model", "CURRENT_VERSION", default_version)
        return EnumVersionResponse(
            versions=self.ENUM_VERSIONS.copy(),
            current_version=current_version,
            default_version=default_version,
        )

    def get_directory_versions(self) -> DirectoryVersionResponse:
        models_dir = self._config.resolve_path("paths", "MODELS_DIR")
        discovered: set[str] = set()

        if models_dir and models_dir.exists():
            for item in models_dir.iterdir():
                if not item.is_dir():
                    continue
                name = item.name.strip()
                if not name:
                    continue
                if name in self.ENUM_VERSIONS or name.lower().startswith("v"):
                    discovered.add(name)

        versions = sorted(discovered)
        return DirectoryVersionResponse(versions=versions, count=len(versions))
