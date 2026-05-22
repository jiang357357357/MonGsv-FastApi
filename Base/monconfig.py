"""
MonConfig loader for FastApi.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


class MonConfig:
    """Load and merge `.monconfig` files from the current workspace upwards."""

    def __init__(self, start_path: Optional[Path] = None, max_depth: int = 10):
        self.start_path = Path(start_path or Path.cwd()).resolve()
        self.max_depth = max_depth
        self._config: Dict[str, Dict[str, str]] = {}
        self._loaded_files: List[Path] = []
        self._workspace_root: Optional[Path] = None
        self._load()

    def get(self, section: str, key: str, default: Any = None, cast: Optional[type] = None) -> Any:
        value = self._config.get(section, {}).get(key)
        if value is None:
            return default
        if cast is None:
            return value
        return self._cast(value, cast, default)

    def section(self, section: str) -> Dict[str, str]:
        return self._config.get(section, {}).copy()

    def workspace_root(self) -> Optional[Path]:
        return self._workspace_root

    def loaded_files(self) -> List[Path]:
        return self._loaded_files.copy()

    def resolve_path(self, section: str, key: str, default: Optional[str] = None) -> Optional[Path]:
        raw_value = self.get(section, key, default)
        if raw_value in (None, ""):
            return None
        path = Path(str(raw_value))
        if path.is_absolute():
            return path
        root = self.workspace_root()
        return (root / path).resolve() if root else path.resolve()

    def _load(self):
        config_files = self._find_config_files()
        if config_files:
            self._workspace_root = config_files[0].parent

        for config_file in reversed(config_files):
            self._parse_config_file(config_file)
            self._loaded_files.append(config_file)

    def _find_config_files(self) -> List[Path]:
        config_files: List[Path] = []
        current = self.start_path if self.start_path.is_dir() else self.start_path.parent

        for _ in range(self.max_depth):
            config_file = current / ".monconfig"
            if config_file.exists():
                config_files.append(config_file)
            if current == current.parent:
                break
            current = current.parent

        return config_files

    def _parse_config_file(self, config_file: Path):
        current_section = "default"
        with config_file.open("r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                if line.startswith("[") and line.endswith("]"):
                    current_section = line[1:-1].strip()
                    self._config.setdefault(current_section, {})
                    continue

                if "=" not in line:
                    continue

                line_without_comment = line.split("#", 1)[0].strip()
                if "=" not in line_without_comment:
                    continue

                key, value = line_without_comment.split("=", 1)
                self._config.setdefault(current_section, {})[key.strip()] = value.strip()

    @staticmethod
    def _cast(value: str, cast_type: type, default: Any = None) -> Any:
        try:
            if cast_type is bool:
                return value.strip().lower() in {"true", "yes", "1", "on", "enabled"}
            if cast_type is int:
                return int(value)
            if cast_type is float:
                return float(value)
            if cast_type is Path:
                return Path(value)
            return cast_type(value)
        except Exception:
            return default
