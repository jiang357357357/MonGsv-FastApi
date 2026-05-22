"""
Shared helpers for resource domain persistence.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List

from Code.FastApi.Base.monconfig import MonConfig


class JsonResourceStore:
    """Simple JSON-backed store for small resource collections."""

    def __init__(self, filename: str):
        self._lock = RLock()
        self._config = MonConfig(start_path=Path(__file__).resolve())
        self._base_dir = self._resolve_base_dir()
        self._file_path = self._base_dir / filename

    @property
    def file_path(self) -> Path:
        return self._file_path

    def load_items(self) -> List[Dict[str, Any]]:
        with self._lock:
            payload = self._load_payload()
            items = payload.get("items", [])
            return items if isinstance(items, list) else []

    def save_items(self, items: List[Dict[str, Any]]) -> None:
        with self._lock:
            payload = self._load_payload()
            payload["items"] = items
            self._write_payload(payload)

    def next_id(self) -> int:
        with self._lock:
            payload = self._load_payload()
            next_value = int(payload.get("next_id", 1))
            payload["next_id"] = next_value + 1
            self._write_payload(payload)
            return next_value

    def _resolve_base_dir(self) -> Path:
        configured = self._config.resolve_path("paths", "RESOURCE_META_DIR")
        if configured is not None:
            configured.mkdir(parents=True, exist_ok=True)
            return configured

        workspace_root = self._config.workspace_root() or Path.cwd()
        base_dir = (workspace_root / "Data" / "Meta" / "FastApi").resolve()
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    def _load_payload(self) -> Dict[str, Any]:
        self._base_dir.mkdir(parents=True, exist_ok=True)
        if not self._file_path.exists():
            return {"next_id": 1, "items": []}

        with self._file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            return {"next_id": 1, "items": []}

        payload.setdefault("next_id", 1)
        payload.setdefault("items", [])
        return payload

    def _write_payload(self, payload: Dict[str, Any]) -> None:
        self._base_dir.mkdir(parents=True, exist_ok=True)
        with self._file_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

