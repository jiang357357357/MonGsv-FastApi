"""
World domain service.
"""

from __future__ import annotations

import shutil
import zlib
from pathlib import Path
from typing import List

from Code.FastApi.Base.monconfig import MonConfig
from Code.FastApi.Domain.World.Models.requests import WorldCreateRequest
from Code.FastApi.Domain.World.Models.world_info import WorldInfo


class WorldService:
    """CRUD service for world resources."""

    def __init__(self):
        self._config = MonConfig(start_path=Path(__file__).resolve())

    def list_worlds(self, version: str | None = None) -> List[WorldInfo]:
        _ = version
        items: list[WorldInfo] = []
        model_root = self._model_root()
        for world_dir in sorted(model_root.iterdir(), key=lambda item: item.name.lower()) if model_root.exists() else []:
            if not world_dir.is_dir():
                continue
            items.append(
                WorldInfo(
                    id=self._world_id(world_dir.name),
                    name=world_dir.name,
                    description="",
                    version=None,
                )
            )
        return items

    def create_world(self, payload: WorldCreateRequest) -> WorldInfo:
        normalized_name = payload.name.strip()
        if not normalized_name:
            raise ValueError("世界名称不能为空")

        model_root = self._model_root()
        target_dir = model_root / normalized_name
        if target_dir.exists():
            raise ValueError(f"世界已存在: {normalized_name}")

        world = WorldInfo(
            id=self._world_id(normalized_name),
            name=normalized_name,
            description="",
            version=None,
        )
        target_dir.mkdir(parents=True, exist_ok=False)
        return world

    def delete_world(self, world_id: int) -> WorldInfo:
        world = self._find_world_by_id(world_id)
        if world is None:
            raise ValueError(f"世界不存在: {world_id}")

        model_world_root = self._model_root() / world.name
        train_world_root = self._train_projects_root() / world.name

        self._remove_tree(model_world_root)
        self._remove_tree(train_world_root)
        return world

    def _find_world_by_id(self, world_id: int) -> WorldInfo | None:
        for world in self.list_worlds():
            if world.id == world_id:
                return world
        return None

    def _resources_root(self) -> Path:
        workspace_root = self._config.workspace_root() or Path.cwd()
        resources_root = (workspace_root / "Resources").resolve()
        resources_root.mkdir(parents=True, exist_ok=True)
        return resources_root

    def _model_root(self) -> Path:
        model_root = (self._resources_root() / "Model").resolve()
        model_root.mkdir(parents=True, exist_ok=True)
        return model_root

    def _train_projects_root(self) -> Path:
        train_root = (self._resources_root() / "Train" / "Projects").resolve()
        train_root.mkdir(parents=True, exist_ok=True)
        return train_root

    @staticmethod
    def _remove_tree(path: Path) -> None:
        if path.exists():
            shutil.rmtree(path)

    @staticmethod
    def _world_id(name: str) -> int:
        return zlib.crc32(name.strip().lower().encode("utf-8")) & 0x7FFFFFFF
