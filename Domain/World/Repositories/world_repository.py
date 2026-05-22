"""
World persistence repository.
"""

from __future__ import annotations

from Code.FastApi.Domain.World.Models.world_info import WorldInfo
from Code.FastApi.Domain.common import JsonResourceStore


class WorldRepository:
    def __init__(self):
        self._store = JsonResourceStore("worlds.json")

    def list_items(self) -> list[WorldInfo]:
        return [WorldInfo.model_validate(item) for item in self._store.load_items()]

    def save_items(self, worlds: list[WorldInfo]) -> None:
        self._store.save_items([world.model_dump() for world in worlds])

    def next_id(self) -> int:
        return self._store.next_id()

