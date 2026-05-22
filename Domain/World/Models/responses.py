"""
World response models.
"""

from __future__ import annotations

from pydantic import BaseModel

from Code.FastApi.Domain.World.Models.world_info import WorldInfo


class WorldListResponse(BaseModel):
    success: bool = True
    message: str = "ok"
    version: str | None = None
    worlds: list[WorldInfo]
    count: int


class WorldMutationResponse(BaseModel):
    success: bool = True
    message: str = "ok"
    data: dict | None = None

