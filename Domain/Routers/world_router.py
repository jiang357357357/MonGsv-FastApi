"""
World domain router.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from Code.FastApi.Domain.Routers.common import error_response
from Code.FastApi.Domain.World.Models import (
    WorldCreateRequest,
    WorldDeleteRequest,
    WorldListResponse,
    WorldMutationResponse,
)
from Code.FastApi.Domain.World.Services import WorldService


def build_world_router() -> APIRouter:
    router = APIRouter(tags=["domain", "world"])
    service = WorldService()

    @router.get("/api/world/list/")
    async def list_worlds(version: str | None = Query(default=None)):
        worlds = service.list_worlds(version=version)
        return WorldListResponse(version=version, worlds=worlds, count=len(worlds))

    @router.post("/api/world/create/")
    async def create_world(payload: WorldCreateRequest):
        try:
            world = service.create_world(payload)
            return WorldMutationResponse(
                message="世界创建成功",
                data={"id": world.id, "name": world.name},
            )
        except ValueError as exc:
            return error_response(400, str(exc))

    @router.post("/api/world/delete/")
    async def delete_world(payload: WorldDeleteRequest):
        try:
            service.delete_world(payload.id)
            return WorldMutationResponse(message="世界删除成功")
        except ValueError as exc:
            return error_response(404, str(exc))

    return router

