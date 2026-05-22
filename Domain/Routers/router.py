"""
Aggregate domain router.
"""

from __future__ import annotations

from fastapi import APIRouter

from Code.FastApi.Domain.Routers.model_router import build_model_router
from Code.FastApi.Domain.Routers.role_router import build_role_router
from Code.FastApi.Domain.Routers.version_router import build_version_router
from Code.FastApi.Domain.Routers.world_router import build_world_router


def build_domain_router() -> APIRouter:
    router = APIRouter(tags=["domain"])
    router.include_router(build_model_router())
    router.include_router(build_version_router())
    router.include_router(build_world_router())
    router.include_router(build_role_router())
    return router
