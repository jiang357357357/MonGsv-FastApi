"""
Version domain router.
"""

from __future__ import annotations

from fastapi import APIRouter

from Code.FastApi.Domain.Version.Services import VersionService


def build_version_router() -> APIRouter:
    router = APIRouter(tags=["domain", "version"])
    service = VersionService()

    @router.get("/api/models/versions/from-enum/")
    async def get_enum_versions():
        return service.get_enum_versions()

    @router.get("/api/models/versions/from-dir/")
    async def get_dir_versions():
        return service.get_directory_versions()

    return router

