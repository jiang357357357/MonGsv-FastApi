"""
Model domain router.
"""

from __future__ import annotations

from fastapi import APIRouter

from Code.FastApi.Domain.Model.Models import ModelListResponse
from Code.FastApi.Domain.Model.Services import ModelCatalogService


def build_model_router() -> APIRouter:
    router = APIRouter(tags=["domain", "model"])
    service = ModelCatalogService()

    @router.get("/api/gpt/list/")
    async def list_gpt_models():
        models = service.list_gpt_models()
        return ModelListResponse(message="获取 GPT 模型列表成功", models=models, count=len(models))

    @router.get("/api/sov/list/")
    async def list_sov_models():
        models = service.list_sovits_models()
        return ModelListResponse(message="获取 SoVITS 模型列表成功", models=models, count=len(models))

    return router

