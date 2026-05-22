"""
Model response models.
"""

from __future__ import annotations

from pydantic import BaseModel

from Code.FastApi.Domain.Model.Models.model_info import ModelInfo


class ModelListResponse(BaseModel):
    success: bool = True
    message: str = "ok"
    models: list[ModelInfo]
    count: int

