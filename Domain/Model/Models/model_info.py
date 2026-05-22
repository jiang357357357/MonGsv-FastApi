"""
Model entity model.
"""

from __future__ import annotations

from pydantic import BaseModel


class ModelInfo(BaseModel):
    id: int
    name: str
    path: str | None = None
    version: str | None = None

