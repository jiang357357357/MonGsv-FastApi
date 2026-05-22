"""
World request models.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class WorldCreateRequest(BaseModel):
    name: str = Field(min_length=1)
    description: str = ""
    version: str | None = None


class WorldDeleteRequest(BaseModel):
    id: int

