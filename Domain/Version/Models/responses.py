"""
Version response models.
"""

from __future__ import annotations

from pydantic import BaseModel


class EnumVersionResponse(BaseModel):
    success: bool = True
    message: str = "ok"
    versions: list[str]
    current_version: str | None = None
    default_version: str | None = None


class DirectoryVersionResponse(BaseModel):
    success: bool = True
    message: str = "ok"
    versions: list[str]
    count: int

