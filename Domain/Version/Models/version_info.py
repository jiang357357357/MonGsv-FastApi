"""
Version entity model.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class VersionInfo(BaseModel):
    name: str = Field(description="Version name")
    source: str = Field(description="Source type: enum or dir")
    is_default: bool = Field(default=False)
    is_current: bool = Field(default=False)

