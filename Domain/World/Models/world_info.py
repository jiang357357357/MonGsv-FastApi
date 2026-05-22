"""
World entity model.
"""

from __future__ import annotations

from pydantic import BaseModel


class WorldInfo(BaseModel):
    id: int
    name: str
    description: str = ""
    version: str | None = None

