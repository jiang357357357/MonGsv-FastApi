"""
Role entity model.
"""

from __future__ import annotations

from pydantic import BaseModel


class RoleInfo(BaseModel):
    id: int
    name: str
    description: str = ""
    world_id: int | None = None
    world_name: str | None = None
    version: str | None = None
    gpt_model_id: int | None = None
    gpt_model_name: str | None = None
    sov_model_id: int | None = None
    sov_model_name: str | None = None
    gpt_model_path: str | None = None
    sov_model_path: str | None = None
    prompt_text: str | None = None
    prompt_audio_path: str | None = None
    language: str | None = "zh"
