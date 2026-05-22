"""
Role request models.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RoleCreateRequest(BaseModel):
    name: str = Field(min_length=1)
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


class RoleUpdateRequest(RoleCreateRequest):
    id: int


class RoleDeleteRequest(BaseModel):
    id: int


class RoleWorkspaceCreateRequest(BaseModel):
    role_name: str = Field(min_length=1)
    description: str = ""
    world_name: str | None = None
    language: str | None = "zh-CN"
    version: str | None = None
    base_version: str | None = None
    experiment_name: str | None = None
    overwrite: bool = False
