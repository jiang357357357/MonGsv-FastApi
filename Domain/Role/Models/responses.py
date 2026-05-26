"""
Role response models.
"""

from __future__ import annotations

from pydantic import BaseModel

from Code.FastApi.Domain.Role.Models.role_info import RoleInfo


class RoleEmotionInfo(BaseModel):
    name: str
    text: str
    music_url: str
    text_language: str | None = None
    file_name: str


class RoleEmotionListResponse(BaseModel):
    success: bool = True
    message: str = "ok"
    role_id: int
    emotions: list[RoleEmotionInfo]
    count: int


class RoleListResponse(BaseModel):
    success: bool = True
    message: str = "ok"
    world_id: str | None = None
    world_name: str | None = None
    roles: list[RoleInfo]
    count: int


class RoleMutationResponse(BaseModel):
    success: bool = True
    message: str = "ok"
    data: dict | None = None


class RoleWorkspaceMutationResponse(BaseModel):
    success: bool = True
    message: str = "ok"
    data: dict | None = None


class RoleWorkspaceInitializeResponse(BaseModel):
    success: bool = True
    message: str = "ok"
    data: dict | None = None


class RoleWorkspaceSummary(BaseModel):
    role_name: str
    world_name: str | None = None
    base_version: str | None = None
    experiment_name: str | None = None
    role_root: str
    model_root: str | None = None
    train_root: str | None = None
    model_sliced_dir: str
    raw_dir: str
    sliced_dir: str
    raw_files: list[str]
    model_sliced_files: list[str]
    sliced_files: list[str] = []
    prompt_dir: str
    prompt_files: list[str]
    gpt_models: list[str]
    sovits_models: list[str]


class RoleWorkspaceListResponse(BaseModel):
    success: bool = True
    message: str = "ok"
    workspaces: list[RoleWorkspaceSummary]
    count: int
