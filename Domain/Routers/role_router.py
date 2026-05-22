"""
Role domain router.
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, Query, UploadFile

from Code.FastApi.Domain.Role.Models import (
    RoleEmotionListResponse,
    RoleCreateRequest,
    RoleDeleteRequest,
    RoleListResponse,
    RoleMutationResponse,
    RoleUpdateRequest,
    RoleWorkspaceCreateRequest,
    RoleWorkspaceInitializeResponse,
    RoleWorkspaceListResponse,
    RoleWorkspaceMutationResponse,
)
from Code.FastApi.Domain.Role.Services import RoleService
from Code.FastApi.Domain.Routers.common import error_response


def build_role_router() -> APIRouter:
    router = APIRouter(tags=["domain", "role"])
    service = RoleService()

    @router.get("/api/role/list/")
    async def list_roles(
        version: str | None = Query(default=None),
        world_id: int | None = Query(default=None),
        world_name: str | None = Query(default=None),
    ):
        roles = service.list_roles(version=version, world_id=world_id, world_name=world_name)
        return RoleListResponse(
            world_id=str(world_id) if world_id is not None else None,
            world_name=world_name,
            roles=roles,
            count=len(roles),
        )

    @router.post("/api/role/create/")
    async def create_role(payload: RoleCreateRequest):
        try:
            role = service.create_role(payload)
            return RoleMutationResponse(
                message="角色创建成功",
                data={"id": role.id, "name": role.name},
            )
        except ValueError as exc:
            return error_response(400, str(exc))

    @router.post("/api/role/update/")
    async def update_role(payload: RoleUpdateRequest):
        try:
            role = service.update_role(payload)
            return RoleMutationResponse(
                message="角色更新成功",
                data={"id": role.id, "name": role.name},
            )
        except ValueError as exc:
            return error_response(400, str(exc))

    @router.post("/api/role/delete/")
    async def delete_role(payload: RoleDeleteRequest):
        try:
            service.delete_role(payload.id)
            return RoleMutationResponse(message="角色删除成功")
        except ValueError as exc:
            return error_response(404, str(exc))

    @router.get("/api/role/emotions/")
    async def list_role_emotions(role_id: int = Query(...)):
        try:
            emotions = service.list_role_emotions(role_id)
            return RoleEmotionListResponse(
                role_id=role_id,
                emotions=emotions,
                count=len(emotions),
            )
        except ValueError as exc:
            return error_response(404, str(exc))

    @router.post("/api/role/emotions/upsert/")
    async def upsert_role_emotion(
        role_id: int = Form(...),
        emotion_name: str = Form(...),
        emotion_text: str = Form(...),
        text_language: str | None = Form(default=None),
        audio_file: UploadFile | None = File(default=None),
        audio_source_path: str | None = Form(default=None),
    ):
        try:
            audio_content = await audio_file.read() if audio_file is not None else None
            result = service.upsert_role_emotion(
                role_id=role_id,
                emotion_name=emotion_name,
                emotion_text=emotion_text,
                text_language=text_language,
                audio_filename=audio_file.filename if audio_file is not None else None,
                audio_content=audio_content,
                audio_source_path=audio_source_path,
            )
            return RoleMutationResponse(
                message="情感配置保存成功",
                data=result,
            )
        except ValueError as exc:
            return error_response(400, str(exc))

    @router.post("/api/role/emotions/delete/")
    async def delete_role_emotion(
        role_id: int = Form(...),
        emotion_name: str = Form(...),
    ):
        try:
            result = service.delete_role_emotion(role_id, emotion_name)
            return RoleMutationResponse(
                message="情感配置删除成功",
                data=result,
            )
        except ValueError as exc:
            return error_response(404, str(exc))

    @router.post("/api/role/import/")
    async def import_role(
        name: str = Form(...),
        description: str = Form(default=""),
        world_id: int | None = Form(default=None),
        world_name: str | None = Form(default=None),
        version: str | None = Form(default=None),
        prompt_text: str | None = Form(default=None),
        language: str | None = Form(default="zh"),
        gpt_file: UploadFile = File(...),
        sov_file: UploadFile = File(...),
        prompt_audio: UploadFile | None = File(default=None),
    ):
        payload = RoleCreateRequest(
            name=name,
            description=description,
            world_id=world_id,
            world_name=world_name,
            version=version,
            prompt_text=prompt_text,
            language=language,
        )
        try:
            role = service.import_role(payload, gpt_file=gpt_file, sov_file=sov_file, prompt_audio=prompt_audio)
            return RoleMutationResponse(
                message="角色导入成功",
                data={"id": role.id, "name": role.name},
            )
        except ValueError as exc:
            return error_response(400, str(exc))

    @router.post("/api/role/workspace/create/")
    async def create_role_workspace(payload: RoleWorkspaceCreateRequest):
        try:
            result = service.create_role_workspace(payload)
            return RoleWorkspaceMutationResponse(
                message="角色工作区创建成功",
                data=result,
            )
        except ValueError as exc:
            return error_response(400, str(exc))

    @router.get("/api/role/workspace/list/")
    async def list_role_workspaces():
        workspaces = service.list_role_workspaces()
        return RoleWorkspaceListResponse(
            workspaces=workspaces,
            count=len(workspaces),
        )

    @router.post("/api/role/workspace/upload-audio/")
    async def upload_role_audio(
        role_name: str = Form(...),
        target: str = Form(default="raw"),
        create_if_missing: bool = Form(default=True),
        files: list[UploadFile] = File(...),
    ):
        try:
            result = service.upload_role_audios(
                role_name=role_name,
                uploads=files,
                target=target,
                create_if_missing=create_if_missing,
            )
            return RoleWorkspaceMutationResponse(
                message="角色音频上传成功",
                data=result,
            )
        except ValueError as exc:
            return error_response(400, str(exc))

    @router.post("/api/role/workspace/initialize/")
    async def initialize_role_workspace(
        role_name: str = Form(...),
        description: str = Form(default=""),
        world_name: str | None = Form(default=None),
        language: str | None = Form(default="zh-CN"),
        version: str | None = Form(default=None),
        base_version: str | None = Form(default=None),
        experiment_name: str | None = Form(default=None),
        overwrite: bool = Form(default=False),
        raw_files: list[UploadFile] | None = File(default=None),
        prompt_files: list[UploadFile] | None = File(default=None),
    ):
        payload = RoleWorkspaceCreateRequest(
            role_name=role_name,
            description=description,
            world_name=world_name,
            language=language,
            version=version,
            base_version=base_version,
            experiment_name=experiment_name,
            overwrite=overwrite,
        )
        try:
            result = service.initialize_role_workspace(
                payload=payload,
                raw_uploads=raw_files or [],
                prompt_uploads=prompt_files or [],
            )
            return RoleWorkspaceInitializeResponse(
                message="角色彻底初始化成功",
                data=result,
            )
        except ValueError as exc:
            return error_response(400, str(exc))

    return router
