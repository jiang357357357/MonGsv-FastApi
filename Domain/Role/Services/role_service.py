"""
Role domain service.
"""

from __future__ import annotations

import json
import re
import shutil
import zlib
from pathlib import Path
from typing import List, NamedTuple

from fastapi import UploadFile

from Code.FastApi.Base.monconfig import MonConfig
from Code.FastApi.Domain.Model.Services import ModelCatalogService
from Code.FastApi.Domain.Role.Models.requests import (
    RoleCreateRequest,
    RoleUpdateRequest,
    RoleWorkspaceCreateRequest,
)
from Code.FastApi.Domain.Role.Models.role_info import RoleInfo
from Code.FastApi.Domain.World.Services.world_service import WorldService


class RoleService:
    """CRUD service for role resources."""

    AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}

    def __init__(self):
        self._world_service = WorldService()
        self._model_service = ModelCatalogService()
        self._config = MonConfig(start_path=Path(__file__).resolve())

    def list_roles(
        self,
        version: str | None = None,
        world_id: int | None = None,
        world_name: str | None = None,
    ) -> List[RoleInfo]:
        items = self._scan_role_items()
        if version:
            items = [item for item in items if item.version == version]
        if world_id is not None:
            items = [item for item in items if item.world_id == world_id]
        if world_name:
            world_name_lower = world_name.strip().lower()
            items = [
                item for item in items
                if str(item.world_name or "").strip().lower() == world_name_lower
            ]
        return sorted(items, key=lambda item: ((item.world_name or "").lower(), item.name.lower(), item.version or "", item.id))

    def get_role(self, role_id: int) -> RoleInfo:
        entry = self._require_role_entry(role_id)
        return entry.info

    def list_role_workspaces(self) -> list[dict]:
        model_root = self._model_root()
        workspaces: list[dict] = []
        if not model_root.exists():
            return workspaces
        for role_model_root in sorted(model_root.glob("*/*/*"), key=lambda item: str(item).lower()):
            if not role_model_root.is_dir():
                continue
            world_name, role_name, base_version = role_model_root.parts[-3:]
            train_root = self._latest_train_root(world_name, role_name, base_version)
            model_sliced_dir = role_model_root / "sliced"
            raw_dir = train_root / "source" / "raw" if train_root else None
            prompt_dir = train_root / "source" / "prompt" if train_root else None
            sliced_dir = train_root / "dataset" / "sliced" if train_root else None
            gpt_weights_dir = role_model_root / "GPT"
            sovits_weights_dir = role_model_root / "SoVITS"
            model_sliced_files = self._list_files(model_sliced_dir)
            sliced_files = self._list_files(sliced_dir)
            raw_files = self._list_files(raw_dir)
            prompt_files = self._list_files(prompt_dir)
            gpt_models = self._list_files(gpt_weights_dir, {".ckpt"})
            sovits_models = self._list_files(sovits_weights_dir, {".pth"})
            print(
                "[role-workspace] "
                f"world={world_name} role={role_name} version={base_version} "
                f"model_sliced={len(model_sliced_files)} train_sliced={len(sliced_files)} "
                f"model_dir={model_sliced_dir} train_dir={sliced_dir or ''}",
                flush=True,
            )
            workspaces.append(
                {
                    "role_name": role_name,
                    "world_name": world_name,
                    "base_version": base_version,
                    "experiment_name": None,
                    "role_root": str(role_model_root),
                    "model_root": str(role_model_root),
                    "train_root": str(train_root) if train_root else None,
                    "model_sliced_dir": str(model_sliced_dir),
                    "raw_dir": str(raw_dir) if raw_dir is not None else "",
                    "sliced_dir": str(sliced_dir) if sliced_dir is not None else "",
                    "raw_files": raw_files,
                    "model_sliced_files": model_sliced_files,
                    "sliced_files": sliced_files,
                    "prompt_dir": str(prompt_dir) if prompt_dir is not None else "",
                    "prompt_files": prompt_files,
                    "gpt_models": gpt_models,
                    "sovits_models": sovits_models,
                }
            )
        return workspaces

    def create_role(self, payload: RoleCreateRequest) -> RoleInfo:
        role = self._build_role_info(payload=payload)
        existing = self._find_role_entry(role.id)
        if existing is not None:
            raise ValueError(f"角色已存在: {role.name}")

        workspace = self.create_role_workspace(
            RoleWorkspaceCreateRequest(
                role_name=role.name,
                description=role.description,
                world_name=role.world_name,
                language=role.language or "zh",
                version=role.version,
                base_version=role.version,
                experiment_name=None,
                overwrite=False,
            )
        )
        self._write_role_metadata(Path(workspace["model_root"]), role)
        return role

    def update_role(self, payload: RoleUpdateRequest) -> RoleInfo:
        existing = self._find_role_entry(payload.id)
        if existing is None:
            raise ValueError(f"角色不存在: {payload.id}")

        updated = self._build_role_info(payload=payload)
        collision = self._find_role_entry(updated.id)
        if collision is not None and collision.info.id != existing.info.id:
            raise ValueError(f"目标角色已存在: {updated.name}")

        if (
            existing.info.world_name != updated.world_name
            or existing.info.name != updated.name
            or existing.info.version != updated.version
        ):
            self._move_role_directories(existing, updated)
            refreshed = self._find_role_entry(updated.id)
            if refreshed is None:
                raise ValueError("角色目录迁移失败")
            existing = refreshed

        self._write_role_metadata(existing.model_root, updated)
        return updated

    def delete_role(self, role_id: int) -> RoleInfo:
        entry = self._find_role_entry(role_id)
        if entry is None:
            raise ValueError(f"角色不存在: {role_id}")

        self._remove_tree(entry.model_base_root)
        self._remove_tree(entry.train_base_root)
        self._prune_empty_parents(entry.model_base_root, self._model_root() / (entry.info.world_name or ""))
        self._prune_empty_parents(entry.train_base_root, self._train_projects_root())
        return entry.info

    def list_role_emotions(self, role_id: int) -> list[dict]:
        entry = self._require_role_entry(role_id)
        emotion_dir = entry.model_root / "emotion"
        emotions: list[dict] = []
        if not emotion_dir.exists():
            return emotions

        for path in sorted(emotion_dir.iterdir(), key=lambda item: item.name.lower()):
            if not path.is_file() or path.suffix.lower() not in self.AUDIO_SUFFIXES:
                continue
            parsed = self._parse_emotion_filename(path)
            if parsed is None:
                continue
            emotions.append(
                {
                    "name": parsed["name"],
                    "text": parsed["text"],
                    "music_url": str(path),
                    "text_language": parsed["language"],
                    "file_name": path.name,
                }
            )
        return emotions

    def upsert_role_emotion(
        self,
        role_id: int,
        emotion_name: str,
        emotion_text: str,
        audio_filename: str | None = None,
        audio_content: bytes | None = None,
        text_language: str | None = None,
        audio_source_path: str | None = None,
    ) -> dict:
        entry = self._require_role_entry(role_id)
        normalized_name = self._normalize_emotion_value(emotion_name, "情感名称")
        normalized_text = self._normalize_emotion_value(emotion_text, "参考文本")
        normalized_language = self._normalize_language_code(text_language)
        emotion_dir = entry.model_root / "emotion"
        emotion_dir.mkdir(parents=True, exist_ok=True)

        existing = self._find_emotion_file(emotion_dir, normalized_name)
        source_path = self._resolve_role_audio_source(entry.model_root, entry.train_root, audio_source_path)
        source_name = audio_filename or (source_path.name if source_path else existing.name if existing else "audio.wav")
        suffix = Path(source_name).suffix.lower() or ".wav"
        if suffix not in self.AUDIO_SUFFIXES:
            raise ValueError(f"不支持的音频格式: {suffix}")

        target_name = self._emotion_filename(normalized_name, normalized_language, normalized_text, suffix)
        target_path = emotion_dir / target_name

        if audio_content is not None:
            target_path.write_bytes(audio_content)
            if existing and existing.resolve() != target_path.resolve():
                existing.unlink(missing_ok=True)
        elif source_path is not None:
            if source_path.resolve() != target_path.resolve():
                shutil.copy2(source_path, target_path)
            if existing and existing.resolve() != target_path.resolve():
                existing.unlink(missing_ok=True)
        elif existing:
            if existing.resolve() != target_path.resolve():
                existing.rename(target_path)
        else:
            raise ValueError("新增情感必须上传参考音频或选择切分音频")

        updated_role = entry.info.copy(update={
            "prompt_text": normalized_text,
            "prompt_audio_path": str(target_path),
            "language": normalized_language,
        })
        self._write_role_metadata(entry.model_root, updated_role)
        return {
            "emotion": normalized_name,
            "text": normalized_text,
            "audio_file": str(target_path),
            "file_name": target_path.name,
            "emotions": self.list_role_emotions(role_id),
        }

    def delete_role_emotion(self, role_id: int, emotion_name: str) -> dict:
        entry = self._require_role_entry(role_id)
        normalized_name = self._normalize_emotion_value(emotion_name, "情感名称")
        emotion_dir = entry.model_root / "emotion"
        target = self._find_emotion_file(emotion_dir, normalized_name)
        if target is None:
            raise ValueError(f"情感不存在: {emotion_name}")
        target.unlink()

        emotions = self.list_role_emotions(role_id)
        next_prompt = emotions[0] if emotions else None
        updated_role = entry.info.copy(update={
            "prompt_text": next_prompt["text"] if next_prompt else None,
            "prompt_audio_path": next_prompt["music_url"] if next_prompt else None,
        })
        self._write_role_metadata(entry.model_root, updated_role)
        return {
            "deleted_emotion": normalized_name,
            "emotions": emotions,
        }

    def import_role(
        self,
        payload: RoleCreateRequest,
        gpt_file: UploadFile,
        sov_file: UploadFile,
        prompt_audio: UploadFile | None = None,
    ) -> RoleInfo:
        role = self.create_role(payload)
        entry = self._find_role_entry(role.id)
        if entry is None:
            return role

        try:
            gpt_path = self._store_upload_file(
                entry.model_root / "GPT",
                gpt_file,
                ModelCatalogService.GPT_EXTENSIONS,
            )
            sov_path = self._store_upload_file(
                entry.model_root / "SoVITS",
                sov_file,
                ModelCatalogService.SOVITS_EXTENSIONS,
            )
            prompt_audio_path = None
            if prompt_audio is not None and prompt_audio.filename:
                prompt_audio_path = self._store_upload_file(
                    entry.model_root / "emotion",
                    prompt_audio,
                    self.AUDIO_SUFFIXES,
                )

            updated_role = role.copy(update={
                "gpt_model_id": self._resolve_model_id_by_path(gpt_path, "gpt"),
                "gpt_model_name": Path(gpt_path).name,
                "gpt_model_path": gpt_path,
                "sov_model_id": self._resolve_model_id_by_path(sov_path, "sovits"),
                "sov_model_name": Path(sov_path).name,
                "sov_model_path": sov_path,
                "prompt_audio_path": prompt_audio_path or role.prompt_audio_path,
            })
            self._write_role_metadata(entry.model_root, updated_role)
            return updated_role
        except Exception:
            self._remove_tree(entry.model_base_root)
            self._remove_tree(entry.train_base_root)
            self._prune_empty_parents(entry.model_base_root, self._model_root() / (entry.info.world_name or ""))
            self._prune_empty_parents(entry.train_base_root, self._train_projects_root())
            raise

    def create_role_workspace(self, payload: RoleWorkspaceCreateRequest) -> dict:
        spec = self._workspace_spec(payload)
        role_name = spec.role_name
        model_root = spec.model_root
        train_root = spec.train_root
        if (model_root.exists() and any(model_root.iterdir()) or train_root.exists() and any(train_root.iterdir())) and not payload.overwrite:
            raise ValueError(f"角色目录已存在: model={model_root}, train={train_root}")
        directories = [
            model_root / "GPT",
            model_root / "SoVITS",
            model_root / "meta",
            model_root / "emotion",
            model_root / "sliced",
            model_root / "config",
            train_root / "source" / "raw",
            train_root / "source" / "cleaned",
            train_root / "source" / "prompt",
            train_root / "dataset" / "sliced",
            train_root / "dataset" / "asr",
            train_root / "dataset" / "3-bert",
            train_root / "dataset" / "4-cnhubert",
            train_root / "dataset" / "5-wav32k",
            train_root / "dataset" / "7-sv_cn",
            train_root / "train" / "sovits",
            train_root / "train" / "gpt",
            train_root / "train" / "combined",
            train_root / "infer" / "outputs",
            train_root / "infer" / "cache",
            train_root / "config",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        role_meta = {
            "role_name": role_name,
            "display_name": role_name,
            "world_name": spec.world_name,
            "language": payload.language or "zh-CN",
            "description": payload.description,
            "base_version": spec.base_version,
        }
        self._write_json(model_root / "meta" / "role.json", role_meta)
        self._write_text(
            model_root / "meta" / "README.md",
            "# Meta\n\nStore role metadata here.\n",
        )
        self._write_text(
            train_root / "dataset" / "2-name2text.txt",
            "# format:\n# wav_name|speaker_name|language|text\n",
        )
        self._write_text(
            train_root / "dataset" / "6-name2semantic.tsv",
            "# format:\n# item_name<TAB>semantic_tokens\n",
        )

        return {
            "role_name": role_name,
            "world_name": spec.world_name,
            "base_version": spec.base_version,
            "experiment_name": spec.experiment_name,
            "role_root": str(model_root),
            "model_root": str(model_root),
            "train_root": str(train_root),
            "model_sliced_dir": str(model_root / "sliced"),
            "created_directories": [str(path) for path in directories],
        }

    def upload_role_audios(
        self,
        role_name: str,
        uploads: list[UploadFile],
        target: str = "raw",
        create_if_missing: bool = True,
    ) -> dict:
        normalized_name = self._normalize_role_name(role_name)
        train_root = self._latest_train_root_by_role(normalized_name)
        if train_root is None:
            if not create_if_missing:
                raise ValueError(f"角色目录不存在: {normalized_name}")
            self.create_role_workspace(RoleWorkspaceCreateRequest(role_name=normalized_name))
            train_root = self._latest_train_root_by_role(normalized_name)
        if train_root is None:
            raise ValueError(f"无法定位角色训练目录: {normalized_name}")
        target_dir = self._resolve_audio_target(train_root, target)
        target_dir.mkdir(parents=True, exist_ok=True)

        saved_files: list[str] = []
        for upload in uploads:
            filename = self._sanitize_filename(upload.filename or "upload.bin")
            target_path = target_dir / filename
            with target_path.open("wb") as handle:
                shutil.copyfileobj(upload.file, handle)
            saved_files.append(str(target_path))

        return {
            "role_name": normalized_name,
            "role_root": str(train_root),
            "target": target,
            "saved_files": saved_files,
            "count": len(saved_files),
        }

    def initialize_role_workspace(
        self,
        payload: RoleWorkspaceCreateRequest,
        raw_uploads: list[UploadFile] | None = None,
        prompt_uploads: list[UploadFile] | None = None,
    ) -> dict:
        workspace = self.create_role_workspace(payload)
        uploaded: dict[str, dict] = {}

        if raw_uploads:
            uploaded["raw"] = self.upload_role_audios(
                role_name=payload.role_name,
                uploads=raw_uploads,
                target="raw",
                create_if_missing=True,
            )

        if prompt_uploads:
            uploaded["prompt"] = self.upload_role_audios(
                role_name=payload.role_name,
                uploads=prompt_uploads,
                target="prompt",
                create_if_missing=True,
            )

        return {
            "role_name": workspace["role_name"],
            "world_name": workspace["world_name"],
            "base_version": workspace["base_version"],
            "experiment_name": workspace["experiment_name"],
            "role_root": workspace["role_root"],
            "model_root": workspace["model_root"],
            "train_root": workspace["train_root"],
            "created_directories": workspace["created_directories"],
            "uploaded": uploaded,
        }

    def _build_role_info(self, payload: RoleCreateRequest) -> RoleInfo:
        world_name = payload.world_name
        resolved_version = payload.version.strip() if payload.version else None
        if payload.world_id is not None:
            world = self._find_world_by_id(payload.world_id)
            world_name = world.name
            if world.version and resolved_version and world.version != resolved_version:
                raise ValueError(
                    f"角色版本与世界版本不一致: role={resolved_version}, world={world.version}"
                )
            if world.version and not resolved_version:
                resolved_version = world.version

        gpt_model_name = payload.gpt_model_name
        gpt_model_path = payload.gpt_model_path
        if payload.gpt_model_id is not None:
            gpt_model = self._model_service.get_gpt_model(payload.gpt_model_id)
            gpt_model_name = gpt_model.name
            gpt_model_path = gpt_model.path
            if gpt_model.version and resolved_version and gpt_model.version != resolved_version:
                raise ValueError(
                    f"GPT 模型版本与角色版本不一致: model={gpt_model.version}, role={resolved_version}"
                )
            if gpt_model.version and not resolved_version:
                resolved_version = gpt_model.version

        sov_model_name = payload.sov_model_name
        sov_model_path = payload.sov_model_path
        if payload.sov_model_id is not None:
            sov_model = self._model_service.get_sovits_model(payload.sov_model_id)
            sov_model_name = sov_model.name
            sov_model_path = sov_model.path
            if sov_model.version and resolved_version and sov_model.version != resolved_version:
                raise ValueError(
                    f"SoVITS 模型版本与角色版本不一致: model={sov_model.version}, role={resolved_version}"
                )
            if sov_model.version and not resolved_version:
                resolved_version = sov_model.version

        normalized_name = self._normalize_role_name(payload.name)
        normalized_world_name = self._normalize_segment(world_name or "Standalone")
        normalized_version = self._normalize_segment(resolved_version or "v2ProPlus")

        return RoleInfo(
            id=self._role_id(normalized_world_name, normalized_name, normalized_version),
            name=normalized_name,
            description=payload.description.strip(),
            world_id=self._world_id(normalized_world_name),
            world_name=normalized_world_name,
            version=normalized_version,
            gpt_model_id=payload.gpt_model_id,
            gpt_model_name=gpt_model_name,
            sov_model_id=payload.sov_model_id,
            sov_model_name=sov_model_name,
            gpt_model_path=gpt_model_path,
            sov_model_path=sov_model_path,
            prompt_text=payload.prompt_text,
            prompt_audio_path=payload.prompt_audio_path,
            language=payload.language or "zh",
        )

    def _find_world_by_id(self, world_id: int):
        for world in self._world_service.list_worlds():
            if world.id == world_id:
                return world
        raise ValueError(f"世界不存在: {world_id}")

    def _resources_root(self) -> Path:
        workspace_root = self._config.workspace_root() or Path.cwd()
        resources_root = (workspace_root / "Resources").resolve()
        resources_root.mkdir(parents=True, exist_ok=True)
        return resources_root

    def _model_root(self) -> Path:
        root = (self._resources_root() / "Model").resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _train_projects_root(self) -> Path:
        root = (self._resources_root() / "Train" / "Projects").resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _scan_role_items(self) -> list[RoleInfo]:
        return [entry.info for entry in self._scan_role_entries()]

    def _scan_role_entries(self) -> list["ScannedRoleEntry"]:
        entries: list[ScannedRoleEntry] = []
        model_root = self._model_root()
        if not model_root.exists():
            return entries

        for role_model_root in sorted(model_root.glob("*/*/*"), key=lambda item: str(item).lower()):
            if not role_model_root.is_dir():
                continue

            world_name, role_name, base_version = role_model_root.parts[-3:]
            world_id = self._world_id(world_name)
            meta = self._read_role_meta(role_model_root / "meta" / "role.json")
            gpt_model_path = self._latest_file(role_model_root / "GPT", {".ckpt"})
            sov_model_path = self._latest_file(role_model_root / "SoVITS", {".pth"})
            resolved_gpt_model_id = self._resolve_model_id_by_path(gpt_model_path, "gpt")
            resolved_sov_model_id = self._resolve_model_id_by_path(sov_model_path, "sovits")
            prompt_audio_path = self._latest_file(role_model_root / "emotion", {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"})
            train_root = self._latest_train_root(world_name, role_name, base_version)
            info = RoleInfo(
                id=self._role_id(world_name, role_name, base_version),
                name=meta.get("display_name") or meta.get("role_name") or role_name,
                description=str(meta.get("description") or ""),
                world_id=world_id,
                world_name=world_name,
                version=base_version,
                gpt_model_id=resolved_gpt_model_id or meta.get("gpt_model_id"),
                gpt_model_name=meta.get("gpt_model_name") or (Path(gpt_model_path).name if gpt_model_path else None),
                sov_model_id=resolved_sov_model_id or meta.get("sov_model_id"),
                sov_model_name=meta.get("sov_model_name") or (Path(sov_model_path).name if sov_model_path else None),
                gpt_model_path=gpt_model_path or meta.get("gpt_model_path"),
                sov_model_path=sov_model_path or meta.get("sov_model_path"),
                prompt_text=meta.get("prompt_text"),
                prompt_audio_path=prompt_audio_path or meta.get("prompt_audio_path"),
                language=str(meta.get("language") or "zh"),
            )
            entries.append(
                ScannedRoleEntry(
                    info=info,
                    model_root=role_model_root,
                    model_base_root=role_model_root,
                    train_root=train_root,
                    train_base_root=(self._train_projects_root() / world_name / role_name / base_version),
                )
            )

        return entries

    def _find_role_entry(self, role_id: int) -> "ScannedRoleEntry | None":
        for entry in self._scan_role_entries():
            if entry.info.id == role_id:
                return entry
        return None

    def _require_role_entry(self, role_id: int) -> "ScannedRoleEntry":
        entry = self._find_role_entry(role_id)
        if entry is None:
            raise ValueError(f"角色不存在: {role_id}")
        return entry

    @staticmethod
    def _read_role_meta(path: Path) -> dict:
        if not path.exists() or not path.is_file():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _latest_file(directory: Path, allowed_suffixes: set[str]) -> str | None:
        if not directory.exists() or not directory.is_dir():
            return None
        normalized_suffixes = {suffix.lower() for suffix in allowed_suffixes}
        candidates = [
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in normalized_suffixes
        ]
        if not candidates:
            return None

        def sort_key(path: Path) -> tuple[int, float, str]:
            epoch_match = re.search(r"(?:^|[-_])e(\d+)(?:[-_.]|$)", path.stem, re.IGNORECASE)
            epoch = int(epoch_match.group(1)) if epoch_match else -1
            return epoch, path.stat().st_mtime, path.name.lower()

        return str(max(candidates, key=sort_key))

    def _resolve_model_id_by_path(self, model_path: str | None, model_type: str) -> int | None:
        if not model_path:
            return None
        try:
            target = Path(model_path).resolve()
        except Exception:
            return None

        models = (
            self._model_service.list_gpt_models()
            if model_type == "gpt"
            else self._model_service.list_sovits_models()
        )
        for model in models:
            if not model.path:
                continue
            try:
                if Path(model.path).resolve() == target:
                    return model.id
            except Exception:
                continue
        return None

    @staticmethod
    def _world_id(world_name: str) -> int:
        return zlib.crc32(world_name.strip().lower().encode("utf-8")) & 0x7FFFFFFF

    @staticmethod
    def _role_id(world_name: str, role_name: str, base_version: str) -> int:
        token = f"{world_name.strip().lower()}::{role_name.strip().lower()}::{base_version.strip().lower()}"
        return zlib.crc32(token.encode("utf-8")) & 0x7FFFFFFF

    @staticmethod
    def _normalize_role_name(role_name: str) -> str:
        normalized = role_name.strip()
        if not normalized:
            raise ValueError("角色名不能为空")
        if normalized in {".", ".."}:
            raise ValueError("角色名无效")
        if re.search(r'[\\/:*?"<>|]', normalized):
            raise ValueError(f"角色名包含非法字符: {role_name}")
        return normalized

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        sanitized = re.sub(r'[\\/:*?"<>|]+', "_", filename).strip()
        if not sanitized:
            raise ValueError("文件名无效")
        return sanitized

    @staticmethod
    def _normalize_emotion_value(value: str, label: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{label}不能为空")
        return normalized

    @staticmethod
    def _normalize_language_code(value: str | None) -> str:
        normalized = (value or "zh").strip().lower()
        if not normalized:
            raise ValueError("语言代码不能为空")
        if not re.fullmatch(r"[a-z][a-z0-9_-]{1,15}", normalized):
            raise ValueError(f"语言代码无效: {value}")
        return normalized

    def _emotion_filename(self, emotion_name: str, language: str, emotion_text: str, suffix: str) -> str:
        return self._sanitize_filename(f"[{emotion_name}][{language}]{emotion_text}{suffix}")

    @staticmethod
    def _parse_emotion_filename(path: Path) -> dict | None:
        stem = path.stem
        match = re.match(r"^\[(?P<emotion>[^\]]+)\]\[(?P<language>[^\]]+)\](?P<text>.+)$", stem)
        if not match:
            return None
        return {
            "name": match.group("emotion").strip(),
            "language": match.group("language").strip().lower(),
            "text": match.group("text").strip(),
        }

    def _find_emotion_file(self, emotion_dir: Path, emotion_name: str) -> Path | None:
        if not emotion_dir.exists():
            return None
        normalized_name = emotion_name.strip().lower()
        for path in sorted(emotion_dir.iterdir(), key=lambda item: item.name.lower()):
            if not path.is_file() or path.suffix.lower() not in self.AUDIO_SUFFIXES:
                continue
            parsed = self._parse_emotion_filename(path)
            if parsed is None:
                continue
            if parsed["name"].strip().lower() == normalized_name:
                return path
        return None

    def _resolve_role_audio_source(
        self,
        model_root: Path,
        train_root: Path | None,
        audio_source_path: str | None,
    ) -> Path | None:
        raw_path = (audio_source_path or "").strip()
        if not raw_path:
            return None

        source_path = Path(raw_path).expanduser().resolve()
        allowed_roots = [(model_root / "sliced").resolve()]
        if train_root is not None:
            allowed_roots.append((train_root / "dataset" / "sliced").resolve())
        if not source_path.exists() or not source_path.is_file():
            print(
                "[role-emotion] 切分音频不存在: "
                f"path={audio_source_path} allowed_roots={[str(root) for root in allowed_roots]}",
                flush=True,
            )
            raise ValueError(f"切分音频不存在: {audio_source_path}")
        if source_path.suffix.lower() not in self.AUDIO_SUFFIXES:
            print(
                "[role-emotion] 不支持的切分音频格式: "
                f"path={source_path} suffix={source_path.suffix}",
                flush=True,
            )
            raise ValueError(f"不支持的切分音频格式: {source_path.suffix}")
        if not any(source_path.parent == root or root in source_path.parents for root in allowed_roots):
            print(
                "[role-emotion] 切分音频目录不匹配: "
                f"path={source_path} allowed_roots={[str(root) for root in allowed_roots]}",
                flush=True,
            )
            raise ValueError("选择的切分音频不在当前角色的模型 sliced 目录或训练 dataset/sliced 目录中")

        return source_path

    @staticmethod
    def _list_files(directory: Path | None, allowed_suffixes: set[str] | None = None) -> list[str]:
        if directory is None or not directory.exists():
            return []
        suffixes = {item.lower() for item in allowed_suffixes} if allowed_suffixes else None
        files: list[str] = []
        for path in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
            if not path.is_file():
                continue
            if suffixes and path.suffix.lower() not in suffixes:
                continue
            files.append(str(path))
        return files

    @staticmethod
    def _resolve_audio_target(train_root: Path, target: str) -> Path:
        target_key = (target or "raw").strip().lower()
        target_map = {
            "raw": train_root / "source" / "raw",
            "cleaned": train_root / "source" / "cleaned",
            "prompt": train_root / "source" / "prompt",
        }
        if target_key not in target_map:
            raise ValueError(f"不支持的上传目标: {target}")
        return target_map[target_key]

    def _latest_train_root(self, world_name: str, role_name: str, base_version: str) -> Path | None:
        base_root = self._train_projects_root() / world_name / role_name / base_version
        return base_root if base_root.exists() and base_root.is_dir() else None

    def _latest_train_root_by_role(self, role_name: str) -> Path | None:
        train_root = self._train_projects_root()
        matches = [path for path in train_root.glob(f"*/{role_name}/*") if path.is_dir()]
        matches.sort(key=lambda p: str(p).lower())
        return matches[-1] if matches else None

    def _workspace_spec(self, payload: RoleWorkspaceCreateRequest) -> "WorkspaceSpec":
        role_name = self._normalize_role_name(payload.role_name)
        world_name = self._normalize_segment(payload.world_name or "Standalone")
        base_version = self._normalize_segment(payload.base_version or payload.version or "v2ProPlus")
        model_root = self._model_root() / world_name / role_name / base_version
        train_root = self._train_projects_root() / world_name / role_name / base_version
        return WorkspaceSpec(
            role_name=role_name,
            world_name=world_name,
            base_version=base_version,
            experiment_name="",
            model_root=model_root,
            train_root=train_root,
        )

    @staticmethod
    def _normalize_segment(value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("目录段不能为空")
        if normalized in {".", ".."}:
            raise ValueError("目录段无效")
        if re.search(r'[\\/:*?"<>|]', normalized):
            raise ValueError(f"目录段包含非法字符: {value}")
        return normalized

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    @staticmethod
    def _write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write(content)

    def _write_role_metadata(self, model_root: Path, role: RoleInfo) -> None:
        role_meta = {
            "role_name": role.name,
            "display_name": role.name,
            "world_name": role.world_name,
            "language": role.language or "zh",
            "description": role.description,
            "base_version": role.version,
            "gpt_model_id": role.gpt_model_id,
            "gpt_model_name": role.gpt_model_name,
            "gpt_model_path": role.gpt_model_path,
            "sov_model_id": role.sov_model_id,
            "sov_model_name": role.sov_model_name,
            "sov_model_path": role.sov_model_path,
            "prompt_text": role.prompt_text,
            "prompt_audio_path": role.prompt_audio_path,
        }
        self._write_json(model_root / "meta" / "role.json", role_meta)

    def _store_upload_file(
        self,
        target_dir: Path,
        upload: UploadFile,
        allowed_suffixes: set[str],
    ) -> str:
        filename = self._sanitize_filename(upload.filename or "upload.bin")
        suffix = Path(filename).suffix.lower()
        normalized_suffixes = {item.lower() for item in allowed_suffixes}
        if suffix not in normalized_suffixes:
            raise ValueError(f"不支持的文件格式: {suffix or '<none>'}")

        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / filename
        upload.file.seek(0)
        with target_path.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle)
        return str(target_path)

    def _move_role_directories(self, entry: "ScannedRoleEntry", updated: RoleInfo) -> None:
        target_model_base = self._model_root() / updated.world_name / updated.name / updated.version
        target_train_base = self._train_projects_root() / updated.world_name / updated.name / updated.version

        if target_model_base.exists() and target_model_base.resolve() != entry.model_base_root.resolve():
            raise ValueError(f"目标角色目录已存在: {target_model_base}")
        if target_train_base.exists() and target_train_base.resolve() != entry.train_base_root.resolve():
            raise ValueError(f"目标训练目录已存在: {target_train_base}")

        target_model_base.parent.mkdir(parents=True, exist_ok=True)
        if entry.model_base_root.exists() and entry.model_base_root.resolve() != target_model_base.resolve():
            shutil.move(str(entry.model_base_root), str(target_model_base))

        if entry.train_base_root.exists() and entry.train_base_root.resolve() != target_train_base.resolve():
            target_train_base.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(entry.train_base_root), str(target_train_base))

        self._prune_empty_parents(entry.model_base_root, self._model_root() / (entry.info.world_name or ""))
        self._prune_empty_parents(entry.train_base_root, self._train_projects_root())

    @staticmethod
    def _remove_tree(path: Path) -> None:
        if path.exists():
            shutil.rmtree(path)

    @staticmethod
    def _prune_empty_parents(path: Path, stop_at: Path) -> None:
        stop_at = stop_at.resolve()
        current = path.parent.resolve()
        while current != stop_at and stop_at in current.parents:
            try:
                next(current.iterdir())
                break
            except StopIteration:
                current.rmdir()
                current = current.parent.resolve()


class WorkspaceSpec(NamedTuple):
    role_name: str
    world_name: str
    base_version: str
    experiment_name: str
    model_root: Path
    train_root: Path


class ScannedRoleEntry(NamedTuple):
    info: RoleInfo
    model_root: Path
    model_base_root: Path
    train_root: Path | None
    train_base_root: Path
