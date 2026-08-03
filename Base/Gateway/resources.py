"""训练工程与成品模型的资源目录规则。"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict

from Code.FastApi.Base.monconfig import MonConfig


def resources_root() -> Path:
    config = MonConfig(start_path=Path(__file__).resolve())
    workspace_root = config.workspace_root() or Path.cwd()
    root = (workspace_root / "Resources").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def normalize_path_segment(value: str) -> str:
    normalized = (value or "").strip()
    if not normalized:
        raise ValueError("路径段不能为空")
    invalid = set('\\/:*?"<>|')
    if any(ch in invalid for ch in normalized) or normalized in {".", ".."}:
        raise ValueError(f"路径段包含非法字符: {value}")
    return normalized


def sync_directory_files(source_dir: Path, target_dir: Path) -> list[str]:
    if not source_dir.exists() or not source_dir.is_dir():
        return []

    target_dir.mkdir(parents=True, exist_ok=True)
    for child in target_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    copied_files: list[str] = []
    for source_path in sorted(source_dir.iterdir(), key=lambda item: item.name.lower()):
        if not source_path.is_file():
            continue
        target_path = target_dir / source_path.name
        shutil.copy2(source_path, target_path)
        copied_files.append(str(target_path))
    return copied_files


def resource_layout(
    project_name: str,
    version: str,
    world_name: str = "Standalone",
    experiment_name: str = "",
) -> Dict[str, str]:
    role_name = normalize_path_segment(project_name)
    world = normalize_path_segment(world_name or "Standalone")
    base_version = normalize_path_segment(version or "v2Pro")

    root = resources_root()
    model_root = root / "Model" / world / role_name / base_version
    train_root = root / "Train" / "Projects" / world / role_name / base_version
    dataset_root = train_root / "dataset"

    return {
        "role_name": role_name,
        "world_name": world,
        "base_version": base_version,
        "experiment_name": "",
        "model_root": str(model_root),
        "train_root": str(train_root),
        "train_root_parent": str(train_root.parent),
        "dataset_root": str(dataset_root),
        "model_sliced_dir": str(model_root / "sliced"),
        "gpt_model_dir": str(model_root / "GPT"),
        "sovits_model_dir": str(model_root / "SoVITS"),
    }
