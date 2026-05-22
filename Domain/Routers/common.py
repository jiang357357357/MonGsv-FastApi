"""
Shared helpers for domain routers.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import UploadFile
from fastapi.responses import JSONResponse

from Code.FastApi.Base.monconfig import MonConfig


def error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "message": message},
    )


def save_import_audio(upload: UploadFile) -> str:
    config = MonConfig(start_path=Path(__file__).resolve())
    workspace_root = config.workspace_root() or Path.cwd()
    target_dir = (workspace_root / "Data" / "Temp" / "RoleImports").resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / upload.filename

    with target_path.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)

    return str(target_path)

