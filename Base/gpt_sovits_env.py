"""
GPT-SoVITS environment helpers.
"""

import sys
from pathlib import Path
from typing import Iterable, Optional


def find_gpt_sovits_root(start: Path) -> Path:
    """Locate the GPT-SoVITS project root for the current workspace layout."""
    start = start.resolve()

    for parent in [start, *start.parents]:
        if _is_gpt_sovits_root(parent):
            return parent

    fallback_paths = [
        Path("文档/参考/GPT-SoVITS-main"),
        Path("文档/GPT-SoVITS-main"),
        Path("GPT-SoVITS-main"),
    ]
    for relative in fallback_paths:
        candidate = (start.parent / relative).resolve()
        if _is_gpt_sovits_root(candidate):
            return candidate

    raise FileNotFoundError("无法找到 GPT-SoVITS 项目根目录")


def setup_gpt_sovits_paths(start: Path, extra_paths: Optional[Iterable[Path]] = None) -> Path:
    """Inject the repo paths needed by GPT-SoVITS legacy imports."""
    root = find_gpt_sovits_root(start)
    gpt_dir = root / "GPT_SoVITS"

    paths = [
        root,
        gpt_dir,
        gpt_dir / "BigVGAN",
        gpt_dir / "eres2net",
    ]
    if extra_paths:
        paths.extend(extra_paths)

    for path in reversed(paths):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)

    return root


def _is_gpt_sovits_root(path: Path) -> bool:
    return (
        path.exists()
        and (path / "GPT_SoVITS").is_dir()
        and (path / "tools").is_dir()
        and (path / "config.py").is_file()
    )
