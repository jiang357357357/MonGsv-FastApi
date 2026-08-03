"""统一网关的运行时配置。"""

from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel


class UnifiedGatewayConfig(BaseModel):
    """从环境变量解析出的统一网关配置。"""

    enable_auth: bool = False
    api_key: Optional[str] = None
    max_concurrent_jobs: int = 10
    temp_dir: str = "temp"
    log_level: str = "INFO"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"true", "yes", "1", "on", "enabled"}


def build_runtime_config() -> UnifiedGatewayConfig:
    """读取网关进程环境变量，构造类型化配置。"""

    return UnifiedGatewayConfig(
        enable_auth=_env_bool("ENABLE_AUTH", False),
        api_key=os.environ.get("API_KEY"),
        max_concurrent_jobs=int(os.environ.get("MAX_CONCURRENT_JOBS", "10")),
        temp_dir=os.environ.get("TEMP_DIR", "temp"),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
    )
