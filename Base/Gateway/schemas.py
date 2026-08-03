"""统一网关自身使用的请求与工作流数据模型。"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class LocalDialogRequest(BaseModel):
    title: str = ""
    initial_dir: str = ""
    filetypes: list[tuple[str, str]] | None = None


class BatchProjectItem(BaseModel):
    name: str
    input_dir: str
    output_dir: str
    language: str = "zh"
    version: str = "v2Pro"


class BatchProjectsRequest(BaseModel):
    projects: List[BatchProjectItem]


class TrainingLaunchSummary(BaseModel):
    training_type: str
    success: bool
    message: str
    job_id: Optional[str] = None
    config_file: Optional[str] = None
    log_dir: Optional[str] = None
    model_dir: Optional[str] = None


class TrainingWorkflowOptions(BaseModel):
    start_training: bool = False
    train_gpt: bool = True
    train_sovits: bool = True
    gpt_batch_size: int = 8
    gpt_total_epoch: int = 15
    sovits_batch_size: int = 32
    sovits_total_epoch: int = 8
    training_order: str = "sovits_first"


class RoleEmotionSynthesisRequest(BaseModel):
    """业务合成请求：前端只提交角色、情感和文本。"""

    model_config = ConfigDict(extra="forbid")

    role_id: int
    emotion: str
    text: str
    text_language: str = "zh"
    world_id: Optional[int] = None
    version: Optional[str] = None
    how_to_cut: str = "按标点符号切"
    top_k: int = 20
    top_p: float = 0.6
    temperature: float = 0.6
    speed: float = 1.0
    sample_steps: int = 8
    if_sr: bool = False
    ref_free: bool = False
    if_freeze: bool = False
    pause_second: float = 0.3
    inference_mode: str = "normal"
    return_base64: bool = True
