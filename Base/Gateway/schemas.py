"""统一网关自身使用的请求与工作流数据模型。"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    gpu_numbers: str = "0"
    gpt_learning_rate: float = Field(default=0.01, gt=0, le=1)
    sovits_learning_rate: float = Field(default=0.0001, gt=0, le=1)
    save_every_epoch: Optional[int] = Field(default=None, ge=1)
    lora_rank: int = Field(default=32, ge=1, le=256)
    log_interval: int = Field(default=100, ge=1)
    grad_checkpoint: bool = False
    fp16: bool = True

    @field_validator("gpu_numbers")
    @classmethod
    def normalize_gpu_numbers(cls, value: str) -> str:
        import re
        numbers = re.split(r"[-,\s]+", value.strip())
        if not numbers or any(not number.isdigit() for number in numbers) or len(set(numbers)) != len(numbers):
            raise ValueError("GPU 编号应为 0 或 0,1 等不重复的整数")
        return "-".join(numbers)


class TrainingWorkbenchRequest(TrainingWorkflowOptions):
    model_config = ConfigDict(extra="forbid")
    save_every_epoch: int = Field(default=1, ge=1)
    grad_checkpoint: bool = True
    log_interval: int = Field(default=10, ge=1)
    project_name: str
    world_name: str
    version: str
    role_mode: str = "existing"
    preprocessing_mode: str = "cached"
    source_version: str = ""
    source_workspace: str = ""
    input_audio_dir: str = ""
    list_file: str = ""
    language: str = "zh"
    training_mode: str = "fresh"
    resume_workspace: str = ""
    publish_models: bool = True

    @field_validator("gpt_batch_size", "sovits_batch_size", "gpt_total_epoch", "sovits_total_epoch")
    @classmethod
    def positive_training_parameters(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Batch 和训练轮数必须至少为 1")
        return value

    @field_validator("version", "source_version")
    @classmethod
    def supported_version(cls, value: str) -> str:
        if value and value not in {"v1", "v2", "v3", "v4", "v2Pro", "v2ProPlus", "v5dev", "v5turbo"}:
            raise ValueError("不支持的模型版本")
        return value

    @field_validator("training_mode")
    @classmethod
    def supported_training_mode(cls, value: str) -> str:
        if value not in {"fresh", "resume"}:
            raise ValueError("训练方式应为 fresh 或 resume")
        return value

    @field_validator("preprocessing_mode")
    @classmethod
    def supported_preprocessing_mode(cls, value: str) -> str:
        if value not in {"full", "existing", "cached"}:
            raise ValueError("数据方式应为 full、existing 或 cached")
        return value

    @field_validator("role_mode")
    @classmethod
    def supported_role_mode(cls, value: str) -> str:
        if value not in {"new", "existing"}:
            raise ValueError("角色方式应为 new 或 existing")
        return value

    @field_validator("training_order")
    @classmethod
    def supported_training_order(cls, value: str) -> str:
        if value not in {"gpt_first", "sovits_first"}:
            raise ValueError("不支持的训练顺序")
        return value


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
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    speed: float = 1.0
    sample_steps: Optional[int] = None
    cfg_rate: Optional[float] = None
    if_sr: bool = False
    ref_free: bool = False
    if_freeze: bool = False
    pause_second: float = 0.3
    inference_mode: str = "normal"
    return_base64: bool = True
