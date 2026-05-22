#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS SoVITS训练 API 核心模块

提供SoVITS模型训练的核心功能和数据模型
"""

import os
import sys
import json
import asyncio
import tempfile
import shutil
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from subprocess import Popen, PIPE, STDOUT
import traceback
from datetime import datetime

from pydantic import BaseModel, Field


class SoVITSTrainingConfig(BaseModel):
    """SoVITS训练配置参数"""
    # 基础训练参数
    version: str = Field(default="v2Pro", description="模型版本: v1, v2, v3, v4, v2Pro, v2ProPlus")
    batch_size: int = Field(default=32, description="每张显卡的batch_size")
    total_epoch: int = Field(default=8, description="总训练轮数")
    save_every_epoch: int = Field(default=8, description="保存频率")
    
    # 学习率相关
    text_low_lr_rate: float = Field(default=0.4, description="文本模块学习率权重")
    learning_rate: float = Field(default=0.0001, description="基础学习率")
    lr_decay: float = Field(default=0.999875, description="学习率衰减")
    
    # 保存选项
    if_save_latest: bool = Field(default=True, description="是否仅保存最新的权重文件")
    if_save_every_weights: bool = Field(default=True, description="是否保存所有权重到weights文件夹")
    
    # GPU配置
    gpu_numbers: str = Field(default="0", description="GPU卡号，多卡用-分割")
    
    # 预训练模型路径
    pretrained_s2G: str = Field(default="", description="预训练SoVITS-G模型路径")
    pretrained_s2D: str = Field(default="", description="预训练SoVITS-D模型路径")
    
    # 高级选项
    if_grad_ckpt: bool = Field(default=False, description="是否开启梯度检查点节省显存")
    lora_rank: int = Field(default=32, description="LoRA秩(仅v3/v4)")
    fp16_run: bool = Field(default=True, description="是否使用半精度训练")
    
    # 损失权重
    c_mel: float = Field(default=45.0, description="Mel损失权重")
    c_kl: float = Field(default=1.0, description="KL损失权重")
    
    # 数据相关
    segment_size: int = Field(default=20480, description="音频片段大小")
    sampling_rate: int = Field(default=32000, description="采样率")


class SoVITSTrainingRequest(BaseModel):
    """SoVITS训练请求"""
    exp_name: str = Field(description="实验/模型名称")
    exp_root: str = Field(description="实验根目录路径")
    workspace_dir: str = Field(default="", description="显式训练工作区目录，优先级高于 exp_root/exp_name")
    model_output_dir: str = Field(default="", description="最终模型导出目录")
    config: SoVITSTrainingConfig = Field(default_factory=SoVITSTrainingConfig)


class SoVITSTrainingStatus(BaseModel):
    """训练状态"""
    job_id: str
    status: str  # "running", "completed", "failed", "stopped"
    current_epoch: int = 0
    total_epochs: int = 0
    current_loss: Optional[float] = None
    best_loss: Optional[float] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    log_file: Optional[str] = None


class SoVITSTrainingResponse(BaseModel):
    """SoVITS训练响应"""
    success: bool
    message: str
    job_id: Optional[str] = None
    status: Optional[SoVITSTrainingStatus] = None
    config_file: Optional[str] = None
    log_dir: Optional[str] = None
    model_dir: Optional[str] = None


class SoVITSTrainingService:
    """SoVITS训练API类"""
    
    def __init__(self, gpt_sovits_root: str = None):
        """
        初始化SoVITS训练API
        
        Args:
            gpt_sovits_root: GPT-SoVITS项目根目录路径
        """
        self.gpt_sovits_root = gpt_sovits_root or self._find_gpt_sovits_root()
        self.python_exec = sys.executable or "python"
        self.training_jobs: Dict[str, Dict] = {}  # 存储训练任务信息
        
        # 验证必要文件存在
        self.s2_train_script = os.path.join(self.gpt_sovits_root, "GPT_SoVITS", "s2_train.py")
        self.s2_train_v3_script = os.path.join(self.gpt_sovits_root, "GPT_SoVITS", "s2_train_v3_lora.py")
        
        if not os.path.exists(self.s2_train_script):
            raise FileNotFoundError(f"SoVITS训练脚本不存在: {self.s2_train_script}")
    
    def _find_gpt_sovits_root(self) -> str:
        """自动查找GPT-SoVITS项目根目录"""
        current_dir = Path(__file__).parent
        
        # 向上查找包含GPT_SoVITS目录的路径
        for parent in current_dir.parents:
            gpt_sovits_dir = parent / "GPT_SoVITS"
            if gpt_sovits_dir.exists():
                return str(parent)
        
        # 如果找不到，尝试相对路径
        possible_paths = [
            "../../../../文档/GPT-SoVITS-main",
            "../../../GPT-SoVITS-main",
            "../../GPT-SoVITS-main"
        ]
        
        for path in possible_paths:
            abs_path = Path(__file__).parent / path
            if abs_path.exists() and (abs_path / "GPT_SoVITS").exists():
                return str(abs_path.resolve())
        
        raise FileNotFoundError("无法找到GPT-SoVITS项目根目录")
    
    def _generate_job_id(self) -> str:
        """生成训练任务ID"""
        from datetime import datetime
        import uuid
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"sovits_{timestamp}_{short_uuid}"

    def _resolve_pretrained_s2_paths(self, version: str, pretrained_s2G: str, pretrained_s2D: str) -> tuple[str, str]:
        """解析 SoVITS 预训练底模路径，默认与官方 WebUI 保持一致。"""
        default_paths = {
            "v1": (
                os.path.join("GPT_SoVITS", "pretrained_models", "s2G488k.pth"),
                os.path.join("GPT_SoVITS", "pretrained_models", "s2D488k.pth"),
            ),
            "v2": (
                os.path.join("GPT_SoVITS", "pretrained_models", "gsv-v2final-pretrained", "s2G2333k.pth"),
                os.path.join("GPT_SoVITS", "pretrained_models", "gsv-v2final-pretrained", "s2D2333k.pth"),
            ),
            "v2Pro": (
                os.path.join("GPT_SoVITS", "pretrained_models", "v2Pro", "s2Gv2Pro.pth"),
                os.path.join("GPT_SoVITS", "pretrained_models", "v2Pro", "s2Dv2Pro.pth"),
            ),
            "v2ProPlus": (
                os.path.join("GPT_SoVITS", "pretrained_models", "v2Pro", "s2Gv2ProPlus.pth"),
                os.path.join("GPT_SoVITS", "pretrained_models", "v2Pro", "s2Dv2ProPlus.pth"),
            ),
            "v3": (
                os.path.join("GPT_SoVITS", "pretrained_models", "s2Gv3.pth"),
                os.path.join("GPT_SoVITS", "pretrained_models", "s2Dv3.pth"),
            ),
            "v4": (
                os.path.join("GPT_SoVITS", "pretrained_models", "gsv-v4-pretrained", "s2Gv4.pth"),
                os.path.join("GPT_SoVITS", "pretrained_models", "gsv-v4-pretrained", "s2Dv4.pth"),
            ),
        }
        default_g, default_d = default_paths.get(version, default_paths["v2Pro"])
        resolved_g = pretrained_s2G.strip() if pretrained_s2G and pretrained_s2G.strip() else os.path.join(self.gpt_sovits_root, default_g)
        resolved_d = pretrained_s2D.strip() if pretrained_s2D and pretrained_s2D.strip() else os.path.join(self.gpt_sovits_root, default_d)
        return resolved_g, resolved_d
    
    def _create_config_file(self, request: SoVITSTrainingRequest, s2_dir: str) -> str:
        """创建训练配置文件"""
        config = request.config
        version = config.version
        dataset_dir = os.path.join(s2_dir, "dataset")
        checkpoint_dir = os.path.join(s2_dir, "train", "sovits")
        save_every_epoch = max(1, min(config.save_every_epoch, config.total_epoch))
        pretrained_s2g_path, pretrained_s2d_path = self._resolve_pretrained_s2_paths(
            version,
            config.pretrained_s2G,
            config.pretrained_s2D,
        )
        
        # 选择配置模板
        if version not in {"v2Pro", "v2ProPlus"}:
            config_template = "GPT_SoVITS/configs/s2.json"
        else:
            config_template = f"GPT_SoVITS/configs/s2{version}.json"
        
        config_template_path = os.path.join(self.gpt_sovits_root, config_template)
        
        # 读取配置模板
        with open(config_template_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 更新配置
        data["train"]["batch_size"] = config.batch_size
        data["train"]["epochs"] = config.total_epoch
        data["train"]["text_low_lr_rate"] = config.text_low_lr_rate
        data["train"]["pretrained_s2G"] = pretrained_s2g_path
        data["train"]["pretrained_s2D"] = pretrained_s2d_path
        data["train"]["if_save_latest"] = config.if_save_latest
        data["train"]["if_save_every_weights"] = config.if_save_every_weights
        data["train"]["save_every_epoch"] = save_every_epoch
        data["train"]["gpu_numbers"] = config.gpu_numbers
        data["train"]["grad_ckpt"] = config.if_grad_ckpt
        data["train"]["lora_rank"] = config.lora_rank
        data["train"]["fp16_run"] = config.fp16_run
        data["train"]["learning_rate"] = config.learning_rate
        data["train"]["lr_decay"] = config.lr_decay
        
        # 损失权重
        data["train"]["c_mel"] = config.c_mel
        data["train"]["c_kl"] = config.c_kl
        
        # 数据配置
        data["data"]["segment_size"] = config.segment_size
        data["data"]["sampling_rate"] = config.sampling_rate
        
        # 模型和路径配置
        data["model"]["version"] = version
        data["data"]["exp_dir"] = dataset_dir
        data["s2_ckpt_dir"] = checkpoint_dir
        model_output_dir = request.model_output_dir.strip() if request.model_output_dir and request.model_output_dir.strip() else os.path.join(s2_dir, "SoVITS_weights")
        data["save_weight_dir"] = model_output_dir
        data["name"] = request.exp_name
        data["version"] = version
        
        # 如果不是半精度，调整batch_size
        if not config.fp16_run:
            data["train"]["fp16_run"] = False
            data["train"]["batch_size"] = max(1, config.batch_size // 2)
        
        # 保存配置文件
        config_file = os.path.join(s2_dir, f"s2_config_{version}.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return config_file
    
    def _prepare_training_environment(self, request: SoVITSTrainingRequest) -> tuple:
        """准备训练环境"""
        exp_name = request.exp_name.strip()
        s2_dir = request.workspace_dir.strip() if request.workspace_dir and request.workspace_dir.strip() else os.path.join(request.exp_root, exp_name)
        dataset_dir = os.path.join(s2_dir, "dataset")
        checkpoint_dir = os.path.join(s2_dir, "train", "sovits")
        logs_dir = os.path.join(s2_dir, "train", "logs")
        model_output_dir = request.model_output_dir.strip() if request.model_output_dir and request.model_output_dir.strip() else os.path.join(s2_dir, "SoVITS_weights")
        
        # 创建必要目录
        os.makedirs(s2_dir, exist_ok=True)
        os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, f"logs_s2_{request.config.version}"), exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(os.path.join(checkpoint_dir, "eval"), exist_ok=True)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # 生成配置文件
        config_file = self._create_config_file(request, s2_dir)
        
        return s2_dir, config_file

    def _open_log_file(self, log_file: str):
        """以行缓冲方式打开日志文件。"""
        return open(log_file, "a", encoding="utf-8", buffering=1)

    def _start_output_forwarder(self, process: Popen, log_handle, job_id: str) -> Optional[threading.Thread]:
        """将训练输出同时写入日志文件并实时打印到控制台。"""
        if process.stdout is None:
            return None

        def _forward():
            try:
                for line in process.stdout:
                    if not line:
                        continue
                    log_handle.write(line)
                    log_handle.flush()
                    print(f"[{job_id}] {line.rstrip()}")
            except Exception as exc:
                print(f"[{job_id}] 日志转发失败: {exc}")
            finally:
                try:
                    process.stdout.close()
                except Exception:
                    pass

        thread = threading.Thread(target=_forward, name=f"{job_id}_log_forwarder", daemon=True)
        thread.start()
        return thread
    
    async def start_training(self, request: SoVITSTrainingRequest) -> SoVITSTrainingResponse:
        """
        开始SoVITS训练
        
        Args:
            request: 训练请求参数
            
        Returns:
            SoVITSTrainingResponse: 训练响应
        """
        try:
            # 生成任务ID
            job_id = self._generate_job_id()
            
            # 准备训练环境
            s2_dir, config_file = self._prepare_training_environment(request)
            
            # 选择训练脚本
            if request.config.version in ["v1", "v2", "v2Pro", "v2ProPlus"]:
                train_script = self.s2_train_script
            else:
                train_script = self.s2_train_v3_script
            
            # 构建训练命令
            cmd = [
                self.python_exec, "-s", train_script,
                "--config", config_file
            ]
            
            # 设置环境变量
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = request.config.gpu_numbers.replace("-", ",")
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            
            # 创建日志文件
            log_file = os.path.join(s2_dir, "train", "logs", f"training_{job_id}.log")
            log_handle = self._open_log_file(log_file)
            
            # 启动训练进程
            print(f"启动SoVITS训练: {' '.join(cmd)}")
            process = Popen(
                cmd, 
                stdout=PIPE,
                stderr=STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                cwd=self.gpt_sovits_root
            )
            output_thread = self._start_output_forwarder(process, log_handle, job_id)
            
            # 记录训练任务
            self.training_jobs[job_id] = {
                "process": process,
                "output_thread": output_thread,
                "request": request,
                "config_file": config_file,
                "log_file": log_file,
                "log_handle": log_handle,
                "s2_dir": s2_dir,
                "start_time": datetime.now(),
                "status": "running"
            }
            
            # 创建训练状态
            status = SoVITSTrainingStatus(
                job_id=job_id,
                status="running",
                current_epoch=0,
                total_epochs=request.config.total_epoch,
                start_time=datetime.now(),
                log_file=log_file
            )
            
            return SoVITSTrainingResponse(
                success=True,
                message=f"SoVITS训练已启动，任务ID: {job_id}",
                job_id=job_id,
                status=status,
                config_file=config_file,
                log_dir=os.path.join(s2_dir, "train", "logs"),
                model_dir=request.model_output_dir.strip() if request.model_output_dir and request.model_output_dir.strip() else os.path.join(s2_dir, "SoVITS_weights")
            )
            
        except Exception as e:
            return SoVITSTrainingResponse(
                success=False,
                message=f"启动SoVITS训练失败: {str(e)}"
            )
    
    def start_training_sync(self, request: SoVITSTrainingRequest) -> SoVITSTrainingResponse:
        """
        同步版本的SoVITS训练启动
        
        Args:
            request: 训练请求参数
            
        Returns:
            SoVITSTrainingResponse: 训练响应
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.start_training(request))
        finally:
            loop.close()
    
    def get_training_status(self, job_id: str) -> Optional[SoVITSTrainingStatus]:
        """
        获取训练状态
        
        Args:
            job_id: 训练任务ID
            
        Returns:
            SoVITSTrainingStatus: 训练状态
        """
        if job_id not in self.training_jobs:
            return None
        
        job_info = self.training_jobs[job_id]
        process = job_info["process"]
        
        # 检查进程状态
        if process.poll() is None:
            status = "running"
        elif process.returncode == 0:
            status = "completed"
            job_info["status"] = "completed"
            job_info["end_time"] = datetime.now()
            log_handle = job_info.get("log_handle")
            if log_handle and not log_handle.closed:
                log_handle.close()
        else:
            status = "failed"
            job_info["status"] = "failed"
            job_info["end_time"] = datetime.now()
            log_handle = job_info.get("log_handle")
            if log_handle and not log_handle.closed:
                log_handle.close()
        
        return SoVITSTrainingStatus(
            job_id=job_id,
            status=status,
            current_epoch=0,  # TODO: 从日志文件解析当前epoch
            total_epochs=job_info["request"].config.total_epoch,
            start_time=job_info["start_time"],
            end_time=job_info.get("end_time"),
            log_file=job_info["log_file"]
        )
    
    def stop_training(self, job_id: str) -> bool:
        """
        停止训练
        
        Args:
            job_id: 训练任务ID
            
        Returns:
            bool: 是否成功停止
        """
        if job_id not in self.training_jobs:
            return False
        
        job_info = self.training_jobs[job_id]
        process = job_info["process"]
        
        try:
            if process.poll() is None:  # 进程还在运行
                process.terminate()
                process.wait(timeout=10)  # 等待10秒
                job_info["status"] = "stopped"
                job_info["end_time"] = datetime.now()
            log_handle = job_info.get("log_handle")
            if log_handle and not log_handle.closed:
                log_handle.close()
            return True
        except Exception as e:
            print(f"停止训练失败: {e}")
            return False
    
    def list_training_jobs(self) -> List[str]:
        """
        列出所有训练任务
        
        Returns:
            List[str]: 任务ID列表
        """
        return list(self.training_jobs.keys())
    
    def get_supported_versions(self) -> List[str]:
        """
        获取支持的模型版本
        
        Returns:
            List[str]: 支持的版本列表
        """
        return ["v1", "v2", "v3", "v4", "v2Pro", "v2ProPlus"]
    
    def validate_config(self, config: SoVITSTrainingConfig) -> Dict[str, Any]:
        """
        验证训练配置
        
        Args:
            config: 训练配置
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        issues = []
        suggestions = []
        
        # 检查版本
        if config.version not in self.get_supported_versions():
            issues.append(f"不支持的版本: {config.version}")
        
        # 检查batch_size
        if config.batch_size < 1:
            issues.append("batch_size必须大于0")
        elif config.batch_size > 64:
            suggestions.append("batch_size过大可能导致显存不足")
        
        # 检查epoch
        if config.total_epoch < 1:
            issues.append("total_epoch必须大于0")
        
        # 检查GPU配置
        try:
            gpu_list = config.gpu_numbers.split("-")
            for gpu in gpu_list:
                int(gpu)  # 验证是否为数字
        except ValueError:
            issues.append("GPU配置格式错误，应为数字用-分割")
        
        # 检查预训练模型
        if config.pretrained_s2G and not os.path.exists(config.pretrained_s2G):
            issues.append(f"预训练SoVITS-G模型不存在: {config.pretrained_s2G}")
        
        if config.pretrained_s2D and not os.path.exists(config.pretrained_s2D):
            issues.append(f"预训练SoVITS-D模型不存在: {config.pretrained_s2D}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "suggestions": suggestions
        }
