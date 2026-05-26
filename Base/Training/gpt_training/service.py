#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS GPT训练 API 核心模块

提供GPT模型训练的核心功能和数据模型
"""

import os
import sys
import yaml
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


class GPTTrainingConfig(BaseModel):
    """GPT训练配置参数"""
    version: str = Field(default="v2Pro", description="训练模型版本: v1, v2, v3, v4, v2Pro, v2ProPlus")
    # 基础训练参数
    batch_size: int = Field(default=8, description="每张显卡的batch_size")
    total_epoch: int = Field(default=15, description="总训练轮数")
    save_every_epoch: int = Field(default=20, description="保存频率")
    
    # 学习率相关
    learning_rate: float = Field(default=0.01, description="初始学习率")
    warmup_steps: int = Field(default=2000, description="预热步数")
    decay_steps: int = Field(default=40000, description="衰减步数")
    
    # 保存选项
    if_save_latest: bool = Field(default=True, description="是否仅保存最新的权重文件")
    if_save_every_weights: bool = Field(default=True, description="是否保存所有权重到weights文件夹")
    
    # GPU配置
    gpu_numbers: str = Field(default="0", description="GPU卡号，多卡用-分割")
    
    # 预训练模型路径
    pretrained_s1: str = Field(default="", description="预训练GPT模型路径")
    
    # 高级选项
    if_dpo: bool = Field(default=False, description="是否开启DPO训练选项(实验性)")
    precision: str = Field(default="16-mixed", description="训练精度: 16-mixed, 32")
    gradient_clip: float = Field(default=1.0, description="梯度裁剪阈值")
    
    # 数据相关
    max_sec: int = Field(default=54, description="最大音频长度(秒)")
    num_workers: int = Field(default=4, description="数据加载进程数")
    
    # 模型架构
    vocab_size: int = Field(default=1025, description="语义token词汇表大小")
    phoneme_vocab_size: int = Field(default=732, description="音素词汇表大小")
    embedding_dim: int = Field(default=512, description="嵌入维度")
    hidden_dim: int = Field(default=512, description="隐藏维度")
    n_layer: int = Field(default=24, description="Transformer层数")
    n_head: int = Field(default=16, description="注意力头数")


class GPTTrainingRequest(BaseModel):
    """GPT训练请求"""
    exp_name: str = Field(description="实验/模型名称")
    exp_root: str = Field(description="实验根目录路径")
    workspace_dir: str = Field(default="", description="显式训练工作区目录，优先级高于 exp_root/exp_name")
    model_output_dir: str = Field(default="", description="最终模型导出目录")
    config: GPTTrainingConfig = Field(default_factory=GPTTrainingConfig)


class GPTTrainingStatus(BaseModel):
    """训练状态"""
    job_id: str
    status: str  # "running", "completed", "failed", "stopped"
    current_epoch: int = 0
    total_epochs: int = 0
    current_loss: Optional[float] = None
    best_loss: Optional[float] = None
    top3_accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    log_file: Optional[str] = None


class GPTTrainingResponse(BaseModel):
    """GPT训练响应"""
    success: bool
    message: str
    job_id: Optional[str] = None
    status: Optional[GPTTrainingStatus] = None
    config_file: Optional[str] = None
    log_dir: Optional[str] = None
    model_dir: Optional[str] = None


class GPTTrainingService:
    """GPT训练API类"""
    
    def __init__(self, gpt_sovits_root: str = None):
        """
        初始化GPT训练API
        
        Args:
            gpt_sovits_root: GPT-SoVITS项目根目录路径
        """
        self.gpt_sovits_root = gpt_sovits_root or self._find_gpt_sovits_root()
        self.python_exec = sys.executable or "python"
        self.training_jobs: Dict[str, Dict] = {}  # 存储训练任务信息
        
        # 验证必要文件存在
        self.s1_train_script = os.path.join(self.gpt_sovits_root, "GPT_SoVITS", "s1_train.py")
        
        if not os.path.exists(self.s1_train_script):
            raise FileNotFoundError(f"GPT训练脚本不存在: {self.s1_train_script}")
    
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
        return f"gpt_{timestamp}_{short_uuid}"

    def _resolve_pretrained_s1(self, version: str, pretrained_s1: str) -> str:
        """解析 GPT 预训练底模路径，默认与官方 WebUI 保持一致。"""
        if pretrained_s1 and pretrained_s1.strip():
            return pretrained_s1.strip()

        relative_path = (
            os.path.join("GPT_SoVITS", "pretrained_models", "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")
            if version == "v1"
            else os.path.join("GPT_SoVITS", "pretrained_models", "s1v3.ckpt")
        )
        return os.path.join(self.gpt_sovits_root, relative_path)
    
    def _create_config_file(self, request: GPTTrainingRequest, s1_dir: str) -> str:
        """创建训练配置文件"""
        config = request.config
        dataset_dir = os.path.join(s1_dir, "dataset")
        checkpoint_dir = os.path.join(s1_dir, "train", "gpt")
        save_every_epoch = max(1, min(config.save_every_epoch, config.total_epoch))
        
        # 选择配置模板
        config_template = (
            "GPT_SoVITS/configs/s1longer.yaml"
            if config.version == "v1"
            else "GPT_SoVITS/configs/s1longer-v2.yaml"
        )
        config_template_path = os.path.join(self.gpt_sovits_root, config_template)
        
        # 读取配置模板
        with open(config_template_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # 更新训练配置
        data["train"]["batch_size"] = config.batch_size
        data["train"]["epochs"] = config.total_epoch
        data["train"]["save_every_n_epoch"] = save_every_epoch
        data["train"]["if_save_latest"] = config.if_save_latest
        data["train"]["if_save_every_weights"] = config.if_save_every_weights
        data["train"]["if_dpo"] = config.if_dpo
        data["train"]["precision"] = config.precision
        data["train"]["exp_name"] = request.exp_name
        
        # 学习率配置
        data["optimizer"]["lr"] = config.learning_rate
        data["optimizer"]["warmup_steps"] = config.warmup_steps
        data["optimizer"]["decay_steps"] = config.decay_steps
        
        # 数据配置
        data["data"]["max_sec"] = config.max_sec
        data["data"]["num_workers"] = config.num_workers
        
        # 模型配置
        data["model"]["vocab_size"] = config.vocab_size
        data["model"]["phoneme_vocab_size"] = config.phoneme_vocab_size
        data["model"]["embedding_dim"] = config.embedding_dim
        data["model"]["hidden_dim"] = config.hidden_dim
        data["model"]["n_layer"] = config.n_layer
        data["model"]["head"] = config.n_head
        
        # 路径配置
        data["pretrained_s1"] = self._resolve_pretrained_s1(config.version, config.pretrained_s1)
        model_output_dir = request.model_output_dir.strip() if request.model_output_dir and request.model_output_dir.strip() else os.path.join(s1_dir, "GPT_weights")
        data["train"]["half_weights_save_dir"] = model_output_dir
        data["train_semantic_path"] = os.path.join(dataset_dir, "6-name2semantic.tsv")
        data["train_phoneme_path"] = os.path.join(dataset_dir, "2-name2text.txt")
        data["output_dir"] = checkpoint_dir
        
        # 如果不是混合精度，调整batch_size
        if config.precision == "32":
            data["train"]["batch_size"] = max(1, config.batch_size // 2)
        
        # 保存配置文件
        config_file = os.path.join(s1_dir, "s1_config.yaml")
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        
        return config_file

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
    
    def _prepare_training_environment(self, request: GPTTrainingRequest) -> tuple:
        """准备训练环境"""
        exp_name = request.exp_name.strip()
        s1_dir = request.workspace_dir.strip() if request.workspace_dir and request.workspace_dir.strip() else os.path.join(request.exp_root, exp_name)
        dataset_dir = os.path.join(s1_dir, "dataset")
        checkpoint_dir = os.path.join(s1_dir, "train", "gpt")
        logs_dir = os.path.join(s1_dir, "train", "logs")
        model_output_dir = request.model_output_dir.strip() if request.model_output_dir and request.model_output_dir.strip() else os.path.join(s1_dir, "GPT_weights")
        
        # 创建必要目录
        os.makedirs(s1_dir, exist_ok=True)
        os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # 生成配置文件
        config_file = self._create_config_file(request, s1_dir)
        
        return s1_dir, config_file
    
    def _validate_training_data(self, s1_dir: str) -> Dict[str, Any]:
        """验证训练数据"""
        dataset_dir = os.path.join(s1_dir, "dataset")
        required_files = [
            "2-name2text.txt",
            "6-name2semantic.tsv"
        ]
        
        missing_files = []
        for file_name in required_files:
            file_path = os.path.join(dataset_dir, file_name)
            if not os.path.exists(file_path):
                missing_files.append(file_name)
        
        if missing_files:
            return {
                "valid": False,
                "error": f"缺少必要的训练数据文件: {', '.join(missing_files)}"
            }
        
        # 检查数据量
        text_file = os.path.join(dataset_dir, "2-name2text.txt")
        semantic_file = os.path.join(dataset_dir, "6-name2semantic.tsv")
        
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                text_lines = len(f.readlines())
            
            with open(semantic_file, 'r', encoding='utf-8') as f:
                semantic_lines = len(f.readlines()) - 1  # 减去标题行
            
            if text_lines < 10 or semantic_lines < 10:
                return {
                    "valid": False,
                    "error": f"训练数据量不足: 文本{text_lines}行, 语义{semantic_lines}行 (建议至少100行)"
                }
            
            return {
                "valid": True,
                "text_samples": text_lines,
                "semantic_samples": semantic_lines
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"读取训练数据失败: {str(e)}"
            }
    
    async def start_training(self, request: GPTTrainingRequest) -> GPTTrainingResponse:
        """
        开始GPT训练
        
        Args:
            request: 训练请求参数
            
        Returns:
            GPTTrainingResponse: 训练响应
        """
        try:
            # 生成任务ID
            job_id = self._generate_job_id()
            
            # 准备训练环境
            s1_dir, config_file = self._prepare_training_environment(request)
            
            # 验证训练数据
            validation_result = self._validate_training_data(s1_dir)
            if not validation_result["valid"]:
                print(f"[gpt_training] 停止启动: {validation_result['error']}")
                return GPTTrainingResponse(
                    success=False,
                    message=validation_result["error"]
                )
            print(f"[gpt_training] 训练前置数据检查通过: {validation_result}")
            
            # 构建训练命令
            cmd = [
                self.python_exec, "-s", self.s1_train_script,
                "--config_file", config_file
            ]
            
            # 设置环境变量
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = request.config.gpu_numbers.replace("-", ",")
            env["MASTER_ADDR"] = "127.0.0.1"
            env["USE_LIBUV"] = "0"
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            
            # 创建日志文件
            log_file = os.path.join(s1_dir, "train", "logs", f"training_{job_id}.log")
            log_handle = self._open_log_file(log_file)
            
            # 启动训练进程
            print(f"启动GPT训练: {' '.join(cmd)}")
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
                "s1_dir": s1_dir,
                "start_time": datetime.now(),
                "status": "running",
                "validation_result": validation_result
            }
            
            # 创建训练状态
            status = GPTTrainingStatus(
                job_id=job_id,
                status="running",
                current_epoch=0,
                total_epochs=request.config.total_epoch,
                start_time=datetime.now(),
                log_file=log_file
            )
            
            return GPTTrainingResponse(
                success=True,
                message=f"GPT训练已启动，任务ID: {job_id}",
                job_id=job_id,
                status=status,
                config_file=config_file,
                log_dir=os.path.join(s1_dir, "train", "logs"),
                model_dir=request.model_output_dir.strip() if request.model_output_dir and request.model_output_dir.strip() else os.path.join(s1_dir, "GPT_weights")
            )
            
        except Exception as e:
            return GPTTrainingResponse(
                success=False,
                message=f"启动GPT训练失败: {str(e)}"
            )
    
    def start_training_sync(self, request: GPTTrainingRequest) -> GPTTrainingResponse:
        """
        同步版本的GPT训练启动
        
        Args:
            request: 训练请求参数
            
        Returns:
            GPTTrainingResponse: 训练响应
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.start_training(request))
        finally:
            loop.close()
    
    def get_training_status(self, job_id: str) -> Optional[GPTTrainingStatus]:
        """
        获取训练状态
        
        Args:
            job_id: 训练任务ID
            
        Returns:
            GPTTrainingStatus: 训练状态
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
        
        return GPTTrainingStatus(
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
    
    def validate_config(self, config: GPTTrainingConfig) -> Dict[str, Any]:
        """
        验证训练配置
        
        Args:
            config: 训练配置
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        issues = []
        suggestions = []
        
        # 检查batch_size
        if config.batch_size < 1:
            issues.append("batch_size必须大于0")
        elif config.batch_size > 32:
            suggestions.append("batch_size过大可能导致显存不足")
        
        # 检查epoch
        if config.total_epoch < 1:
            issues.append("total_epoch必须大于0")
        elif config.total_epoch > 100:
            suggestions.append("total_epoch过大可能导致过拟合")
        
        # 检查学习率
        if config.learning_rate <= 0:
            issues.append("learning_rate必须大于0")
        elif config.learning_rate > 0.1:
            suggestions.append("learning_rate过大可能导致训练不稳定")
        
        # 检查GPU配置
        try:
            gpu_list = config.gpu_numbers.split("-")
            for gpu in gpu_list:
                int(gpu)  # 验证是否为数字
        except ValueError:
            issues.append("GPU配置格式错误，应为数字用-分割")
        
        # 检查预训练模型
        if config.pretrained_s1 and not os.path.exists(config.pretrained_s1):
            issues.append(f"预训练GPT模型不存在: {config.pretrained_s1}")
        
        # 检查精度设置
        if config.precision not in ["16-mixed", "32"]:
            issues.append("precision必须为'16-mixed'或'32'")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "suggestions": suggestions
        }
