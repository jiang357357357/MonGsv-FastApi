#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 音频切分 API 核心模块

提供音频切分的核心功能和数据模型
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import List
from subprocess import Popen, PIPE

from pydantic import BaseModel, Field


class SliceConfig(BaseModel):
    """音频切分配置参数"""
    threshold: float = Field(default=-34.0, description="静音阈值(dB)，越小切得越碎")
    min_length: int = Field(default=4000, description="每段最短时长(ms)")
    min_interval: int = Field(default=300, description="最短切割间隔(ms)")
    hop_size: int = Field(default=10, description="音量检测精度(ms)")
    max_sil_kept: int = Field(default=500, description="句首句尾保留的静音长度(ms)")
    max_volume: float = Field(default=0.9, description="音量归一化最大值")
    alpha: float = Field(default=0.25, description="音量混合参数")
    n_parts: int = Field(default=1, description="并行处理分片数")


class SliceRequest(BaseModel):
    """音频切分请求"""
    input_path: str = Field(description="输入音频文件或目录路径")
    output_dir: str = Field(description="输出目录路径")
    config: SliceConfig = Field(default_factory=SliceConfig)


class SliceResponse(BaseModel):
    """音频切分响应"""
    success: bool
    message: str
    output_dir: str
    processed_files: List[str] = []
    output_files: List[str] = []
    error_files: List[str] = []


class AudioSliceService:
    """音频切分API类"""
    
    def __init__(self, gpt_sovits_root: str = None):
        """
        初始化音频切分API
        
        Args:
            gpt_sovits_root: GPT-SoVITS项目根目录路径
        """
        self.gpt_sovits_root = gpt_sovits_root or self._find_gpt_sovits_root()
        self.python_exec = sys.executable or "python"
        self.slice_script = os.path.join(self.gpt_sovits_root, "tools", "slice_audio.py")
        
        # 验证必要文件存在
        if not os.path.exists(self.slice_script):
            raise FileNotFoundError(f"切分脚本不存在: {self.slice_script}")
    
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
    
    def _clean_path(self, path_str: str) -> str:
        """清理路径字符串"""
        if path_str.endswith(("\\", "/")):
            return self._clean_path(path_str[0:-1])
        path_str = path_str.replace("/", os.sep).replace("\\", os.sep)
        return path_str.strip(" '\n\"\u202a")
    
    def _validate_input(self, input_path: str) -> bool:
        """验证输入路径"""
        clean_path = self._clean_path(input_path)
        return os.path.exists(clean_path)

    def _normalize_threshold(self, threshold: float) -> int:
        """切分脚本要求 threshold 为整数，这里提前收敛，避免子进程直接失败。"""
        return int(round(threshold))
    
    async def slice_audio(self, request: SliceRequest) -> SliceResponse:
        """
        执行音频切分
        
        Args:
            request: 切分请求参数
            
        Returns:
            SliceResponse: 切分结果
        """
        try:
            # 清理路径
            input_path = self._clean_path(request.input_path)
            output_dir = self._clean_path(request.output_dir)
            
            # 验证输入
            if not self._validate_input(input_path):
                return SliceResponse(
                    success=False,
                    message=f"输入路径不存在: {input_path}",
                    output_dir=output_dir
                )
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 确定处理分片数
            config = request.config
            if os.path.isfile(input_path):
                n_parts = 1
            else:
                n_parts = config.n_parts
            
            # 执行切分
            processes = []
            for i_part in range(n_parts):
                cmd = [
                    self.python_exec, "-s", self.slice_script,
                    input_path, output_dir,
                    str(self._normalize_threshold(config.threshold)),
                    str(config.min_length),
                    str(config.min_interval),
                    str(config.hop_size),
                    str(config.max_sil_kept),
                    str(config.max_volume),
                    str(config.alpha),
                    str(i_part),
                    str(n_parts)
                ]
                
                print(f"执行命令: {' '.join(cmd)}")
                process = Popen(cmd, stdout=PIPE, stderr=PIPE, text=True)
                processes.append(process)
            
            # 等待所有进程完成
            results = []
            for index, process in enumerate(processes):
                stdout, stderr = process.communicate()
                result = {
                    'returncode': process.returncode,
                    'stdout': stdout,
                    'stderr': stderr
                }
                results.append(result)
                if stdout.strip():
                    print(f"[audio_slice][part={index}] stdout:\n{stdout.strip()}")
                if stderr.strip():
                    print(f"[audio_slice][part={index}] stderr:\n{stderr.strip()}")
            
            # 检查结果
            success = all(result['returncode'] == 0 for result in results)
            
            # 统计输出文件
            output_files = []
            if os.path.exists(output_dir):
                output_files = [
                    os.path.join(output_dir, f) 
                    for f in os.listdir(output_dir) 
                    if f.endswith('.wav')
                ]
            
            # 处理错误信息
            error_messages = []
            for i, result in enumerate(results):
                if result['returncode'] != 0:
                    error_messages.append(f"进程{i}: {result['stderr']}")
            
            message = "切分完成" if success else f"切分失败: {'; '.join(error_messages)}"
            
            return SliceResponse(
                success=success,
                message=message,
                output_dir=output_dir,
                processed_files=[input_path] if os.path.isfile(input_path) else 
                               [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(('.wav', '.mp3', '.flac'))],
                output_files=output_files,
                error_files=[] if success else [input_path]
            )
            
        except Exception as e:
            return SliceResponse(
                success=False,
                message=f"切分过程出错: {str(e)}",
                output_dir=request.output_dir,
                error_files=[request.input_path]
            )

    async def process(self, request: SliceRequest) -> SliceResponse:
        """基础层统一入口。"""
        return await self.slice_audio(request)

    async def process_audio_slice(self, request: SliceRequest) -> SliceResponse:
        """兼容旧调用方的方法名。"""
        return await self.slice_audio(request)
    
    def slice_audio_sync(self, request: SliceRequest) -> SliceResponse:
        """
        同步版本的音频切分
        
        Args:
            request: 切分请求参数
            
        Returns:
            SliceResponse: 切分结果
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.slice_audio(request))
        finally:
            loop.close()
