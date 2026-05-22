"""
GPT-SoVITS 音频特征提取 API

实现步骤1Ab：音频特征提取
- CNHubert SSL特征提取（768维）
- 音频重采样和增强处理（32kHz）
- 说话人特征提取（v2Pro版本，20480维）
"""

import os
import sys
import torch
import traceback
import shutil
import numpy as np
import librosa
import torchaudio
from time import time as ttime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Literal
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.io import wavfile
from Code.FastApi.Base.gpt_sovits_env import setup_gpt_sovits_paths

current_dir = Path(__file__).resolve().parent
gpt_sovits_root = setup_gpt_sovits_paths(current_dir)

try:
    from feature_extractor import cnhubert
    from tools.my_utils import load_audio, clean_path
    
    # v2Pro说话人特征提取相关
    from ERes2NetV2 import ERes2NetV2
    import kaldi as Kaldi
    
except ImportError as e:
    print(f"警告: 无法导入GPT-SoVITS模块: {e}")
    print("请确保GPT-SoVITS路径正确")


@dataclass
class AudioFeaturesConfig:
    """音频特征提取配置"""
    # CNHubert模型配置
    cnhubert_base_dir: str = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
    
    # 说话人模型配置（v2Pro）
    sv_model_path: str = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
    
    # 版本配置
    version: str = "v2"  # v1, v2, v2Pro, v2ProPlus, v3, v4
    
    # 处理配置
    is_half: bool = True  # 是否使用半精度
    device: str = "auto"  # 设备选择
    
    # 音频处理参数
    maxx: float = 0.95    # 最大归一化值
    alpha: float = 0.5    # 混合比例
    max_audio_value: float = 2.2  # 音频最大值过滤阈值
    
    # 并行配置
    n_parts: int = 1  # 并行处理数
    
    # 输出配置
    save_wav32k: bool = True      # 是否保存32kHz音频
    save_cnhubert: bool = True    # 是否保存CNHubert特征
    save_speaker: bool = None     # 是否保存说话人特征（None=根据版本自动判断）


@dataclass
class AudioFeaturesRequest:
    """音频特征提取请求"""
    input_text_file: str  # 输入标注文件路径
    input_wav_dir: str    # 音频目录路径
    experiment_name: str  # 实验名称
    output_dir: str       # 输出目录
    config: AudioFeaturesConfig = field(default_factory=AudioFeaturesConfig)


@dataclass
class AudioFeaturesResponse:
    """音频特征提取响应"""
    success: bool
    message: str
    output_files: Dict[str, str] = field(default_factory=dict)
    processed_count: int = 0
    failed_count: int = 0
    nan_failed_count: int = 0
    processing_time: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class AudioFeaturesService:
    """音频特征提取API类"""
    
    def __init__(self, gpt_sovits_root: str = None):
        """
        初始化音频特征提取API
        
        Args:
            gpt_sovits_root: GPT-SoVITS根目录路径
        """
        self.gpt_sovits_root = gpt_sovits_root
        self.cnhubert_model = None
        self.speaker_model = None
        self.device = None
        
    def _setup_device(self, config: AudioFeaturesConfig) -> str:
        """设置计算设备"""
        if config.device == "auto":
            if torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"
        else:
            device = config.device
            
        self.device = device
        return device
    
    def _load_cnhubert_model(self, config: AudioFeaturesConfig):
        """加载CNHubert模型"""
        if not os.path.exists(config.cnhubert_base_dir):
            raise FileNotFoundError(f"CNHubert模型路径不存在: {config.cnhubert_base_dir}")
            
        print(f"加载CNHubert模型: {config.cnhubert_base_dir}")
        
        # 设置CNHubert路径
        cnhubert.cnhubert_base_path = config.cnhubert_base_dir
        
        # 加载模型
        self.cnhubert_model = cnhubert.get_model()
        
        if config.is_half and torch.cuda.is_available():
            self.cnhubert_model = self.cnhubert_model.half().to(self.device)
        else:
            self.cnhubert_model = self.cnhubert_model.to(self.device)
    
    def _load_speaker_model(self, config: AudioFeaturesConfig):
        """加载说话人特征提取模型（v2Pro）"""
        if not os.path.exists(config.sv_model_path):
            raise FileNotFoundError(f"说话人模型路径不存在: {config.sv_model_path}")
            
        print(f"加载说话人模型: {config.sv_model_path}")
        
        class SpeakerModel:
            def __init__(self, model_path, device, is_half):
                pretrained_state = torch.load(model_path, map_location="cpu")
                embedding_model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
                embedding_model.load_state_dict(pretrained_state)
                embedding_model.eval()
                self.embedding_model = embedding_model
                self.res = torchaudio.transforms.Resample(32000, 16000).to(device)
                
                if is_half:
                    self.embedding_model = self.embedding_model.half().to(device)
                else:
                    self.embedding_model = self.embedding_model.to(device)
                self.is_half = is_half

            def compute_embedding(self, wav):  # (1,x) 范围-1~1
                with torch.no_grad():
                    wav = self.res(wav)
                    if self.is_half:
                        wav = wav.half()
                    feat = torch.stack([
                        Kaldi.fbank(wav0.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0) 
                        for wav0 in wav
                    ])
                    sv_emb = self.embedding_model.forward3(feat)
                return sv_emb
        
        self.speaker_model = SpeakerModel(config.sv_model_path, self.device, config.is_half)
    
    def _save_tensor(self, tensor: torch.Tensor, path: str, i_part: int = 0):
        """
        安全保存张量（解决中文路径问题）
        
        Args:
            tensor: 要保存的张量
            path: 保存路径
            i_part: 进程编号
        """
        dir_path = os.path.dirname(path)
        name = os.path.basename(path)
        
        # 使用临时文件名
        tmp_path = f"{ttime()}{i_part}.pth"
        torch.save(tensor, tmp_path)
        shutil.move(tmp_path, os.path.join(dir_path, name))
    
    def _process_audio_file(self, 
                           wav_name: str, 
                           wav_path: str,
                           output_dirs: Dict[str, str],
                           config: AudioFeaturesConfig,
                           i_part: int = 0) -> Dict[str, Any]:
        """
        处理单个音频文件
        
        Args:
            wav_name: 音频文件名
            wav_path: 音频文件路径
            output_dirs: 输出目录字典
            config: 处理配置
            i_part: 进程编号
            
        Returns:
            处理结果字典
        """
        result = {
            "wav_name": wav_name,
            "success": False,
            "cnhubert_saved": False,
            "wav32k_saved": False,
            "speaker_saved": False,
            "nan_detected": False,
            "error": None
        }
        
        try:
            cnhubert_path = os.path.join(output_dirs["cnhubert"], f"{wav_name}.pt")
            speaker_path = os.path.join(output_dirs["speaker"], f"{wav_name}.pt")
            if config.save_cnhubert and os.path.exists(cnhubert_path):
                print(f"CNHubert特征已存在，将覆盖: {wav_name}")
            if config.save_speaker and os.path.exists(speaker_path):
                print(f"说话人特征已存在，将覆盖: {wav_name}")
            
            print(f"处理音频: {wav_name}")
            
            # 加载音频（32kHz）
            tmp_audio = load_audio(wav_path, 32000)
            tmp_max = np.abs(tmp_audio).max()
            
            # 过滤过大的音频
            if tmp_max > config.max_audio_value:
                result["error"] = f"音频幅值过大: {tmp_max}"
                print(f"{wav_name}-filtered, max_value={tmp_max}")
                return result
            
            # 音频增强处理
            tmp_audio32 = (tmp_audio / tmp_max * (config.maxx * config.alpha * 32768)) + \
                         ((1 - config.alpha) * 32768) * tmp_audio
            tmp_audio32b = (tmp_audio / tmp_max * (config.maxx * config.alpha * 1145.14)) + \
                          ((1 - config.alpha) * 1145.14) * tmp_audio
            
            # 保存32kHz音频
            if config.save_wav32k:
                wav32k_path = os.path.join(output_dirs["wav32k"], wav_name)
                wavfile.write(wav32k_path, 32000, tmp_audio32.astype("int16"))
                result["wav32k_saved"] = True
            
            # CNHubert特征提取
            if config.save_cnhubert:
                # 重采样到16kHz用于CNHubert
                tmp_audio_16k = librosa.resample(tmp_audio32b, orig_sr=32000, target_sr=16000)
                tensor_wav16 = torch.from_numpy(tmp_audio_16k)
                
                if config.is_half:
                    tensor_wav16 = tensor_wav16.half().to(self.device)
                else:
                    tensor_wav16 = tensor_wav16.to(self.device)
                
                # 提取SSL特征
                ssl = self.cnhubert_model.model(tensor_wav16.unsqueeze(0))["last_hidden_state"].transpose(1, 2).cpu()
                # 输出形状: torch.Size([1, 768, time_steps])
                
                # 检查NaN值
                if np.isnan(ssl.detach().numpy()).sum() != 0:
                    result["nan_detected"] = True
                    result["error"] = "CNHubert特征包含NaN值"
                    print(f"NaN检测到: {wav_name}")
                    return result
                
                # 保存CNHubert特征
                self._save_tensor(ssl, cnhubert_path, i_part)
                result["cnhubert_saved"] = True
            
            # 说话人特征提取（v2Pro）
            if config.save_speaker:
                # 从32kHz音频文件加载
                wav32k_path = os.path.join(output_dirs["wav32k"], wav_name)
                wav32k, sr0 = torchaudio.load(wav32k_path)
                assert sr0 == 32000, f"音频采样率不正确: {sr0}, 期望: 32000"
                
                wav32k = wav32k.to(self.device)
                emb = self.speaker_model.compute_embedding(wav32k).cpu()
                # 输出形状: torch.Size([1, 20480])
                
                # 保存说话人特征
                self._save_tensor(emb, speaker_path, i_part)
                result["speaker_saved"] = True
            
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            print(f"处理失败 {wav_name}: {traceback.format_exc()}")
            
        return result
    
    def _process_audio_batch(self, 
                            data: List[Tuple[str, str]], 
                            output_dirs: Dict[str, str],
                            config: AudioFeaturesConfig,
                            i_part: int = 0) -> List[Dict[str, Any]]:
        """
        处理音频批次
        
        Args:
            data: 音频数据列表 [(wav_name, wav_path), ...]
            output_dirs: 输出目录字典
            config: 处理配置
            i_part: 进程编号
            
        Returns:
            处理结果列表
        """
        results = []
        
        for wav_name, wav_path in data:
            result = self._process_audio_file(wav_name, wav_path, output_dirs, config, i_part)
            results.append(result)
            
        return results
    
    def _parse_input_file(self, input_file: str, input_wav_dir: str) -> List[Tuple[str, str]]:
        """
        解析输入标注文件
        
        Args:
            input_file: 输入文件路径
            input_wav_dir: 音频目录路径
            
        Returns:
            解析结果 [(wav_name, wav_path), ...]
        """
        todo = []
        
        with open(input_file, "r", encoding="utf8") as f:
            lines = f.read().strip("\n").split("\n")
        
        for line in lines:
            try:
                parts = line.split("|")
                if len(parts) >= 4:
                    wav_name, spk_name, language, text = parts[:4]
                    wav_name = clean_path(wav_name)
                    
                    if input_wav_dir and input_wav_dir.strip():
                        wav_name = os.path.basename(wav_name)
                        wav_path = os.path.join(input_wav_dir, wav_name)
                    else:
                        wav_path = wav_name
                        wav_name = os.path.basename(wav_name)
                    
                    todo.append((wav_name, wav_path))
                    
            except Exception as e:
                print(f"解析行失败: {line}, 错误: {traceback.format_exc()}")
                
        return todo
    
    async def extract_features(self, request: AudioFeaturesRequest) -> AudioFeaturesResponse:
        """
        异步提取音频特征
        
        Args:
            request: 处理请求
            
        Returns:
            处理响应
        """
        return self.extract_features_sync(request)

    async def process(self, request: AudioFeaturesRequest) -> AudioFeaturesResponse:
        """基础层统一入口。"""
        return await self.extract_features(request)

    async def extract_audio_features(self, request: AudioFeaturesRequest) -> AudioFeaturesResponse:
        """兼容旧调用方的方法名。"""
        return await self.extract_features(request)
    
    def extract_features_sync(self, request: AudioFeaturesRequest) -> AudioFeaturesResponse:
        """
        同步提取音频特征
        
        Args:
            request: 处理请求
            
        Returns:
            处理响应
        """
        start_time = ttime()
        
        try:
            # 验证输入
            if not os.path.exists(request.input_text_file):
                return AudioFeaturesResponse(
                    success=False,
                    message=f"输入文件不存在: {request.input_text_file}"
                )
            
            if request.input_wav_dir and not os.path.exists(request.input_wav_dir):
                return AudioFeaturesResponse(
                    success=False,
                    message=f"音频目录不存在: {request.input_wav_dir}"
                )
            
            # 设置设备
            device = self._setup_device(request.config)
            print(f"使用设备: {device}")
            
            # 创建输出目录
            os.makedirs(request.output_dir, exist_ok=True)
            
            output_dirs = {
                "cnhubert": os.path.join(request.output_dir, "4-cnhubert"),
                "wav32k": os.path.join(request.output_dir, "5-wav32k"),
                "speaker": os.path.join(request.output_dir, "7-sv_cn")
            }
            
            for dir_path in output_dirs.values():
                os.makedirs(dir_path, exist_ok=True)
            
            # 判断是否需要说话人特征
            if request.config.save_speaker is None:
                request.config.save_speaker = "Pro" in request.config.version
            
            # 加载模型
            if request.config.save_cnhubert:
                self._load_cnhubert_model(request.config)
            
            if request.config.save_speaker:
                self._load_speaker_model(request.config)
            
            # 解析输入文件
            todo = self._parse_input_file(request.input_text_file, request.input_wav_dir)
            if not todo:
                return AudioFeaturesResponse(
                    success=False,
                    message="没有找到有效的音频数据"
                )
            
            print(f"找到 {len(todo)} 个音频文件")
            
            # 分批处理
            all_results = []
            n_parts = request.config.n_parts
            
            if n_parts > 1:
                # 多进程处理
                batch_size = len(todo) // n_parts + (1 if len(todo) % n_parts else 0)
                batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
                
                with ProcessPoolExecutor(max_workers=n_parts) as executor:
                    futures = []
                    for i, batch in enumerate(batches):
                        future = executor.submit(
                            self._process_audio_batch,
                            batch, output_dirs, request.config, i
                        )
                        futures.append(future)
                    
                    for future in as_completed(futures):
                        batch_results = future.result()
                        all_results.extend(batch_results)
            else:
                # 单进程处理
                all_results = self._process_audio_batch(todo, output_dirs, request.config)
            
            # 统计结果
            processed_count = sum(1 for r in all_results if r["success"])
            failed_count = len(all_results) - processed_count
            nan_failed_count = sum(1 for r in all_results if r.get("nan_detected", False))
            
            # 处理NaN失败的情况（降级到float32）
            if nan_failed_count > 0 and request.config.is_half:
                print(f"检测到 {nan_failed_count} 个NaN失败，降级到float32重试...")
                request.config.is_half = False
                
                # 重新加载模型
                if request.config.save_cnhubert:
                    self._load_cnhubert_model(request.config)
                
                # 重试失败的文件
                retry_todo = [(r["wav_name"], "") for r in all_results if r.get("nan_detected", False)]
                if retry_todo:
                    retry_results = self._process_audio_batch(retry_todo, output_dirs, request.config)
                    # 更新结果
                    for i, result in enumerate(all_results):
                        if result.get("nan_detected", False):
                            for retry_result in retry_results:
                                if retry_result["wav_name"] == result["wav_name"]:
                                    all_results[i] = retry_result
                                    break
            
            # 重新统计
            processed_count = sum(1 for r in all_results if r["success"])
            failed_count = len(all_results) - processed_count
            
            processing_time = ttime() - start_time
            
            return AudioFeaturesResponse(
                success=True,
                message=f"音频特征提取完成，处理了 {processed_count} 个文件",
                output_files={
                    "cnhubert_dir": output_dirs["cnhubert"] if request.config.save_cnhubert else None,
                    "wav32k_dir": output_dirs["wav32k"] if request.config.save_wav32k else None,
                    "speaker_dir": output_dirs["speaker"] if request.config.save_speaker else None
                },
                processed_count=processed_count,
                failed_count=failed_count,
                nan_failed_count=nan_failed_count,
                processing_time=processing_time,
                details={
                    "total_input": len(todo),
                    "cnhubert_features_extracted": request.config.save_cnhubert,
                    "wav32k_saved": request.config.save_wav32k,
                    "speaker_features_extracted": request.config.save_speaker,
                    "device_used": device,
                    "parallel_parts": n_parts,
                    "version": request.config.version
                }
            )
            
        except Exception as e:
            return AudioFeaturesResponse(
                success=False,
                message=f"处理失败: {str(e)}",
                processing_time=ttime() - start_time
            )
