"""
GPT-SoVITS 语义编码 API

实现步骤1Ac：语义编码
- 使用预训练SoVITS-G编码器提取语义Token序列
- 支持多版本模型自动检测
- 输出6-name2semantic.tsv格式文件
"""

import os
import sys
import torch
import traceback
import json
from time import time as ttime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Literal
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from Code.FastApi.Base.gpt_sovits_env import setup_gpt_sovits_paths

current_dir = Path(__file__).resolve().parent
gpt_sovits_root = setup_gpt_sovits_paths(current_dir)

try:
    import utils as gpt_utils
    from tools.my_utils import clean_path
    
    # 动态导入模型（根据版本）
    def import_model_class(version: str):
        if version in ["v3", "v4"]:
            from module.models import SynthesizerTrnV3 as SynthesizerTrn
        else:
            from module.models import SynthesizerTrn
        return SynthesizerTrn
    
except ImportError as e:
    print(f"警告: 无法导入GPT-SoVITS模块: {e}")
    print("请确保GPT-SoVITS路径正确")


@dataclass
class SemanticEncodingConfig:
    """语义编码配置"""
    # 模型配置
    pretrained_s2G: str = ""  # 预训练SoVITS模型路径，为空时根据版本自动推导
    s2config_path: str = ""  # 模型配置文件路径，为空时根据版本自动推导
    
    # 版本配置（自动检测或手动指定）
    version: Optional[str] = None  # v1, v2, v3, v4, v2Pro, v2ProPlus, None=自动检测
    
    # 处理配置
    is_half: bool = True  # 是否使用半精度
    device: str = "auto"  # 设备选择：auto, cuda, cpu
    
    # 并行配置
    n_parts: int = 1  # 并行处理数
    
    # 输出配置
    output_format: str = "tsv"  # 输出格式：tsv, json


@dataclass
class SemanticEncodingRequest:
    """语义编码请求"""
    input_text_file: str  # 输入标注文件路径
    cnhubert_dir: str     # CNHubert特征目录路径
    experiment_name: str  # 实验名称
    output_dir: str       # 输出目录
    config: SemanticEncodingConfig = field(default_factory=SemanticEncodingConfig)


@dataclass
class SemanticEncodingResponse:
    """语义编码响应"""
    success: bool
    message: str
    output_file: Optional[str] = None
    processed_count: int = 0
    failed_count: int = 0
    processing_time: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class SemanticEncodingService:
    """语义编码API类"""
    
    def __init__(self, gpt_sovits_root: str = None):
        """
        初始化语义编码API
        
        Args:
            gpt_sovits_root: GPT-SoVITS根目录路径
        """
        self.gpt_sovits_root = gpt_sovits_root or str(globals().get("gpt_sovits_root", ""))
        self.vq_model = None
        self.device = None
        self.version = None

    def _resolve_runtime_path(self, value: str) -> str:
        """将路径解析到当前 GPT-SoVITS 工作区。"""
        if not value:
            return value

        path = Path(value)
        if path.is_absolute():
            return str(path)

        if self.gpt_sovits_root:
            return str((Path(self.gpt_sovits_root) / path).resolve())

        return str(path.resolve())

    def _resolve_default_pretrained_s2g(self, version: Optional[str]) -> str:
        """按官方版本映射选择默认 SoVITS-G 预训练模型。"""
        version_key = (version or "v2Pro").strip() or "v2Pro"
        default_map = {
            "v1": "GPT_SoVITS/pretrained_models/s2G488k.pth",
            "v2": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
            "v3": "GPT_SoVITS/pretrained_models/s2Gv3.pth",
            "v4": "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth",
            "v2Pro": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth",
            "v2ProPlus": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
        }
        return self._resolve_runtime_path(default_map.get(version_key, default_map["v2Pro"]))

    def _resolve_default_s2config_path(self, version: Optional[str]) -> str:
        """按版本选择默认 SoVITS 配置文件。"""
        version_key = (version or "v2Pro").strip() or "v2Pro"
        config_map = {
            "v2Pro": "GPT_SoVITS/configs/s2v2Pro.json",
            "v2ProPlus": "GPT_SoVITS/configs/s2v2ProPlus.json",
        }
        return self._resolve_runtime_path(config_map.get(version_key, "GPT_SoVITS/configs/s2.json"))

    def _normalize_config_paths(self, config: SemanticEncodingConfig) -> SemanticEncodingConfig:
        """规范化模型与配置文件路径。"""
        if config.pretrained_s2G and config.pretrained_s2G.strip():
            config.pretrained_s2G = self._resolve_runtime_path(config.pretrained_s2G.strip())
        else:
            config.pretrained_s2G = self._resolve_default_pretrained_s2g(config.version)

        if config.s2config_path and config.s2config_path.strip():
            config.s2config_path = self._resolve_runtime_path(config.s2config_path.strip())
        else:
            config.s2config_path = self._resolve_default_s2config_path(config.version)

        return config
        
    def _detect_model_version(self, model_path: str) -> str:
        """
        根据模型文件大小自动检测版本
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            检测到的版本
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
        size = os.path.getsize(model_path)
        size_mb = size / (1024 * 1024)
        
        print(f"模型文件大小: {size_mb:.1f}MB")
        
        # 根据官方逻辑检测版本
        if size < 82978 * 1024:  # < 81MB
            version = "v1"
        elif size < 100 * 1024 * 1024:  # < 100MB
            version = "v2"
        elif size < 103520 * 1024:  # < 101MB
            version = "v1"
        elif size < 700 * 1024 * 1024:  # < 700MB
            version = "v2"
        else:  # >= 700MB
            version = "v3"
            
        print(f"自动检测模型版本: {version}")
        return version
    
    def _setup_device(self, config: SemanticEncodingConfig) -> str:
        """设置计算设备"""
        if config.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        else:
            device = config.device
            
        self.device = device
        return device
    
    def _load_model_config(self, config_path: str) -> Any:
        """
        加载模型配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置对象
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        # 使用GPT-SoVITS的utils加载配置
        hps = gpt_utils.get_hparams_from_file(config_path)
        return hps
    
    def _load_vq_model(self, config: SemanticEncodingConfig):
        """
        加载VQ模型
        
        Args:
            config: 编码配置
        """
        config = self._normalize_config_paths(config)

        # 检测或使用指定版本
        if config.version is None:
            self.version = self._detect_model_version(config.pretrained_s2G)
        else:
            self.version = config.version
            
        print(f"使用模型版本: {self.version}")
        
        # 加载配置
        hps = self._load_model_config(config.s2config_path)
        
        # 动态导入模型类
        SynthesizerTrn = import_model_class(self.version)
        
        # 创建模型
        self.vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            version=self.version,
            **hps.model,
        )
        
        # 设置设备和精度
        if config.is_half and torch.cuda.is_available():
            self.vq_model = self.vq_model.half().to(self.device)
        else:
            self.vq_model = self.vq_model.to(self.device)
            
        self.vq_model.eval()
        
        # 加载预训练权重
        print(f"加载预训练模型: {config.pretrained_s2G}")
        checkpoint = torch.load(config.pretrained_s2G, map_location="cpu", weights_only=False)
        
        # 处理不同的权重格式
        if "weight" in checkpoint:
            state_dict = checkpoint["weight"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
            
        missing_keys, unexpected_keys = self.vq_model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"缺失的键: {missing_keys}")
        if unexpected_keys:
            print(f"意外的键: {unexpected_keys}")
            
        print("模型加载完成")
    
    def _extract_semantic_tokens(self, 
                                wav_name: str, 
                                cnhubert_dir: str,
                                config: SemanticEncodingConfig) -> Optional[str]:
        """
        提取单个音频的语义Token
        
        Args:
            wav_name: 音频文件名（不含扩展名）
            cnhubert_dir: CNHubert特征目录
            config: 编码配置
            
        Returns:
            语义Token序列字符串，失败返回None
        """
        try:
            hubert_candidates = [
                os.path.join(cnhubert_dir, f"{wav_name}.pt"),
            ]
            wav_name_no_ext = os.path.splitext(wav_name)[0]
            if wav_name_no_ext != wav_name:
                hubert_candidates.append(os.path.join(cnhubert_dir, f"{wav_name_no_ext}.pt"))

            hubert_path = next((path for path in hubert_candidates if os.path.exists(path)), None)
            if hubert_path is None:
                print(f"CNHubert特征文件不存在: {' | '.join(hubert_candidates)}")
                return None
            
            # 加载CNHubert特征
            ssl_content = torch.load(hubert_path, map_location="cpu")
            
            # 设置设备和精度
            if config.is_half and torch.cuda.is_available():
                ssl_content = ssl_content.half().to(self.device)
            else:
                ssl_content = ssl_content.to(self.device)
            
            # 提取语义Token
            with torch.no_grad():
                codes = self.vq_model.extract_latent(ssl_content)
                
            # 转换为字符串序列
            semantic_tokens = codes[0, 0, :].tolist()
            semantic_str = " ".join([str(token) for token in semantic_tokens])
            
            return semantic_str
            
        except Exception as e:
            print(f"提取语义Token失败 {wav_name}: {str(e)}")
            traceback.print_exc()
            return None
    
    def _parse_input_file(self, input_file: str) -> List[Tuple[str, str, str, str]]:
        """
        解析输入标注文件
        
        Args:
            input_file: 输入文件路径
            
        Returns:
            解析结果 [(wav_name, spk_name, language, text), ...]
        """
        results = []
        
        with open(input_file, "r", encoding="utf8") as f:
            lines = f.read().strip("\n").split("\n")
        
        for line in lines:
            try:
                parts = line.split("|")
                if len(parts) >= 4:
                    wav_name, spk_name, language, text = parts[:4]
                    wav_name = clean_path(wav_name)
                    wav_name = os.path.basename(wav_name)
                    
                    results.append((wav_name, spk_name, language, text))
                    
            except Exception as e:
                print(f"解析行失败: {line}, 错误: {str(e)}")
                
        return results
    
    def _process_batch(self, 
                      data_batch: List[Tuple[str, str, str, str]], 
                      cnhubert_dir: str,
                      config: SemanticEncodingConfig,
                      i_part: int = 0) -> List[Tuple[str, Optional[str]]]:
        """
        处理数据批次
        
        Args:
            data_batch: 数据批次
            cnhubert_dir: CNHubert特征目录
            config: 编码配置
            i_part: 进程编号
            
        Returns:
            处理结果 [(wav_name, semantic_str), ...]
        """
        results = []
        
        for wav_name, spk_name, language, text in data_batch:
            semantic_str = self._extract_semantic_tokens(wav_name, cnhubert_dir, config)
            results.append((wav_name, semantic_str))
            
        return results
    
    async def encode_semantic(self, request: SemanticEncodingRequest) -> SemanticEncodingResponse:
        """
        异步语义编码
        
        Args:
            request: 编码请求
            
        Returns:
            编码响应
        """
        return self.encode_semantic_sync(request)

    async def process(self, request: SemanticEncodingRequest) -> SemanticEncodingResponse:
        """基础层统一入口。"""
        return await self.encode_semantic(request)

    async def encode_semantic_features(self, request: SemanticEncodingRequest) -> SemanticEncodingResponse:
        """兼容旧调用方的方法名。"""
        return await self.encode_semantic(request)
    
    def encode_semantic_sync(self, request: SemanticEncodingRequest) -> SemanticEncodingResponse:
        """
        同步语义编码
        
        Args:
            request: 编码请求
            
        Returns:
            编码响应
        """
        start_time = ttime()
        
        try:
            request.config = self._normalize_config_paths(request.config)

            # 验证输入
            if not os.path.exists(request.input_text_file):
                return SemanticEncodingResponse(
                    success=False,
                    message=f"输入文件不存在: {request.input_text_file}"
                )
            
            if not os.path.exists(request.cnhubert_dir):
                return SemanticEncodingResponse(
                    success=False,
                    message=f"CNHubert特征目录不存在: {request.cnhubert_dir}"
                )
            
            if not os.path.exists(request.config.pretrained_s2G):
                return SemanticEncodingResponse(
                    success=False,
                    message=f"预训练模型不存在: {request.config.pretrained_s2G}",
                    details={
                        "resolved_pretrained_s2G": request.config.pretrained_s2G,
                        "resolved_s2config_path": request.config.s2config_path,
                        "version": request.config.version or "auto",
                    }
                )
            
            # 设置设备
            device = self._setup_device(request.config)
            print(f"使用设备: {device}")
            
            # 创建输出目录
            os.makedirs(request.output_dir, exist_ok=True)
            
            # 加载模型
            self._load_vq_model(request.config)
            
            # 解析输入文件
            data_list = self._parse_input_file(request.input_text_file)
            if not data_list:
                return SemanticEncodingResponse(
                    success=False,
                    message="没有找到有效的音频数据"
                )
            
            print(f"找到 {len(data_list)} 个音频文件")
            
            # 处理数据
            all_results = []
            n_parts = request.config.n_parts
            
            if n_parts > 1:
                # 多进程处理
                batch_size = len(data_list) // n_parts + (1 if len(data_list) % n_parts else 0)
                batches = [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]
                
                with ProcessPoolExecutor(max_workers=n_parts) as executor:
                    futures = []
                    for i, batch in enumerate(batches):
                        future = executor.submit(
                            self._process_batch,
                            batch, request.cnhubert_dir, request.config, i
                        )
                        futures.append(future)
                    
                    for future in as_completed(futures):
                        batch_results = future.result()
                        all_results.extend(batch_results)
            else:
                # 单进程处理
                all_results = self._process_batch(data_list, request.cnhubert_dir, request.config)
            
            # 统计结果
            processed_count = sum(1 for _, semantic_str in all_results if semantic_str is not None)
            failed_count = len(all_results) - processed_count
            
            # 保存结果
            output_file = os.path.join(request.output_dir, f"6-name2semantic.tsv")
            
            if request.config.output_format == "tsv":
                # TSV格式
                lines = []
                for wav_name, semantic_str in all_results:
                    if semantic_str is not None:
                        lines.append(f"{wav_name}\t{semantic_str}")
                
                with open(output_file, "w", encoding="utf8") as f:
                    f.write("\n".join(lines))
                    
            elif request.config.output_format == "json":
                # JSON格式
                output_file = os.path.join(request.output_dir, f"6-name2semantic.json")
                semantic_data = {}
                for wav_name, semantic_str in all_results:
                    if semantic_str is not None:
                        semantic_data[wav_name] = semantic_str.split()
                
                with open(output_file, "w", encoding="utf8") as f:
                    json.dump(semantic_data, f, ensure_ascii=False, indent=2)
            
            processing_time = ttime() - start_time
            
            return SemanticEncodingResponse(
                success=True,
                message=f"语义编码完成，处理了 {processed_count} 个文件",
                output_file=output_file,
                processed_count=processed_count,
                failed_count=failed_count,
                processing_time=processing_time,
                details={
                    "total_input": len(data_list),
                    "model_version": self.version,
                    "device_used": device,
                    "parallel_parts": n_parts,
                    "output_format": request.config.output_format,
                    "resolved_pretrained_s2G": request.config.pretrained_s2G,
                    "resolved_s2config_path": request.config.s2config_path,
                }
            )
            
        except Exception as e:
            return SemanticEncodingResponse(
                success=False,
                message=f"处理失败: {str(e)}",
                processing_time=ttime() - start_time
            )
