"""
GPT-SoVITS 文本处理 API

实现步骤1a：文本特征提取
- 文本清理和音素转换
- BERT特征提取（中文）
- 多语言支持
"""

import os
import sys
import torch
import traceback
import shutil
from time import time as ttime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed

from Code.FastApi.Base.gpt_sovits_env import setup_gpt_sovits_paths


current_dir = Path(__file__).resolve().parent
gpt_sovits_root = setup_gpt_sovits_paths(current_dir)

try:
    from text.cleaner import clean_text
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    from tools.my_utils import clean_path
except ImportError as e:
    print(f"警告: 无法导入GPT-SoVITS模块: {e}")
    print("请确保GPT-SoVITS路径正确")


@dataclass
class TextProcessingConfig:
    """文本处理配置"""
    # BERT模型配置
    bert_pretrained_dir: str = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
    
    # 处理配置
    version: str = "v2"  # GPT-SoVITS版本
    is_half: bool = True  # 是否使用半精度
    device: str = "auto"  # 设备选择
    
    # 并行配置
    n_parts: int = 1  # 并行处理数
    
    # 语言映射
    language_mapping: Dict[str, str] = field(default_factory=lambda: {
        "ZH": "zh", "zh": "zh",
        "JP": "ja", "jp": "ja", "JA": "ja", "ja": "ja",
        "EN": "en", "en": "en", "En": "en",
        "KO": "ko", "Ko": "ko", "ko": "ko",
        "yue": "yue", "YUE": "yue", "Yue": "yue",
    })


@dataclass
class TextProcessingRequest:
    """文本处理请求"""
    input_text_file: str  # 输入标注文件路径
    input_wav_dir: str    # 音频目录路径
    experiment_name: str  # 实验名称
    output_dir: str       # 输出目录
    config: TextProcessingConfig = field(default_factory=TextProcessingConfig)


@dataclass
class TextProcessingResponse:
    """文本处理响应"""
    success: bool
    message: str
    output_files: Dict[str, str] = field(default_factory=dict)
    processed_count: int = 0
    failed_count: int = 0
    processing_time: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class TextProcessingService:
    """文本处理API类"""
    
    def __init__(self, gpt_sovits_root: str = None):
        """
        初始化文本处理API
        
        Args:
            gpt_sovits_root: GPT-SoVITS根目录路径
        """
        self.gpt_sovits_root = gpt_sovits_root
        self.tokenizer = None
        self.bert_model = None
        self.device = None
        self.gpt_sovits_root = str(gpt_sovits_root or globals().get("gpt_sovits_root"))

    def _resolve_runtime_path(self, path_str: str) -> str:
        """把配置里的相对路径收敛成 GPT-SoVITS 根目录下的绝对路径。"""
        if not path_str:
            return path_str

        candidate = Path(path_str)
        if candidate.is_absolute():
            return str(candidate)

        base_root = Path(self.gpt_sovits_root or gpt_sovits_root)
        return str((base_root / candidate).resolve())

    def _ensure_text_runtime(self, config: TextProcessingConfig, todo: List[Tuple[str, str, str]]) -> None:
        """在真正处理前完成中文文本依赖校验，并注入 GPT-SoVITS 运行环境变量。"""
        has_chinese = any(language == "zh" for _, _, language in todo)
        if not has_chinese:
            return

        resolved_bert_path = self._resolve_runtime_path(config.bert_pretrained_dir)
        os.environ["bert_path"] = resolved_bert_path
        config.bert_pretrained_dir = resolved_bert_path

        if not os.path.exists(resolved_bert_path):
            raise FileNotFoundError(
                "中文文本处理依赖的 BERT 模型不存在: "
                f"{resolved_bert_path}\n"
                "请先补齐 GPT-SoVITS 预训练模型目录 `GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large`。"
            )
        
    def _setup_device(self, config: TextProcessingConfig) -> str:
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
    
    def _load_bert_model(self, config: TextProcessingConfig):
        """加载BERT模型"""
        if not os.path.exists(config.bert_pretrained_dir):
            raise FileNotFoundError(f"BERT模型路径不存在: {config.bert_pretrained_dir}")
            
        print(f"加载BERT模型: {config.bert_pretrained_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_pretrained_dir)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(config.bert_pretrained_dir)
        
        if config.is_half and torch.cuda.is_available():
            self.bert_model = self.bert_model.half().to(self.device)
        else:
            self.bert_model = self.bert_model.to(self.device)
    
    def _get_bert_feature(self, text: str, word2ph: List[int]) -> torch.Tensor:
        """
        提取BERT特征
        
        Args:
            text: 输入文本
            word2ph: 词到音素的映射
            
        Returns:
            BERT特征张量 (1024, phone_length)
        """
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]

        assert len(word2ph) == len(text), f"word2ph长度({len(word2ph)})与文本长度({len(text)})不匹配"
        
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)

        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T
    
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
    
    def _process_text_batch(self, 
                           data: List[Tuple[str, str, str]], 
                           bert_dir: str,
                           config: TextProcessingConfig,
                           i_part: int = 0) -> List[Tuple[str, str, List[int], str]]:
        """
        处理文本批次
        
        Args:
            data: 文本数据列表 [(name, text, language), ...]
            bert_dir: BERT特征输出目录
            config: 处理配置
            i_part: 进程编号
            
        Returns:
            处理结果列表 [(name, phones, word2ph, norm_text), ...]
        """
        results = []
        
        for name, text, language in data:
            try:
                name = clean_path(name)
                name = os.path.basename(name)
                print(f"处理文本: {name}")
                
                # 文本清理和音素转换
                phones, word2ph, norm_text = clean_text(
                    text.replace("%", "-").replace("￥", ","), 
                    language, 
                    config.version
                )
                
                # 提取BERT特征（仅中文）
                if language == "zh" and self.bert_model is not None:
                    bert_path = os.path.join(bert_dir, f"{name}.pt")
                    if not os.path.exists(bert_path):
                        bert_feature = self._get_bert_feature(norm_text, word2ph)
                        assert bert_feature.shape[-1] == len(phones), \
                            f"BERT特征长度({bert_feature.shape[-1]})与音素长度({len(phones)})不匹配"
                        self._save_tensor(bert_feature, bert_path, i_part)
                
                phones_str = " ".join(phones)
                results.append([name, phones_str, word2ph, norm_text])
                
            except Exception as e:
                print(f"处理失败 {name}: {text}, 错误: {traceback.format_exc()}")
                
        return results
    
    def _parse_input_file(self, input_file: str, config: TextProcessingConfig) -> List[Tuple[str, str, str]]:
        """
        解析输入标注文件
        
        Args:
            input_file: 输入文件路径
            config: 处理配置
            
        Returns:
            解析结果 [(wav_name, text, language), ...]
        """
        todo = []
        
        with open(input_file, "r", encoding="utf8") as f:
            lines = f.read().strip("\n").split("\n")
        
        for line in lines:
            try:
                parts = line.split("|")
                if len(parts) >= 4:
                    wav_name, spk_name, language, text = parts[:4]
                    
                    # 语言映射
                    if language in config.language_mapping:
                        mapped_language = config.language_mapping[language]
                        todo.append([wav_name, text, mapped_language])
                    else:
                        print(f"警告: 不支持的语言 {language} (文件: {wav_name})")
                        
            except Exception as e:
                print(f"解析行失败: {line}, 错误: {traceback.format_exc()}")
                
        return todo
    
    async def process_text(self, request: TextProcessingRequest) -> TextProcessingResponse:
        """
        异步处理文本
        
        Args:
            request: 处理请求
            
        Returns:
            处理响应
        """
        return self.process_text_sync(request)

    async def process(self, request: TextProcessingRequest) -> TextProcessingResponse:
        """基础层统一入口。"""
        return await self.process_text(request)

    async def extract_text_features(self, request: TextProcessingRequest) -> TextProcessingResponse:
        """兼容旧调用方的方法名。"""
        return await self.process_text(request)
    
    def process_text_sync(self, request: TextProcessingRequest) -> TextProcessingResponse:
        """
        同步处理文本
        
        Args:
            request: 处理请求
            
        Returns:
            处理响应
        """
        start_time = ttime()
        
        try:
            # 验证输入
            if not os.path.exists(request.input_text_file):
                return TextProcessingResponse(
                    success=False,
                    message=f"输入文件不存在: {request.input_text_file}"
                )
            
            if not os.path.exists(request.input_wav_dir):
                return TextProcessingResponse(
                    success=False,
                    message=f"音频目录不存在: {request.input_wav_dir}"
                )
            
            # 设置设备
            device = self._setup_device(request.config)
            print(f"使用设备: {device}")
            
            # 解析输入文件
            todo = self._parse_input_file(request.input_text_file, request.config)
            if not todo:
                return TextProcessingResponse(
                    success=False,
                    message="没有找到有效的文本数据"
                )

            # 创建输出目录
            os.makedirs(request.output_dir, exist_ok=True)
            bert_dir = os.path.join(request.output_dir, "3-bert")
            os.makedirs(bert_dir, exist_ok=True)

            self._ensure_text_runtime(request.config, todo)

            # 加载BERT模型（如果需要）
            need_bert = any(language == "zh" for _, _, language in todo)
            if need_bert:
                self._load_bert_model(request.config)
            
            print(f"找到 {len(todo)} 条文本数据")
            
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
                            self._process_text_batch,
                            batch, bert_dir, request.config, i
                        )
                        futures.append(future)
                    
                    for future in as_completed(futures):
                        batch_results = future.result()
                        all_results.extend(batch_results)
            else:
                # 单进程处理
                all_results = self._process_text_batch(todo, bert_dir, request.config)
            
            # 保存结果
            output_lines = []
            for name, phones, word2ph, norm_text in all_results:
                output_lines.append(f"{name}\t{phones}\t{word2ph}\t{norm_text}")
            
            output_file = os.path.join(request.output_dir, "2-name2text.txt")
            with open(output_file, "w", encoding="utf8") as f:
                f.write("\n".join(output_lines) + "\n")
            
            processing_time = ttime() - start_time
            
            return TextProcessingResponse(
                success=True,
                message=f"文本处理完成，处理了 {len(all_results)} 条数据",
                output_files={
                    "text_file": output_file,
                    "bert_dir": bert_dir if need_bert else None
                },
                processed_count=len(all_results),
                failed_count=len(todo) - len(all_results),
                processing_time=processing_time,
                details={
                    "total_input": len(todo),
                    "bert_features_extracted": need_bert,
                    "device_used": device,
                    "parallel_parts": n_parts
                }
            )
            
        except Exception as e:
            return TextProcessingResponse(
                success=False,
                message=f"处理失败: {str(e)}",
                processing_time=ttime() - start_time
            )
