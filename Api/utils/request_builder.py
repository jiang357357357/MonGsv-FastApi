#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
请求构建器

提供便捷的API请求构建功能
"""

from typing import Optional, Dict, Any
from ..core.config import APIConfig, get_default_config
from ..models import *


class RequestBuilder:
    """请求构建器"""
    
    def __init__(self, config: Optional[APIConfig] = None):
        """
        初始化请求构建器
        
        Args:
            config: API配置实例
        """
        self.config = config or get_default_config()
    
    def audio_slice_request(self, 
                           input_path: str,
                           output_dir: str,
                           threshold: float = -34.0,
                           min_length: int = 4000,
                           **kwargs) -> AudioSliceRequest:
        """构建音频切分请求"""
        config = AudioSliceConfig(
            threshold=threshold,
            min_length=min_length,
            **kwargs
        )
        
        return AudioSliceRequest(
            input_path=input_path,
            output_dir=output_dir,
            config=config,
            timeout=self.config.get("timeout", 300)
        )
    
    def asr_request(self,
                   input_path: str,
                   output_file: str,
                   model_type: str = "funasr",
                   language: Optional[str] = None,
                   **kwargs) -> ASRRequest:
        """构建ASR识别请求"""
        language = language or self.config.get("default_language", "zh")
        
        config = ASRConfig(
            model_type=model_type,
            language=language,
            **kwargs
        )
        
        return ASRRequest(
            input_path=input_path,
            output_file=output_file,
            config=config,
            timeout=self.config.get("timeout", 300)
        )
    
    def text_processing_request(self,
                               list_file: str,
                               output_dir: str,
                               language: Optional[str] = None,
                               **kwargs) -> TextProcessingRequest:
        """构建文本处理请求"""
        language = language or self.config.get("default_language", "zh")
        
        config = TextProcessingConfig(
            language=language,
            **kwargs
        )
        
        return TextProcessingRequest(
            list_file=list_file,
            output_dir=output_dir,
            config=config,
            timeout=self.config.get("timeout", 600)
        )
    
    def audio_features_request(self,
                              list_file: str,
                              output_dir: str,
                              version: Optional[str] = None,
                              **kwargs) -> AudioFeaturesRequest:
        """构建音频特征请求"""
        version = version or self.config.get("default_version", "v2Pro")
        
        config = AudioFeaturesConfig(
            version=version,
            **kwargs
        )
        
        return AudioFeaturesRequest(
            list_file=list_file,
            output_dir=output_dir,
            config=config,
            timeout=self.config.get("timeout", 1200)
        )
    
    def semantic_encoding_request(self,
                                 list_file: str,
                                 output_dir: str,
                                 version: Optional[str] = None,
                                 **kwargs) -> SemanticEncodingRequest:
        """构建语义编码请求"""
        version = version or self.config.get("default_version", "v2Pro")
        
        config = SemanticEncodingConfig(
            version=version,
            **kwargs
        )
        
        return SemanticEncodingRequest(
            list_file=list_file,
            output_dir=output_dir,
            config=config,
            timeout=self.config.get("timeout", 900)
        )
    
    def gpt_training_request(self,
                            exp_name: str,
                            exp_root: str,
                            **kwargs) -> GPTTrainingRequest:
        """构建GPT训练请求"""
        config = GPTTrainingConfig(**kwargs)
        
        return GPTTrainingRequest(
            exp_name=exp_name,
            exp_root=exp_root,
            config=config,
            timeout=self.config.get("timeout", 7200)
        )
    
    def sovits_training_request(self,
                               exp_name: str,
                               exp_root: str,
                               version: Optional[str] = None,
                               **kwargs) -> SoVITSTrainingRequest:
        """构建SoVITS训练请求"""
        version = version or self.config.get("default_version", "v2Pro")
        
        config = SoVITSTrainingConfig(
            version=version,
            **kwargs
        )
        
        return SoVITSTrainingRequest(
            exp_name=exp_name,
            exp_root=exp_root,
            config=config,
            timeout=self.config.get("timeout", 3600)
        )
    
    def inference_request(self,
                         text: str,
                         ref_audio_path: Optional[str] = None,
                         ref_audio_base64: Optional[str] = None,
                         text_language: Optional[str] = None,
                         **kwargs) -> InferenceRequest:
        """构建推理请求"""
        text_language = text_language or self.config.get("default_language", "zh")
        
        config = InferenceConfig(**kwargs)
        
        return InferenceRequest(
            text=text,
            text_language=text_language,
            ref_audio_path=ref_audio_path,
            ref_audio_base64=ref_audio_base64,
            config=config,
            timeout=self.config.get("timeout", 300)
        )
    
    def batch_inference_request(self,
                               texts: list,
                               output_dir: str,
                               ref_audio_path: Optional[str] = None,
                               text_language: Optional[str] = None,
                               **kwargs) -> BatchInferenceRequest:
        """构建批量推理请求"""
        text_language = text_language or self.config.get("default_language", "zh")
        
        config = InferenceConfig(**kwargs)
        
        return BatchInferenceRequest(
            texts=texts,
            text_language=text_language,
            ref_audio_path=ref_audio_path,
            output_dir=output_dir,
            config=config,
            timeout=self.config.get("timeout", 600)
        )
    
    def workflow_request(self,
                        project_name: str,
                        input_audio_dir: str,
                        output_dir: str,
                        language: Optional[str] = None,
                        version: Optional[str] = None,
                        **kwargs) -> WorkflowRequest:
        """构建工作流请求"""
        language = language or self.config.get("default_language", "zh")
        version = version or self.config.get("default_version", "v2Pro")
        
        config = WorkflowConfig(
            language=language,
            version=version,
            **kwargs
        )
        
        return WorkflowRequest(
            project_name=project_name,
            input_audio_dir=input_audio_dir,
            output_dir=output_dir,
            config=config,
            timeout=self.config.get("timeout", 3600)
        )
    
    def batch_request(self,
                     batch_name: str,
                     projects: list,
                     max_concurrent: int = 3,
                     **kwargs) -> BatchRequest:
        """构建批量处理请求"""
        config = WorkflowConfig(**kwargs)
        
        return BatchRequest(
            batch_name=batch_name,
            projects=projects,
            max_concurrent=max_concurrent,
            config=config,
            timeout=self.config.get("timeout", 7200)
        )


def get_request_builder(config: Optional[APIConfig] = None) -> RequestBuilder:
    """获取请求构建器实例"""
    return RequestBuilder(config)