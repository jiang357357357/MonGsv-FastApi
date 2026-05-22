#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 推理模块测试脚本

测试推理API的各项功能
"""

import os
import sys
import asyncio
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from Code.FastApi.Base.Inference.service import InferenceService, InferenceRequest, InferenceConfig
from Code.FastApi.Base.Inference.model_manager import ModelManager, ModelConfig
from Code.FastApi.Base.Inference.audio_processor import AudioProcessor
from Code.FastApi.Base.Inference.utils import (
    validate_text_input, detect_language, clean_text,
    split_text_by_punctuation, validate_model_files,
    get_system_info
)


def create_test_audio(duration: float = 3.0, sample_rate: int = 22050) -> str:
    """创建测试音频文件"""
    # 生成简单的正弦波
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t) * 0.3  # 440Hz正弦波
    
    # 保存到临时文件
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    sf.write(temp_file.name, audio, sample_rate)
    temp_file.close()
    
    return temp_file.name


def test_text_validation():
    """测试文本验证功能"""
    print("=== 测试文本验证 ===")
    
    test_cases = [
        ("你好，世界！", "zh"),
        ("Hello, world!", "en"),
        ("こんにちは、世界！", "ja"),
        ("", "zh"),  # 空文本
        ("a", "zh"),  # 过短文本
        ("很长的文本" * 100, "zh"),  # 过长文本
    ]
    
    for text, language in test_cases:
        result = validate_text_input(text, language)
        print(f"文本: '{text[:20]}...' 语言: {language}")
        print(f"  有效: {result['valid']}")
        if result['issues']:
            print(f"  问题: {result['issues']}")
        print()


def test_language_detection():
    """测试语言检测功能"""
    print("=== 测试语言检测 ===")
    
    test_texts = [
        "你好，世界！这是中文文本。",
        "Hello, world! This is English text.",
        "こんにちは、世界！これは日本語のテキストです。",
        "안녕하세요, 세계! 이것은 한국어 텍스트입니다.",
        "Mixed text 中英文混合 テキスト",
    ]
    
    for text in test_texts:
        detected = detect_language(text)
        print(f"文本: {text}")
        print(f"检测语言: {detected}")
        print()


def test_text_cleaning():
    """测试文本清理功能"""
    print("=== 测试文本清理 ===")
    
    test_texts = [
        "  你好，世界！  ",
        "Hello,    world!   Multiple   spaces.",
        "文本\t包含\n特殊\r字符",
        '中文"标点"符号\'测试\'',
    ]
    
    for text in test_texts:
        cleaned = clean_text(text, "zh")
        print(f"原文: '{text}'")
        print(f"清理后: '{cleaned}'")
        print()


def test_text_splitting():
    """测试文本分割功能"""
    print("=== 测试文本分割 ===")
    
    long_text = "这是一个很长的文本。它包含多个句子！每个句子都有不同的标点符号？我们需要将它分割成更小的片段；这样处理起来更容易，也更高效。"
    
    segments = split_text_by_punctuation(long_text, max_length=20)
    print(f"原文: {long_text}")
    print("分割结果:")
    for i, segment in enumerate(segments, 1):
        print(f"  {i}: {segment}")
    print()


def test_audio_processor():
    """测试音频处理器"""
    print("=== 测试音频处理器 ===")
    
    processor = AudioProcessor()
    
    # 创建测试音频
    test_audio_path = create_test_audio()
    
    try:
        # 测试音频加载
        audio, sr = processor.load_audio(test_audio_path)
        print(f"音频加载成功: 长度={len(audio)}, 采样率={sr}")
        
        # 测试音频验证
        validation = processor.validate_audio(audio, sr)
        print(f"音频验证: 有效={validation['valid']}")
        if validation['issues']:
            print(f"  问题: {validation['issues']}")
        
        # 测试音频特征提取
        features = processor.extract_features(audio, sr)
        print(f"音频特征: {features}")
        
        # 测试音频处理
        normalized = processor.normalize_audio(audio)
        trimmed = processor.trim_silence(audio, sr)
        padded = processor.add_padding(audio, sr)
        
        print(f"归一化后长度: {len(normalized)}")
        print(f"去静音后长度: {len(trimmed)}")
        print(f"填充后长度: {len(padded)}")
        
        # 测试Base64转换
        base64_data = processor.audio_to_base64(audio, sr)
        decoded_audio, decoded_sr = processor.base64_to_audio(base64_data)
        print(f"Base64转换成功: 原长度={len(audio)}, 解码长度={len(decoded_audio)}")
        
    finally:
        # 清理测试文件
        if os.path.exists(test_audio_path):
            os.unlink(test_audio_path)
    
    print()


def test_model_manager():
    """测试模型管理器"""
    print("=== 测试模型管理器 ===")
    
    # 创建临时目录作为模型目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        manager = ModelManager(models_dir=temp_dir)
        
        # 测试模型注册（使用虚拟路径）
        config = ModelConfig(
            name="test_model",
            gpt_path="/fake/path/gpt.ckpt",
            sovits_path="/fake/path/sovits.pth",
            description="测试模型"
        )
        
        # 由于文件不存在，注册会失败，这是预期的
        success = manager.register_model(config)
        print(f"模型注册结果: {success} (预期失败，因为文件不存在)")
        
        # 测试模型列表
        models = manager.list_models()
        print(f"模型列表: {len(models)} 个模型")
        
        # 测试统计信息
        stats = manager.get_model_statistics()
        print(f"统计信息: {stats}")
        
    finally:
        # 清理临时目录
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print()


async def test_inference_api():
    """测试推理API"""
    print("=== 测试推理API ===")
    
    try:
        # 初始化API（可能会失败，因为没有真实的GPT-SoVITS环境）
        api = InferenceService()
        print("推理API初始化成功")
        
        # 获取模型信息
        model_info = api.get_model_info()
        print(f"模型信息: {model_info}")
        
        # 获取支持的语言和格式
        languages = api.get_supported_languages()
        formats = api.get_supported_formats()
        print(f"支持的语言: {languages}")
        print(f"支持的格式: {formats}")
        
        # 创建测试音频
        test_audio_path = create_test_audio()
        
        try:
            # 创建推理请求
            config = InferenceConfig(
                top_k=20,
                top_p=0.6,
                temperature=0.6,
                how_to_cut="不切"
            )
            
            request = InferenceRequest(
                text="你好，这是一个测试文本。",
                text_language="zh",
                ref_audio_path=test_audio_path,
                prompt_text="测试参考文本",
                prompt_language="zh",
                config=config,
                return_base64=True
            )
            
            # 执行推理（使用占位实现）
            response = await api.inference(request)
            print(f"推理结果: 成功={response.success}")
            if response.success:
                print(f"  处理时间: {response.processing_time:.2f}s")
                print(f"  音频时长: {response.duration:.2f}s")
                print(f"  采样率: {response.sample_rate}")
                print(f"  文本片段: {response.text_segments}")
                print(f"  Base64数据长度: {len(response.audio_data) if response.audio_data else 0}")
            else:
                print(f"  错误信息: {response.message}")
        
        finally:
            # 清理测试文件
            if os.path.exists(test_audio_path):
                os.unlink(test_audio_path)
    
    except Exception as e:
        print(f"推理API测试失败: {e}")
        print("这是预期的，因为没有真实的GPT-SoVITS环境")
    
    print()


def test_model_validation():
    """测试模型验证功能"""
    print("=== 测试模型验证 ===")
    
    # 测试不存在的文件
    result = validate_model_files("/fake/gpt.ckpt", "/fake/sovits.pth")
    print("不存在文件的验证结果:")
    print(f"  有效: {result['valid']}")
    print(f"  问题: {result['issues']}")
    print()


def test_system_info():
    """测试系统信息获取"""
    print("=== 测试系统信息 ===")
    
    info = get_system_info()
    print("系统信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()


def run_all_tests():
    """运行所有测试"""
    print("开始运行GPT-SoVITS推理模块测试")
    print("=" * 50)
    
    # 文本处理测试
    test_text_validation()
    test_language_detection()
    test_text_cleaning()
    test_text_splitting()
    
    # 音频处理测试
    test_audio_processor()
    
    # 模型管理测试
    test_model_manager()
    
    # 模型验证测试
    test_model_validation()
    
    # 系统信息测试
    test_system_info()
    
    # 推理API测试（异步）
    asyncio.run(test_inference_api())
    
    print("=" * 50)
    print("所有测试完成")


if __name__ == "__main__":
    run_all_tests()
