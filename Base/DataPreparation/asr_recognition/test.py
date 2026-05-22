#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR语音识别API测试脚本

测试 asr_recognition 模块的各种功能
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path

import requests
import numpy as np
import soundfile as sf

from .service import ASRRecognitionService, ASRRequest, ASRConfig
from .utils import ASRUtils, create_test_audio_list


def create_test_audio(duration=10, sample_rate=16000, output_path="test_audio.wav", language="zh"):
    """创建测试音频文件"""
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # 根据语言生成不同频率的音频
    if language == "zh":
        # 中文：使用多个频率模拟声调
        frequencies = [220, 330, 440, 550]  # 不同声调
        audio = np.zeros_like(t)
        for i, freq in enumerate(frequencies):
            start_idx = int(i * len(t) / len(frequencies))
            end_idx = int((i + 1) * len(t) / len(frequencies))
            audio[start_idx:end_idx] = 0.3 * np.sin(2 * np.pi * freq * t[start_idx:end_idx])
    elif language == "en":
        # 英文：使用稳定频率
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    else:
        # 其他语言：使用混合频率
        audio = 0.2 * (np.sin(2 * np.pi * 330 * t) + np.sin(2 * np.pi * 660 * t))
    
    # 添加一些静音段
    silence_segments = [(2.0, 2.5), (5.0, 5.5), (8.0, 8.5)]
    for start, end in silence_segments:
        start_idx = int(start * sample_rate)
        end_idx = int(end * sample_rate)
        if end_idx <= len(audio):
            audio[start_idx:end_idx] = 0
    
    # 保存音频
    sf.write(output_path, audio, sample_rate)
    print(f"创建测试音频: {output_path} ({language})")
    return output_path


async def test_asr_api_direct():
    """直接测试API类"""
    print("=== 直接ASR API测试 ===")
    
    try:
        # 初始化API
        api = ASRRecognitionService()
        print(f"GPT-SoVITS根目录: {api.gpt_sovits_root}")
        print(f"支持的模型: {list(api.asr_models.keys())}")
        
        # 创建测试音频
        with tempfile.TemporaryDirectory() as temp_dir:
            input_audio = os.path.join(temp_dir, "test_zh.wav")
            output_dir = os.path.join(temp_dir, "output")
            
            create_test_audio(duration=8, output_path=input_audio, language="zh")
            
            # 获取智能配置建议
            config = api.suggest_config("zh")
            print(f"建议配置: {config.model_type}/{config.language}")
            
            request = ASRRequest(
                input_path=input_audio,
                output_dir=output_dir,
                config=config
            )
            
            # 执行识别
            result = await api.recognize_audio(request)
            
            print(f"识别结果: {result.success}")
            print(f"消息: {result.message}")
            print(f"处理时间: {result.processing_time:.2f}秒")
            
            if result.success:
                print(f"输出文件: {result.output_file}")
                print(f"识别条目数: {len(result.recognition_results)}")
                
                for i, item in enumerate(result.recognition_results[:3]):  # 显示前3条
                    print(f"  {i+1}. [{item['language']}] {item['text']}")
            
            return result.success
            
    except Exception as e:
        print(f"直接测试失败: {e}")
        return False


def test_asr_api_http():
    """测试HTTP API"""
    print("\n=== HTTP ASR API测试 ===")
    
    base_url = "http://localhost:8002"
    
    try:
        # 健康检查
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"健康检查: {health['status']}")
            print(f"支持的模型: {list(health['supported_models'].keys())}")
        else:
            print("API服务未启动，请先运行服务器")
            return False
        
        # 获取配置建议
        response = requests.post(f"{base_url}/suggest-config", params={"language": "zh"})
        if response.status_code == 200:
            suggested_config = response.json()
            print(f"建议配置: {suggested_config}")
        
        # 创建测试音频
        with tempfile.TemporaryDirectory() as temp_dir:
            input_audio = os.path.join(temp_dir, "test_http.wav")
            output_dir = os.path.join(temp_dir, "http_output")
            
            create_test_audio(duration=6, output_path=input_audio, language="zh")
            
            # 准备请求数据
            request_data = {
                "input_path": input_audio,
                "output_dir": output_dir,
                "config": {
                    "model_type": "funasr",
                    "model_size": "large",
                    "language": "zh",
                    "precision": "float32"
                }
            }
            
            # 发送识别请求
            response = requests.post(
                f"{base_url}/recognize",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"HTTP识别结果: {result['success']}")
                print(f"消息: {result['message']}")
                print(f"处理时间: {result['processing_time']:.2f}秒")
                return result['success']
            else:
                print(f"HTTP请求失败: {response.status_code}")
                print(f"错误信息: {response.text}")
                return False
                
    except requests.exceptions.ConnectionError:
        print("无法连接到API服务，请确保服务正在运行")
        return False
    except Exception as e:
        print(f"HTTP测试失败: {e}")
        return False


def test_upload_api():
    """测试文件上传API"""
    print("\n=== 文件上传ASR API测试 ===")
    
    base_url = "http://localhost:8002"
    
    try:
        # 创建测试音频
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            create_test_audio(duration=5, output_path=temp_file.name, language="en")
            
            # 上传并识别
            with open(temp_file.name, "rb") as audio_file:
                files = {"file": ("test_upload.wav", audio_file, "audio/wav")}
                data = {
                    "model_type": "faster_whisper",
                    "model_size": "large-v3",
                    "language": "en",
                    "precision": "float16"
                }
                
                response = requests.post(
                    f"{base_url}/recognize/upload",
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    # 保存返回的识别结果文件
                    result_path = "test_recognition_result.list"
                    with open(result_path, "wb") as f:
                        f.write(response.content)
                    print(f"上传识别成功，结果文件: {result_path}")
                    
                    # 解析结果
                    results = ASRUtils.parse_recognition_result(result_path)
                    print(f"识别条目数: {len(results)}")
                    
                    # 清理临时文件
                    os.unlink(temp_file.name)
                    return True
                else:
                    print(f"上传识别失败: {response.status_code}")
                    print(f"错误信息: {response.text}")
                    return False
                    
    except Exception as e:
        print(f"上传测试失败: {e}")
        return False


async def test_batch_recognition():
    """测试批量识别"""
    print("\n=== 批量识别测试 ===")
    
    try:
        api = ASRRecognitionService()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建多个测试音频文件
            input_dir = os.path.join(temp_dir, "input")
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(input_dir)
            
            test_files = [
                ("chinese_1.wav", "zh", 4),
                ("chinese_2.wav", "zh", 6),
                ("english_1.wav", "en", 5)
            ]
            
            for filename, lang, duration in test_files:
                file_path = os.path.join(input_dir, filename)
                create_test_audio(duration=duration, output_path=file_path, language=lang)
            
            # 批量分析
            batch_info = ASRUtils.batch_analyze_directory(input_dir)
            print(f"批量分析结果:")
            print(f"  总文件数: {batch_info['total_files']}")
            print(f"  总时长: {batch_info['total_duration']:.1f}秒")
            print(f"  语言分布: {batch_info['language_distribution']}")
            print(f"  主要语言: {batch_info['summary']['main_language']}")
            
            # 获取批量配置建议
            batch_config = batch_info['summary']['suggested_config']
            print(f"  建议配置: {batch_config.model_type}/{batch_config.language}")
            
            # 执行批量识别
            results = await api.batch_recognize(input_dir, output_dir, batch_config)
            
            success_count = sum(1 for r in results if r.success)
            print(f"批量识别完成: {success_count}/{len(results)} 成功")
            
            return success_count == len(results)
            
    except Exception as e:
        print(f"批量识别测试失败: {e}")
        return False


def test_utils_functions():
    """测试工具函数"""
    print("\n=== 工具函数测试 ===")
    
    try:
        # 测试语言检测
        test_filenames = [
            "chinese_audio_zh.wav",
            "english_speech_en.wav", 
            "japanese_voice_ja.wav",
            "cantonese_yue.wav",
            "unknown_file.wav"
        ]
        
        print("语言检测测试:")
        for filename in test_filenames:
            detected = ASRUtils.detect_language_from_filename(filename)
            print(f"  {filename} -> {detected}")
        
        # 测试配置建议
        print("\n配置建议测试:")
        for lang in ["zh", "en", "ja", "yue", "auto"]:
            config = ASRUtils.suggest_config_for_file("test.wav", lang)
            print(f"  {lang}: {config.model_type}/{config.model_size}/{config.precision}")
        
        # 创建测试音频并分析
        with tempfile.TemporaryDirectory() as temp_dir:
            test_audio = os.path.join(temp_dir, "analysis_test.wav")
            create_test_audio(duration=10, output_path=test_audio)
            
            info = ASRUtils.analyze_audio_file(test_audio)
            print(f"\n音频分析测试:")
            print(f"  时长: {info['duration']:.1f}秒")
            print(f"  采样率: {info['sample_rate']}Hz")
            print(f"  复杂度: {info['complexity']}")
            print(f"  估算词数: {info['estimated_words']}")
        
        return True
        
    except Exception as e:
        print(f"工具函数测试失败: {e}")
        return False


async def run_all_tests():
    """运行所有测试"""
    print("🎤 ASR语音识别模块测试")
    print("=" * 50)
    
    results = []
    
    try:
        # 直接API测试
        results.append(await test_asr_api_direct())
        
        # HTTP API测试（需要服务运行）
        results.append(test_asr_api_http())
        
        # 上传API测试（需要服务运行）
        results.append(test_upload_api())
        
        # 批量识别测试
        results.append(await test_batch_recognition())
        
        # 工具函数测试
        results.append(test_utils_functions())
        
        # 总结
        print(f"\n=== 测试总结 ===")
        test_names = ["直接API", "HTTP API", "上传API", "批量识别", "工具函数"]
        for i, (name, result) in enumerate(zip(test_names, results)):
            print(f"{name}测试: {'✓' if result else '✗'}")
        
        print(f"总体成功率: {sum(results)}/{len(results)}")
        
    except Exception as e:
        print(f"测试运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行测试
    asyncio.run(run_all_tests())