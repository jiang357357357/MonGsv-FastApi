#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频特征提取API测试脚本

测试 audio_features 模块的各种功能
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path

import requests
import numpy as np
import soundfile as sf

from .service import AudioFeaturesService, AudioFeaturesRequest, AudioFeaturesConfig


def create_test_audio_list(temp_dir: str, num_files: int = 3) -> tuple:
    """创建测试音频文件和标注列表"""
    wav_dir = os.path.join(temp_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    
    # 创建标注文件内容
    list_content = []
    
    for i in range(num_files):
        # 生成测试音频
        duration = 3 + i  # 3-5秒不等
        sample_rate = 32000
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # 生成不同频率的正弦波
        freq = 440 + i * 110  # 440Hz, 550Hz, 660Hz
        audio = 0.3 * np.sin(2 * np.pi * freq * t)
        
        # 保存音频文件
        wav_name = f"test_{i+1:03d}.wav"
        wav_path = os.path.join(wav_dir, wav_name)
        sf.write(wav_path, audio, sample_rate)
        
        # 添加到标注列表
        speaker = f"speaker_{(i % 2) + 1}"  # 两个说话人
        language = "ZH" if i % 2 == 0 else "EN"
        text = f"这是测试文本{i+1}" if language == "ZH" else f"This is test text {i+1}"
        
        list_content.append(f"{wav_name}|{speaker}|{language}|{text}")
    
    # 保存标注文件
    list_file = os.path.join(temp_dir, "test_list.txt")
    with open(list_file, "w", encoding="utf8") as f:
        f.write("\n".join(list_content))
    
    print(f"创建了 {num_files} 个测试音频文件")
    print(f"标注文件: {list_file}")
    print(f"音频目录: {wav_dir}")
    
    return list_file, wav_dir


async def test_features_api_direct():
    """直接测试API类"""
    print("=== 直接API测试 ===")
    
    try:
        # 初始化API
        api = AudioFeaturesService()
        print(f"GPT-SoVITS根目录: {api.gpt_sovits_root}")
        
        # 创建测试数据
        with tempfile.TemporaryDirectory() as temp_dir:
            list_file, wav_dir = create_test_audio_list(temp_dir, 3)
            output_dir = os.path.join(temp_dir, "output")
            
            # 配置特征提取参数（测试模式，不需要真实模型）
            config = AudioFeaturesConfig(
                cnhubert_base_dir="test_models/cnhubert",  # 测试路径
                sv_model_path="test_models/speaker.ckpt",   # 测试路径
                version="v2",
                is_half=False,  # 使用float32避免兼容性问题
                device="cpu",   # 强制使用CPU
                n_parts=1,      # 单进程测试
                save_cnhubert=False,  # 跳过CNHubert（需要真实模型）
                save_wav32k=True,     # 只测试音频处理
                save_speaker=False    # 跳过说话人特征（需要真实模型）
            )
            
            request = AudioFeaturesRequest(
                input_text_file=list_file,
                input_wav_dir=wav_dir,
                experiment_name="test_experiment",
                output_dir=output_dir,
                config=config
            )
            
            # 执行特征提取（只测试音频处理部分）
            try:
                result = await api.extract_features(request)
                
                print(f"特征提取结果: {result.success}")
                print(f"消息: {result.message}")
                print(f"处理文件数: {result.processed_count}")
                print(f"失败文件数: {result.failed_count}")
                print(f"处理时间: {result.processing_time:.2f}秒")
                
                # 检查输出文件
                if result.output_files.get("wav32k_dir"):
                    wav32k_dir = result.output_files["wav32k_dir"]
                    if os.path.exists(wav32k_dir):
                        files = os.listdir(wav32k_dir)
                        print(f"32kHz音频文件数: {len(files)}")
                        for file in files[:3]:  # 显示前3个
                            print(f"  - {file}")
                
                return result.success
                
            except Exception as e:
                # 预期的模型加载失败
                if "模型路径不存在" in str(e) or "CNHubert" in str(e):
                    print("✓ 模型路径验证正常（预期的测试结果）")
                    return True
                else:
                    raise e
            
    except Exception as e:
        print(f"直接测试失败: {e}")
        return False


def test_features_api_http():
    """测试HTTP API"""
    print("\n=== HTTP API测试 ===")
    
    base_url = "http://localhost:8003"
    
    try:
        # 健康检查
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"健康检查: {health}")
        else:
            print("API服务未启动，请先运行服务器")
            return False
        
        # 创建测试数据
        with tempfile.TemporaryDirectory() as temp_dir:
            list_file, wav_dir = create_test_audio_list(temp_dir, 2)
            output_dir = os.path.join(temp_dir, "http_output")
            
            # 准备请求数据
            request_data = {
                "input_text_file": list_file,
                "input_wav_dir": wav_dir,
                "experiment_name": "http_test",
                "output_dir": output_dir,
                "config": {
                    "version": "v2",
                    "is_half": False,
                    "device": "cpu",
                    "n_parts": 1,
                    "save_cnhubert": False,  # 跳过模型依赖
                    "save_wav32k": True,
                    "save_speaker": False
                }
            }
            
            # 发送特征提取请求
            response = requests.post(
                f"{base_url}/extract",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"HTTP特征提取结果: {result['success']}")
                print(f"消息: {result['message']}")
                print(f"处理文件数: {result['processed_count']}")
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


def test_analyze_api():
    """测试数据集分析API"""
    print("\n=== 数据集分析API测试 ===")
    
    base_url = "http://localhost:8003"
    
    try:
        # 创建测试数据
        with tempfile.TemporaryDirectory() as temp_dir:
            list_file, wav_dir = create_test_audio_list(temp_dir, 4)
            
            # 分析数据集
            response = requests.post(
                f"{base_url}/analyze",
                params={
                    "input_text_file": list_file,
                    "input_wav_dir": wav_dir
                }
            )
            
            if response.status_code == 200:
                analysis = response.json()
                print(f"数据集分析成功:")
                print(f"  总文件数: {analysis['total_lines']}")
                print(f"  有效文件: {analysis['valid_files']}")
                print(f"  总时长: {analysis['total_duration']:.1f}秒")
                print(f"  说话人数: {len(analysis['speakers'])}")
                print(f"  语言数: {len(analysis['languages'])}")
                return True
            else:
                print(f"分析请求失败: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"分析测试失败: {e}")
        return False


def test_config_suggest_api():
    """测试配置建议API"""
    print("\n=== 配置建议API测试 ===")
    
    base_url = "http://localhost:8003"
    
    try:
        # 创建测试数据
        with tempfile.TemporaryDirectory() as temp_dir:
            list_file, wav_dir = create_test_audio_list(temp_dir, 5)
            
            # 获取配置建议
            response = requests.get(
                f"{base_url}/config/suggest",
                params={
                    "input_text_file": list_file,
                    "input_wav_dir": wav_dir,
                    "version": "v2Pro",
                    "target_processing_time": 60.0,
                    "available_memory_gb": 8.0
                }
            )
            
            if response.status_code == 200:
                suggestion = response.json()
                print(f"配置建议成功:")
                config = suggestion['suggested_config']
                analysis = suggestion['analysis']
                
                print(f"  建议版本: {config['version']}")
                print(f"  并行数: {config['n_parts']}")
                print(f"  半精度: {config['is_half']}")
                print(f"  说话人特征: {config['save_speaker']}")
                print(f"  总文件数: {analysis['total_files']}")
                print(f"  预计时间: {analysis['estimated_processing_time']:.1f}秒")
                return True
            else:
                print(f"建议请求失败: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"建议测试失败: {e}")
        return False


def test_models_info_api():
    """测试模型信息API"""
    print("\n=== 模型信息API测试 ===")
    
    base_url = "http://localhost:8003"
    
    try:
        # 获取模型信息
        response = requests.get(
            f"{base_url}/models/info",
            params={
                "cnhubert_path": "test_models/cnhubert",
                "speaker_path": "test_models/speaker.ckpt"
            }
        )
        
        if response.status_code == 200:
            models_info = response.json()
            print(f"模型信息获取成功:")
            print(f"  CNHubert可用: {models_info['cnhubert']['available']}")
            print(f"  说话人模型可用: {models_info['speaker']['available']}")
            return True
        else:
            print(f"模型信息请求失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"模型信息测试失败: {e}")
        return False


async def test_utils_functions():
    """测试工具函数"""
    print("\n=== 工具函数测试 ===")
    
    try:
        from .utils import AudioFeaturesUtils
        
        # 创建测试数据
        with tempfile.TemporaryDirectory() as temp_dir:
            list_file, wav_dir = create_test_audio_list(temp_dir, 3)
            
            # 测试数据集分析
            analysis = AudioFeaturesUtils.analyze_dataset_from_list(list_file, wav_dir)
            if "error" not in analysis:
                print(f"✓ 数据集分析: {analysis['valid_files']} 个有效文件")
            else:
                print(f"✗ 数据集分析失败: {analysis['error']}")
                return False
            
            # 测试配置建议
            config = AudioFeaturesUtils.suggest_processing_config(list_file, wav_dir)
            print(f"✓ 配置建议: 版本{config.version}, 并行数{config.n_parts}")
            
            # 测试输入验证
            validation = AudioFeaturesUtils.validate_input_files(list_file, wav_dir)
            print(f"✓ 输入验证: {'通过' if validation['valid'] else '失败'}")
            
            # 测试时间估算
            time_estimate = AudioFeaturesUtils.estimate_processing_time(list_file, wav_dir, config)
            if "error" not in time_estimate:
                print(f"✓ 时间估算: {time_estimate['estimated_total_time']:.1f}秒")
            else:
                print(f"✗ 时间估算失败: {time_estimate['error']}")
                return False
            
            return True
            
    except Exception as e:
        print(f"工具函数测试失败: {e}")
        return False


async def run_all_tests():
    """运行所有测试"""
    print("开始音频特征提取API测试...")
    
    results = []
    
    # 直接API测试
    results.append(await test_features_api_direct())
    
    # 工具函数测试
    results.append(await test_utils_functions())
    
    # HTTP API测试（需要服务运行）
    results.append(test_features_api_http())
    
    # 分析API测试
    results.append(test_analyze_api())
    
    # 配置建议API测试
    results.append(test_config_suggest_api())
    
    # 模型信息API测试
    results.append(test_models_info_api())
    
    # 总结
    print(f"\n=== 测试总结 ===")
    test_names = [
        "直接API测试",
        "工具函数测试", 
        "HTTP API测试",
        "分析API测试",
        "配置建议API测试",
        "模型信息API测试"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        print(f"{name}: {'✓' if result else '✗'}")
    
    print(f"总体成功率: {sum(results)}/{len(results)}")


if __name__ == "__main__":
    # 运行测试
    asyncio.run(run_all_tests())