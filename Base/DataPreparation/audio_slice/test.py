#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频切分API测试脚本

测试 audio_slice 模块的各种功能
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path

import requests
import numpy as np
import soundfile as sf

from .service import AudioSliceService, SliceRequest, SliceConfig


def create_test_audio(duration=10, sample_rate=32000, output_path="test_audio.wav"):
    """创建测试音频文件"""
    # 生成包含静音段的测试音频
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # 创建有声段和静音段交替的音频
    audio = np.zeros_like(t)
    
    # 添加几段有声音频（正弦波）
    segments = [
        (0, 2),      # 0-2秒：有声
        (2, 3),      # 2-3秒：静音
        (3, 6),      # 3-6秒：有声
        (6, 7),      # 6-7秒：静音
        (7, 10),     # 7-10秒：有声
    ]
    
    for start, end in segments:
        start_idx = int(start * sample_rate)
        end_idx = int(end * sample_rate)
        if end_idx - start_idx > 0:
            segment_t = t[start_idx:end_idx]
            # 有声段：440Hz正弦波
            if (end - start) > 1:  # 有声段
                audio[start_idx:end_idx] = 0.5 * np.sin(2 * np.pi * 440 * segment_t[0:end_idx-start_idx])
    
    # 保存音频
    sf.write(output_path, audio, sample_rate)
    print(f"创建测试音频: {output_path}")
    return output_path


async def test_slice_api_direct():
    """直接测试API类"""
    print("=== 直接API测试 ===")
    
    try:
        # 初始化API
        api = AudioSliceService()
        print(f"GPT-SoVITS根目录: {api.gpt_sovits_root}")
        
        # 创建测试音频
        with tempfile.TemporaryDirectory() as temp_dir:
            input_audio = os.path.join(temp_dir, "test_input.wav")
            output_dir = os.path.join(temp_dir, "output")
            
            create_test_audio(duration=10, output_path=input_audio)
            
            # 配置切分参数
            config = SliceConfig(
                threshold=-40.0,    # 较低的阈值，更容易检测到静音
                min_length=2000,    # 最短2秒
                min_interval=500,   # 最短间隔0.5秒
                hop_size=10,
                max_sil_kept=200,   # 保留0.2秒静音
                max_volume=0.9,
                alpha=0.25
            )
            
            request = SliceRequest(
                input_path=input_audio,
                output_dir=output_dir,
                config=config
            )
            
            # 执行切分
            result = await api.slice_audio(request)
            
            print(f"切分结果: {result.success}")
            print(f"消息: {result.message}")
            print(f"输出文件数: {len(result.output_files)}")
            
            for i, output_file in enumerate(result.output_files):
                print(f"  {i+1}. {os.path.basename(output_file)}")
            
            return result.success
            
    except Exception as e:
        print(f"直接测试失败: {e}")
        return False


def test_slice_api_http():
    """测试HTTP API"""
    print("\n=== HTTP API测试 ===")
    
    base_url = "http://localhost:8001"
    
    try:
        # 健康检查
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"健康检查: {health}")
        else:
            print("API服务未启动，请先运行服务器")
            return False
        
        # 创建测试音频
        with tempfile.TemporaryDirectory() as temp_dir:
            input_audio = os.path.join(temp_dir, "test_http.wav")
            output_dir = os.path.join(temp_dir, "http_output")
            
            create_test_audio(duration=8, output_path=input_audio)
            
            # 准备请求数据
            request_data = {
                "input_path": input_audio,
                "output_dir": output_dir,
                "config": {
                    "threshold": -35.0,
                    "min_length": 3000,
                    "min_interval": 400,
                    "hop_size": 10,
                    "max_sil_kept": 300,
                    "max_volume": 0.9,
                    "alpha": 0.25,
                    "n_parts": 1
                }
            }
            
            # 发送切分请求
            response = requests.post(
                f"{base_url}/slice",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"HTTP切分结果: {result['success']}")
                print(f"消息: {result['message']}")
                print(f"输出文件数: {len(result['output_files'])}")
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
    print("\n=== 文件上传API测试 ===")
    
    base_url = "http://localhost:8001"
    
    try:
        # 创建测试音频
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            create_test_audio(duration=6, output_path=temp_file.name)
            
            # 上传并切分
            with open(temp_file.name, "rb") as audio_file:
                files = {"file": ("test_upload.wav", audio_file, "audio/wav")}
                data = {
                    "threshold": -38.0,
                    "min_length": 2500,
                    "min_interval": 300,
                    "hop_size": 10,
                    "max_sil_kept": 400,
                    "max_volume": 0.9,
                    "alpha": 0.25
                }
                
                response = requests.post(
                    f"{base_url}/slice/upload",
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    # 保存返回的zip文件
                    zip_path = "test_output.zip"
                    with open(zip_path, "wb") as f:
                        f.write(response.content)
                    print(f"上传切分成功，输出文件: {zip_path}")
                    
                    # 清理临时文件
                    os.unlink(temp_file.name)
                    return True
                else:
                    print(f"上传切分失败: {response.status_code}")
                    print(f"错误信息: {response.text}")
                    return False
                    
    except Exception as e:
        print(f"上传测试失败: {e}")
        return False


async def run_all_tests():
    """运行所有测试"""
    print("开始音频切分API测试...")
    
    results = []
    
    # 直接API测试
    results.append(await test_slice_api_direct())
    
    # HTTP API测试（需要服务运行）
    results.append(test_slice_api_http())
    
    # 上传API测试（需要服务运行）
    results.append(test_upload_api())
    
    # 总结
    print(f"\n=== 测试总结 ===")
    print(f"直接API测试: {'✓' if results[0] else '✗'}")
    print(f"HTTP API测试: {'✓' if results[1] else '✗'}")
    print(f"上传API测试: {'✓' if results[2] else '✗'}")
    print(f"总体成功率: {sum(results)}/{len(results)}")


if __name__ == "__main__":
    # 运行测试
    asyncio.run(run_all_tests())