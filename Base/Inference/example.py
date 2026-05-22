#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 推理模块使用示例

展示如何使用推理API进行语音合成
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


def create_sample_audio(text: str = "这是一个示例音频", 
                       duration: float = 2.0, sample_rate: int = 22050) -> str:
    """创建示例音频文件"""
    print(f"创建示例音频: {text}")
    
    # 生成简单的音频信号（实际使用中应该是真实的语音录音）
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 生成多频率混合的信号，模拟语音
    frequencies = [220, 440, 660]  # 基频和谐波
    audio = np.zeros_like(t)
    
    for i, freq in enumerate(frequencies):
        amplitude = 0.3 / (i + 1)  # 递减幅度
        audio += amplitude * np.sin(2 * np.pi * freq * t)
    
    # 添加包络，模拟语音的起伏
    envelope = np.exp(-t * 0.5) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
    audio *= envelope
    
    # 保存到临时文件
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    sf.write(temp_file.name, audio, sample_rate)
    temp_file.close()
    
    print(f"示例音频已保存到: {temp_file.name}")
    return temp_file.name


async def example_basic_inference():
    """基础推理示例"""
    print("\n" + "="*50)
    print("示例1: 基础语音合成推理")
    print("="*50)
    
    try:
        # 初始化推理API
        api = InferenceService()
        print("✓ 推理API初始化成功")
        
        # 创建示例参考音频
        ref_audio_path = create_sample_audio("参考音频内容", duration=3.0)
        
        try:
            # 配置推理参数
            config = InferenceConfig(
                top_k=20,
                top_p=0.6,
                temperature=0.6,
                how_to_cut="不切",
                speed=1.0
            )
            
            # 创建推理请求
            request = InferenceRequest(
                text="你好，欢迎使用GPT-SoVITS语音合成系统！这是一个基础的推理示例。",
                text_language="zh",
                ref_audio_path=ref_audio_path,
                prompt_text="参考音频内容",
                prompt_language="zh",
                config=config,
                output_format="wav",
                return_base64=False
            )
            
            print(f"📝 合成文本: {request.text}")
            print(f"🎵 参考音频: {ref_audio_path}")
            print("🔄 开始推理...")
            
            # 执行推理
            response = await api.inference(request)
            
            if response.success:
                print("✅ 推理成功!")
                print(f"   处理时间: {response.processing_time:.2f}秒")
                print(f"   音频时长: {response.duration:.2f}秒")
                print(f"   采样率: {response.sample_rate}Hz")
                print(f"   文本片段数: {len(response.text_segments) if response.text_segments else 0}")
                
                if response.audio_path:
                    print(f"   输出文件: {response.audio_path}")
                    print("   💡 提示: 在实际应用中，您可以播放或进一步处理这个音频文件")
                
            else:
                print("❌ 推理失败:")
                print(f"   错误信息: {response.message}")
                if response.error_details:
                    print(f"   详细错误: {response.error_details}")
        
        finally:
            # 清理临时文件
            if os.path.exists(ref_audio_path):
                os.unlink(ref_audio_path)
                print(f"🗑️ 已清理临时文件: {ref_audio_path}")
    
    except Exception as e:
        print(f"❌ 示例执行失败: {e}")
        print("💡 这可能是因为没有安装完整的GPT-SoVITS环境")


async def example_batch_inference():
    """批量推理示例"""
    print("\n" + "="*50)
    print("示例2: 批量语音合成")
    print("="*50)
    
    try:
        api = InferenceService()
        
        # 创建参考音频
        ref_audio_path = create_sample_audio("批量推理参考音频")
        
        # 准备多个文本
        texts = [
            "第一句话：欢迎使用语音合成系统。",
            "第二句话：这是批量处理的演示。",
            "第三句话：每句话都会生成独立的音频。",
            "第四句话：感谢您的使用！"
        ]
        
        print(f"📝 准备合成 {len(texts)} 句话")
        
        results = []
        
        try:
            for i, text in enumerate(texts, 1):
                print(f"🔄 处理第 {i} 句: {text}")
                
                config = InferenceConfig(
                    top_k=15,
                    top_p=0.7,
                    temperature=0.5,
                    how_to_cut="不切"
                )
                
                request = InferenceRequest(
                    text=text,
                    text_language="zh",
                    ref_audio_path=ref_audio_path,
                    prompt_text="批量推理参考音频",
                    prompt_language="zh",
                    config=config,
                    return_base64=True  # 使用Base64返回，便于处理
                )
                
                response = await api.inference(request)
                results.append({
                    "text": text,
                    "success": response.success,
                    "response": response
                })
                
                if response.success:
                    print(f"   ✅ 成功 (耗时: {response.processing_time:.2f}s)")
                else:
                    print(f"   ❌ 失败: {response.message}")
            
            # 统计结果
            successful = sum(1 for r in results if r["success"])
            total_time = sum(r["response"].processing_time or 0 for r in results)
            
            print(f"\n📊 批量处理结果:")
            print(f"   成功: {successful}/{len(texts)}")
            print(f"   总耗时: {total_time:.2f}秒")
            print(f"   平均耗时: {total_time/len(texts):.2f}秒/句")
        
        finally:
            if os.path.exists(ref_audio_path):
                os.unlink(ref_audio_path)
    
    except Exception as e:
        print(f"❌ 批量推理示例失败: {e}")


def example_model_management():
    """模型管理示例"""
    print("\n" + "="*50)
    print("示例3: 模型管理")
    print("="*50)
    
    try:
        # 创建临时目录作为模型存储
        temp_dir = tempfile.mkdtemp()
        print(f"📁 临时模型目录: {temp_dir}")
        
        # 初始化模型管理器
        manager = ModelManager(models_dir=temp_dir)
        print("✓ 模型管理器初始化成功")
        
        # 模拟注册模型（使用虚拟路径）
        models_to_register = [
            {
                "name": "中文女声模型",
                "gpt_path": "/models/chinese_female/gpt.ckpt",
                "sovits_path": "/models/chinese_female/sovits.pth",
                "description": "中文女声语音合成模型"
            },
            {
                "name": "英文男声模型", 
                "gpt_path": "/models/english_male/gpt.ckpt",
                "sovits_path": "/models/english_male/sovits.pth",
                "description": "英文男声语音合成模型"
            }
        ]
        
        print("📝 尝试注册模型...")
        for model_info in models_to_register:
            config = ModelConfig(**model_info)
            success = manager.register_model(config)
            print(f"   {model_info['name']}: {'✅ 成功' if success else '❌ 失败 (文件不存在)'}")
        
        # 列出模型
        models = manager.list_models()
        print(f"\n📋 当前注册的模型: {len(models)} 个")
        for model in models:
            print(f"   - {model.name} ({model.version})")
            print(f"     描述: {model.description}")
            print(f"     状态: {'已加载' if model.is_loaded else '未加载'}")
        
        # 获取统计信息
        stats = manager.get_model_statistics()
        print(f"\n📊 模型统计:")
        print(f"   总模型数: {stats['total_models']}")
        print(f"   总大小: {stats['total_size_mb']} MB")
        print(f"   版本分布: {stats['version_distribution']}")
        print(f"   当前模型: {stats['current_model'] or '无'}")
        
        # 清理
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"🗑️ 已清理临时目录")
    
    except Exception as e:
        print(f"❌ 模型管理示例失败: {e}")


def example_audio_processing():
    """音频处理示例"""
    print("\n" + "="*50)
    print("示例4: 音频处理功能")
    print("="*50)
    
    try:
        # 初始化音频处理器
        processor = AudioProcessor()
        print("✓ 音频处理器初始化成功")
        
        # 创建测试音频
        test_audio_path = create_sample_audio("音频处理测试", duration=5.0)
        
        try:
            # 加载音频
            audio, sr = processor.load_audio(test_audio_path)
            print(f"🎵 音频加载成功: 长度={len(audio)}, 采样率={sr}Hz")
            
            # 音频验证
            validation = processor.validate_audio(audio, sr)
            print(f"✅ 音频验证: {'通过' if validation['valid'] else '失败'}")
            if validation['issues']:
                print(f"   问题: {validation['issues']}")
            
            # 提取音频特征
            features = processor.extract_features(audio, sr)
            print(f"📊 音频特征:")
            for key, value in features.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")
            
            # 音频处理操作
            print(f"\n🔧 音频处理操作:")
            
            # 归一化
            normalized = processor.normalize_audio(audio)
            print(f"   归一化: 原始最大值={np.abs(audio).max():.4f}, 归一化后={np.abs(normalized).max():.4f}")
            
            # 去除静音
            trimmed = processor.trim_silence(audio, sr)
            print(f"   去静音: 原始长度={len(audio)}, 处理后={len(trimmed)}")
            
            # 添加填充
            padded = processor.add_padding(audio, sr, start_padding=0.5, end_padding=0.5)
            print(f"   添加填充: 原始长度={len(audio)}, 填充后={len(padded)}")
            
            # 变速处理
            speed_changed = processor.change_speed(audio, speed_factor=1.2)
            print(f"   变速(1.2x): 原始长度={len(audio)}, 变速后={len(speed_changed)}")
            
            # 音量调节
            volume_adjusted = processor.apply_volume(audio, volume_factor=0.5)
            print(f"   音量调节(0.5x): 原始RMS={np.sqrt(np.mean(audio**2)):.4f}, 调节后={np.sqrt(np.mean(volume_adjusted**2)):.4f}")
            
            # Base64转换
            base64_data = processor.audio_to_base64(audio, sr)
            decoded_audio, decoded_sr = processor.base64_to_audio(base64_data)
            print(f"   Base64转换: 编码长度={len(base64_data)}, 解码成功={'✅' if len(decoded_audio) > 0 else '❌'}")
            
            # 语音活动检测
            voice_segments = processor.detect_voice_activity(audio, sr)
            print(f"   语音活动检测: 发现 {len(voice_segments)} 个语音段")
            for i, (start, end) in enumerate(voice_segments[:3]):  # 只显示前3个
                print(f"     段{i+1}: {start:.2f}s - {end:.2f}s")
        
        finally:
            if os.path.exists(test_audio_path):
                os.unlink(test_audio_path)
                print(f"🗑️ 已清理测试音频文件")
    
    except Exception as e:
        print(f"❌ 音频处理示例失败: {e}")


async def example_advanced_inference():
    """高级推理示例"""
    print("\n" + "="*50)
    print("示例5: 高级推理配置")
    print("="*50)
    
    try:
        api = InferenceService()
        ref_audio_path = create_sample_audio("高级推理参考音频")
        
        # 测试不同的文本切分方式
        long_text = "这是一个很长的文本示例。它包含多个句子！每个句子都有不同的语调？我们将测试不同的切分方式；看看哪种效果最好，处理速度如何。"
        
        cut_methods = ["不切", "凑四句一切", "凑50字一切", "按中文句号。切", "按标点符号切"]
        
        try:
            for method in cut_methods:
                print(f"\n🔄 测试切分方式: {method}")
                
                config = InferenceConfig(
                    top_k=25,
                    top_p=0.8,
                    temperature=0.7,
                    how_to_cut=method,
                    speed=1.1,
                    pause_second=0.5
                )
                
                request = InferenceRequest(
                    text=long_text,
                    text_language="zh",
                    ref_audio_path=ref_audio_path,
                    prompt_text="高级推理参考音频",
                    prompt_language="zh",
                    config=config,
                    return_base64=True
                )
                
                response = await api.inference(request)
                
                if response.success:
                    print(f"   ✅ 成功")
                    print(f"   处理时间: {response.processing_time:.2f}s")
                    print(f"   文本片段: {len(response.text_segments) if response.text_segments else 0}")
                    if response.text_segments:
                        print(f"   片段内容: {response.text_segments[:2]}...")  # 显示前2个片段
                else:
                    print(f"   ❌ 失败: {response.message}")
        
        finally:
            if os.path.exists(ref_audio_path):
                os.unlink(ref_audio_path)
    
    except Exception as e:
        print(f"❌ 高级推理示例失败: {e}")


async def run_all_examples():
    """运行所有示例"""
    print("🚀 GPT-SoVITS 推理模块使用示例")
    print("本示例展示了推理模块的各种功能和用法")
    print("注意: 由于使用的是占位实现，某些功能可能无法完全展示真实效果")
    
    # 运行各个示例
    await example_basic_inference()
    await example_batch_inference()
    example_model_management()
    example_audio_processing()
    await example_advanced_inference()
    
    print("\n" + "="*50)
    print("🎉 所有示例运行完成!")
    print("\n💡 使用提示:")
    print("1. 在实际使用中，需要先加载训练好的GPT和SoVITS模型")
    print("2. 参考音频应该是清晰的语音录音，时长建议3-10秒")
    print("3. 参考文本应该与参考音频的内容一致")
    print("4. 根据需要调整推理参数以获得最佳效果")
    print("5. 支持多种音频格式输出和Base64编码传输")


if __name__ == "__main__":
    asyncio.run(run_all_examples())