#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频切分模块使用示例

展示各种使用方式和最佳实践
"""

import asyncio
import os
import tempfile
from pathlib import Path

from . import AudioSliceService, SliceRequest, SliceConfig
from .utils import AudioSliceUtils, create_test_audio_with_silence, optimize_slice_config_for_training


async def example_basic_usage():
    """基础使用示例"""
    print("=== 基础使用示例 ===")
    
    # 创建测试音频
    with tempfile.TemporaryDirectory() as temp_dir:
        input_audio = os.path.join(temp_dir, "test.wav")
        output_dir = os.path.join(temp_dir, "output")
        
        # 生成测试音频（10秒，包含静音段）
        create_test_audio_with_silence(
            duration=10.0,
            silence_segments=[(2.0, 3.0), (6.0, 7.0)],
            output_path=input_audio
        )
        
        # 初始化API
        api = AudioSliceService()
        
        # 使用默认配置
        request = SliceRequest(
            input_path=input_audio,
            output_dir=output_dir
        )
        
        # 执行切分
        result = await api.slice_audio(request)
        
        print(f"切分结果: {result.success}")
        print(f"消息: {result.message}")
        print(f"输出文件数: {len(result.output_files)}")
        
        for i, file_path in enumerate(result.output_files):
            print(f"  {i+1}. {os.path.basename(file_path)}")


async def example_custom_config():
    """自定义配置示例"""
    print("\n=== 自定义配置示例 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        input_audio = os.path.join(temp_dir, "test.wav")
        output_dir = os.path.join(temp_dir, "output")
        
        create_test_audio_with_silence(duration=8.0, output_path=input_audio)
        
        # 自定义配置
        config = SliceConfig(
            threshold=-40.0,    # 更敏感的静音检测
            min_length=2000,    # 更短的最小片段
            min_interval=200,   # 更短的切割间隔
            max_sil_kept=300,   # 保留更少的静音
        )
        
        api = AudioSliceService()
        request = SliceRequest(
            input_path=input_audio,
            output_dir=output_dir,
            config=config
        )
        
        result = await api.slice_audio(request)
        
        print(f"自定义配置切分结果: {result.success}")
        print(f"输出文件数: {len(result.output_files)}")


async def example_smart_config():
    """智能配置示例"""
    print("\n=== 智能配置示例 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        input_audio = os.path.join(temp_dir, "test.wav")
        output_dir = os.path.join(temp_dir, "output")
        
        create_test_audio_with_silence(duration=12.0, output_path=input_audio)
        
        # 分析音频文件
        file_info = AudioSliceUtils.analyze_audio_file(input_audio)
        print(f"音频信息: 时长{file_info['duration']:.1f}秒, 大小{file_info['file_size_mb']:.1f}MB")
        
        # 获取智能建议配置
        smart_config = AudioSliceUtils.suggest_slice_config(
            input_audio, 
            target_segment_duration=4.0
        )
        
        print(f"建议配置: 阈值{smart_config.threshold}, 最小长度{smart_config.min_length}ms")
        
        # 使用智能配置
        api = AudioSliceService()
        request = SliceRequest(
            input_path=input_audio,
            output_dir=output_dir,
            config=smart_config
        )
        
        result = await api.slice_audio(request)
        print(f"智能配置切分结果: {result.success}")
        print(f"输出文件数: {len(result.output_files)}")


async def example_training_optimized():
    """训练优化配置示例"""
    print("\n=== 训练优化配置示例 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        input_audio = os.path.join(temp_dir, "test.wav")
        output_dir = os.path.join(temp_dir, "output")
        
        create_test_audio_with_silence(duration=15.0, output_path=input_audio)
        
        # 为不同训练类型优化配置
        training_configs = {
            "标准训练": optimize_slice_config_for_training(5.0, "standard"),
            "短片段训练": optimize_slice_config_for_training(3.0, "short"),
            "长片段训练": optimize_slice_config_for_training(8.0, "long")
        }
        
        api = AudioSliceService()
        
        for training_type, config in training_configs.items():
            type_output_dir = os.path.join(output_dir, training_type.replace(" ", "_"))
            
            request = SliceRequest(
                input_path=input_audio,
                output_dir=type_output_dir,
                config=config
            )
            
            result = await api.slice_audio(request)
            print(f"{training_type}: {len(result.output_files)}个片段")


def example_batch_analysis():
    """批量分析示例"""
    print("\n=== 批量分析示例 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建多个测试音频文件
        test_files = []
        for i in range(3):
            file_path = os.path.join(temp_dir, f"test_{i+1}.wav")
            create_test_audio_with_silence(
                duration=5 + i * 2,  # 不同时长
                output_path=file_path
            )
            test_files.append(file_path)
        
        # 批量分析
        batch_info = AudioSliceUtils.batch_analyze_directory(temp_dir)
        
        print(f"总文件数: {batch_info['total_files']}")
        print(f"总时长: {batch_info['total_duration']:.1f}秒")
        print(f"平均时长: {batch_info['summary']['average_duration']:.1f}秒")
        
        # 获取批量处理建议配置
        batch_config = batch_info['summary']['suggested_config']
        print(f"建议并行数: {batch_config.n_parts}")
        print(f"建议阈值: {batch_config.threshold}")


async def example_quality_validation():
    """质量验证示例"""
    print("\n=== 质量验证示例 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        input_audio = os.path.join(temp_dir, "test.wav")
        output_dir = os.path.join(temp_dir, "output")
        
        create_test_audio_with_silence(duration=6.0, output_path=input_audio)
        
        # 执行切分
        api = AudioSliceService()
        request = SliceRequest(
            input_path=input_audio,
            output_dir=output_dir
        )
        
        result = await api.slice_audio(request)
        
        if result.success:
            # 验证切分结果
            validation = AudioSliceUtils.validate_slice_result(output_dir, min_files=2)
            
            print(f"验证结果: {'通过' if validation['valid'] else '失败'}")
            print(f"文件数量: {validation['file_count']}")
            print(f"总大小: {validation['total_size_mb']:.1f}MB")
            
            if not validation['valid']:
                print(f"验证失败原因: {validation.get('error', '未知')}")


def example_time_estimation():
    """处理时间估算示例"""
    print("\n=== 处理时间估算示例 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        input_audio = os.path.join(temp_dir, "test.wav")
        
        create_test_audio_with_silence(duration=20.0, output_path=input_audio)
        
        # 分析文件
        file_info = AudioSliceUtils.analyze_audio_file(input_audio)
        
        # 不同配置的时间估算
        configs = {
            "快速处理": SliceConfig(n_parts=4, hop_size=20),
            "标准处理": SliceConfig(n_parts=2, hop_size=10),
            "精细处理": SliceConfig(n_parts=1, hop_size=5, threshold=-45.0)
        }
        
        for config_name, config in configs.items():
            estimated_time = AudioSliceUtils.estimate_processing_time(file_info, config)
            print(f"{config_name}: 预计{estimated_time:.1f}秒")


async def run_all_examples():
    """运行所有示例"""
    print("🎵 音频切分模块使用示例")
    print("=" * 50)
    
    try:
        await example_basic_usage()
        await example_custom_config()
        await example_smart_config()
        await example_training_optimized()
        example_batch_analysis()
        await example_quality_validation()
        example_time_estimation()
        
        print("\n✅ 所有示例运行完成！")
        
    except Exception as e:
        print(f"\n❌ 示例运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_all_examples())