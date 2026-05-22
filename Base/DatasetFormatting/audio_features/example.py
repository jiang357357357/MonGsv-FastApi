#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频特征提取模块使用示例

展示各种使用方式和最佳实践
"""

import asyncio
import os
import tempfile
from pathlib import Path

from . import AudioFeaturesService, AudioFeaturesRequest, AudioFeaturesConfig
from .utils import AudioFeaturesUtils


def create_sample_dataset(temp_dir: str, num_files: int = 5) -> tuple:
    """创建示例数据集"""
    import numpy as np
    import soundfile as sf
    
    wav_dir = os.path.join(temp_dir, "dataset", "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    
    # 创建标注文件内容
    list_content = []
    
    speakers = ["Alice", "Bob", "Carol"]
    languages = ["ZH", "EN", "JA"]
    
    for i in range(num_files):
        # 生成不同特征的音频
        duration = 2 + (i % 3) * 2  # 2-6秒
        sample_rate = 32000
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # 生成复合音频（多个频率）
        freq1 = 220 + i * 55   # 基频
        freq2 = freq1 * 1.5    # 泛音
        audio = 0.4 * np.sin(2 * np.pi * freq1 * t) + 0.2 * np.sin(2 * np.pi * freq2 * t)
        
        # 添加一些噪声
        noise = np.random.normal(0, 0.05, len(audio))
        audio = audio + noise
        
        # 归一化
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        # 保存音频文件
        wav_name = f"sample_{i+1:03d}.wav"
        wav_path = os.path.join(wav_dir, wav_name)
        sf.write(wav_path, audio, sample_rate)
        
        # 添加到标注列表
        speaker = speakers[i % len(speakers)]
        language = languages[i % len(languages)]
        
        if language == "ZH":
            text = f"这是第{i+1}个中文测试样本"
        elif language == "EN":
            text = f"This is the {i+1}th English test sample"
        else:  # JA
            text = f"これは{i+1}番目の日本語テストサンプルです"
        
        list_content.append(f"{wav_name}|{speaker}|{language}|{text}")
    
    # 保存标注文件
    list_file = os.path.join(temp_dir, "dataset", "train_list.txt")
    with open(list_file, "w", encoding="utf8") as f:
        f.write("\n".join(list_content))
    
    print(f"创建示例数据集: {num_files} 个文件")
    print(f"  标注文件: {list_file}")
    print(f"  音频目录: {wav_dir}")
    
    return list_file, wav_dir


async def example_basic_extraction():
    """基础特征提取示例"""
    print("=== 基础特征提取示例 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建示例数据集
        list_file, wav_dir = create_sample_dataset(temp_dir, 3)
        output_dir = os.path.join(temp_dir, "features")
        
        # 初始化API
        api = AudioFeaturesService()
        
        # 使用基础配置（仅处理音频，跳过模型依赖）
        config = AudioFeaturesConfig(
            version="v2",
            device="cpu",
            save_cnhubert=False,  # 跳过CNHubert特征
            save_wav32k=True,     # 保存32kHz音频
            save_speaker=False    # 跳过说话人特征
        )
        
        request = AudioFeaturesRequest(
            input_text_file=list_file,
            input_wav_dir=wav_dir,
            experiment_name="basic_example",
            output_dir=output_dir,
            config=config
        )
        
        try:
            result = await api.extract_features(request)
            
            print(f"特征提取结果: {result.success}")
            print(f"处理文件数: {result.processed_count}")
            print(f"处理时间: {result.processing_time:.2f}秒")
            
            # 检查输出
            if result.output_files.get("wav32k_dir"):
                wav32k_files = os.listdir(result.output_files["wav32k_dir"])
                print(f"生成32kHz音频: {len(wav32k_files)} 个文件")
                
        except Exception as e:
            print(f"基础提取示例失败: {e}")


async def example_dataset_analysis():
    """数据集分析示例"""
    print("\n=== 数据集分析示例 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建较大的示例数据集
        list_file, wav_dir = create_sample_dataset(temp_dir, 8)
        
        # 分析数据集
        analysis = AudioFeaturesUtils.analyze_dataset_from_list(list_file, wav_dir)
        
        if "error" not in analysis:
            print(f"数据集分析结果:")
            print(f"  总文件数: {analysis['total_lines']}")
            print(f"  有效文件: {analysis['valid_files']}")
            print(f"  无效文件: {analysis['invalid_files']}")
            print(f"  说话人: {', '.join(analysis['speakers'])}")
            print(f"  语言: {', '.join(analysis['languages'])}")
            print(f"  总时长: {analysis['total_duration']:.1f}秒")
            print(f"  平均时长: {analysis['avg_duration']:.1f}秒")
            print(f"  总大小: {analysis['total_size_mb']:.1f}MB")
            
            # 音频质量统计
            amp_stats = analysis['max_amplitude_stats']
            print(f"  幅值统计: 最小{amp_stats['min']:.3f}, 最大{amp_stats['max']:.3f}, 平均{amp_stats['mean']:.3f}")
            
            # 处理时间估算
            time_est = analysis['estimated_processing_time']
            print(f"  预计处理时间: CNHubert {time_est['cnhubert']:.1f}s, 说话人 {time_est['speaker']:.1f}s")
            
        else:
            print(f"分析失败: {analysis['error']}")


def example_config_optimization():
    """配置优化示例"""
    print("\n=== 配置优化示例 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建不同规模的数据集
        datasets = {
            "小数据集": 3,
            "中数据集": 10,
            "大数据集": 25
        }
        
        for dataset_name, num_files in datasets.items():
            list_file, wav_dir = create_sample_dataset(temp_dir, num_files)
            
            # 获取优化配置
            config = AudioFeaturesUtils.suggest_processing_config(
                list_file, wav_dir, 
                target_processing_time=30.0,  # 目标30秒内完成
                available_memory_gb=8.0,      # 8GB内存
                version="v2Pro"
            )
            
            print(f"{dataset_name} ({num_files}文件):")
            print(f"  建议并行数: {config.n_parts}")
            print(f"  半精度: {config.is_half}")
            print(f"  说话人特征: {config.save_speaker}")
            print(f"  过滤阈值: {config.max_audio_value}")


def example_input_validation():
    """输入验证示例"""
    print("\n=== 输入验证示例 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建有问题的数据集
        list_file, wav_dir = create_sample_dataset(temp_dir, 5)
        
        # 删除一个音频文件模拟缺失
        wav_files = os.listdir(wav_dir)
        if wav_files:
            os.remove(os.path.join(wav_dir, wav_files[0]))
            print(f"删除文件 {wav_files[0]} 模拟缺失情况")
        
        # 验证输入
        validation = AudioFeaturesUtils.validate_input_files(
            list_file, wav_dir,
            check_audio_existence=True,
            check_audio_quality=True
        )
        
        print(f"验证结果:")
        print(f"  整体有效: {validation['valid']}")
        print(f"  错误数: {len(validation['errors'])}")
        print(f"  警告数: {len(validation['warnings'])}")
        
        if validation['errors']:
            print(f"  错误: {validation['errors']}")
        
        if validation['warnings']:
            print(f"  警告: {validation['warnings']}")
        
        # 统计信息
        if 'statistics' in validation:
            stats = validation['statistics']
            print(f"  有效文件: {stats.get('valid_files', 0)}")
            print(f"  缺失文件: {stats.get('invalid_files', 0)}")


def example_time_estimation():
    """处理时间估算示例"""
    print("\n=== 处理时间估算示例 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        list_file, wav_dir = create_sample_dataset(temp_dir, 6)
        
        # 不同配置的时间估算
        configs = {
            "CPU单进程": AudioFeaturesConfig(device="cpu", n_parts=1, is_half=False),
            "CPU多进程": AudioFeaturesConfig(device="cpu", n_parts=4, is_half=False),
            "GPU半精度": AudioFeaturesConfig(device="cuda", n_parts=1, is_half=True),
            "仅音频处理": AudioFeaturesConfig(save_cnhubert=False, save_speaker=False)
        }
        
        for config_name, config in configs.items():
            time_estimate = AudioFeaturesUtils.estimate_processing_time(list_file, wav_dir, config)
            
            if "error" not in time_estimate:
                print(f"{config_name}:")
                print(f"  总时间: {time_estimate['estimated_total_time']:.1f}秒")
                print(f"  CNHubert: {time_estimate['cnhubert_time']:.1f}秒")
                print(f"  说话人: {time_estimate['speaker_time']:.1f}秒")
                print(f"  I/O时间: {time_estimate['io_time']:.1f}秒")
                print(f"  并行加速: {time_estimate['parallel_speedup']:.1f}x")
            else:
                print(f"{config_name}: 估算失败 - {time_estimate['error']}")


async def example_version_comparison():
    """版本对比示例"""
    print("\n=== 版本对比示例 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        list_file, wav_dir = create_sample_dataset(temp_dir, 4)
        
        # 不同版本配置
        versions = ["v1", "v2", "v2Pro", "v2ProPlus"]
        
        for version in versions:
            config = AudioFeaturesConfig(version=version)
            
            print(f"GPT-SoVITS {version}:")
            print(f"  说话人特征: {config.save_speaker}")
            print(f"  CNHubert特征: {config.save_cnhubert}")
            print(f"  32kHz音频: {config.save_wav32k}")
            
            # 估算处理时间
            time_est = AudioFeaturesUtils.estimate_processing_time(list_file, wav_dir, config)
            if "error" not in time_est:
                print(f"  预计时间: {time_est['estimated_total_time']:.1f}秒")


def example_output_validation():
    """输出验证示例"""
    print("\n=== 输出验证示例 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        list_file, wav_dir = create_sample_dataset(temp_dir, 3)
        output_dir = os.path.join(temp_dir, "output")
        
        # 模拟部分输出文件
        os.makedirs(os.path.join(output_dir, "5-wav32k"), exist_ok=True)
        
        # 创建一些输出文件
        wav_files = os.listdir(wav_dir)
        for i, wav_file in enumerate(wav_files[:2]):  # 只创建前2个
            output_file = os.path.join(output_dir, "5-wav32k", wav_file)
            with open(output_file, "w") as f:
                f.write("dummy content")
        
        # 验证输出完整性
        expected_files = [os.path.splitext(f)[0] for f in wav_files]
        
        completeness = AudioFeaturesUtils.check_output_completeness(
            output_dir, expected_files,
            check_cnhubert=False,
            check_wav32k=True,
            check_speaker=False
        )
        
        print(f"输出完整性检查:")
        print(f"  完整: {completeness['complete']}")
        print(f"  缺失文件: {len(completeness['missing_files'])}")
        
        if completeness['missing_files']:
            print(f"  缺失: {completeness['missing_files']}")
        
        # 统计信息
        for check_type, stats in completeness['statistics'].items():
            print(f"  {check_type}: 期望{stats['expected']}, 找到{stats['found']}, 缺失{stats['missing']}")


async def example_batch_processing():
    """批量处理示例"""
    print("\n=== 批量处理示例 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建多个实验的数据
        experiments = ["exp1", "exp2", "exp3"]
        
        for exp_name in experiments:
            exp_dir = os.path.join(temp_dir, exp_name)
            os.makedirs(exp_dir, exist_ok=True)
            
            list_file, wav_dir = create_sample_dataset(exp_dir, 4)
            output_dir = os.path.join(exp_dir, "features")
            
            print(f"处理实验: {exp_name}")
            
            # 分析并建议配置
            config = AudioFeaturesUtils.suggest_processing_config(list_file, wav_dir)
            
            # 估算时间
            time_est = AudioFeaturesUtils.estimate_processing_time(list_file, wav_dir, config)
            
            print(f"  文件数: {len(os.listdir(wav_dir))}")
            print(f"  建议并行数: {config.n_parts}")
            print(f"  预计时间: {time_est.get('estimated_total_time', 0):.1f}秒")


async def run_all_examples():
    """运行所有示例"""
    print("🎵 音频特征提取模块使用示例")
    print("=" * 60)
    
    try:
        await example_basic_extraction()
        await example_dataset_analysis()
        example_config_optimization()
        example_input_validation()
        example_time_estimation()
        await example_version_comparison()
        example_output_validation()
        await example_batch_processing()
        
        print("\n✅ 所有示例运行完成！")
        print("\n💡 使用提示:")
        print("1. 在实际使用中，请确保GPT-SoVITS模型文件存在")
        print("2. 根据硬件配置调整并行数和精度设置")
        print("3. 大数据集建议先进行分析和验证")
        print("4. 定期检查输出文件完整性")
        
    except Exception as e:
        print(f"\n❌ 示例运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_all_examples())