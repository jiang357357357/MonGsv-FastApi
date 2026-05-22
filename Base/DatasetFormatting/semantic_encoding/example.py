#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语义编码模块使用示例

展示各种使用方式和最佳实践
"""

import asyncio
import os
import tempfile
import json
from pathlib import Path

from . import SemanticEncodingService, SemanticEncodingRequest, SemanticEncodingConfig
from .utils import SemanticEncodingUtils


def create_sample_dataset(temp_dir: str, num_files: int = 5) -> tuple:
    """创建示例数据集"""
    import torch
    
    # 创建标注文件内容
    list_content = []
    
    speakers = ["Alice", "Bob", "Carol"]
    languages = ["zh", "en", "ja"]
    
    for i in range(num_files):
        speaker = speakers[i % len(speakers)]
        language = languages[i % len(languages)]
        
        if language == "zh":
            text = f"这是第{i+1}个中文测试样本，用于语义编码测试。"
        elif language == "en":
            text = f"This is the {i+1}th English test sample for semantic encoding."
        else:  # ja
            text = f"これは{i+1}番目の日本語テストサンプルです。"
        
        wav_name = f"sample_{i+1:03d}.wav"
        list_content.append(f"{wav_name}|{speaker}|{language}|{text}")
    
    # 保存标注文件
    list_file = os.path.join(temp_dir, "train_list.txt")
    with open(list_file, "w", encoding="utf8") as f:
        f.write("\n".join(list_content))
    
    # 创建CNHubert特征目录和文件
    cnhubert_dir = os.path.join(temp_dir, "4-cnhubert")
    os.makedirs(cnhubert_dir, exist_ok=True)
    
    for i in range(num_files):
        # 创建假的CNHubert特征 (1, 768, time_steps)
        time_steps = 100 + i * 50  # 不同长度的音频
        fake_feature = torch.randn(1, 768, time_steps)
        
        feature_file = os.path.join(cnhubert_dir, f"sample_{i+1:03d}.pt")
        torch.save(fake_feature, feature_file)
    
    print(f"创建示例数据集: {num_files} 个文件")
    print(f"  标注文件: {list_file}")
    print(f"  CNHubert目录: {cnhubert_dir}")
    
    return list_file, cnhubert_dir


async def example_basic_encoding():
    """基础语义编码示例"""
    print("=== 基础语义编码示例 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建示例数据集
        list_file, cnhubert_dir = create_sample_dataset(temp_dir, 3)
        output_dir = os.path.join(temp_dir, "semantic_output")
        
        # 初始化API
        api = SemanticEncodingService()
        
        # 使用基础配置（注意：需要真实的模型文件）
        config = SemanticEncodingConfig(
            pretrained_s2G="GPT_SoVITS/pretrained_models/s2G2333k.pth",
            s2config_path="GPT_SoVITS/configs/s2.json",
            version=None,  # 自动检测
            device="cpu",  # 使用CPU避免CUDA依赖
            is_half=False,
            n_parts=1,
            output_format="tsv"
        )
        
        request = SemanticEncodingRequest(
            input_text_file=list_file,
            cnhubert_dir=cnhubert_dir,
            experiment_name="basic_example",
            output_dir=output_dir,
            config=config
        )
        
        try:
            result = await api.encode_semantic(request)
            
            print(f"编码结果: {result.success}")
            print(f"处理文件数: {result.processed_count}")
            print(f"失败文件数: {result.failed_count}")
            print(f"处理时间: {result.processing_time:.2f}秒")
            
            if result.success and result.output_file:
                print(f"输出文件: {result.output_file}")
                
                # 查看输出内容
                with open(result.output_file, "r", encoding="utf8") as f:
                    content = f.read()
                    print(f"输出内容预览:\n{content[:200]}...")
                    
        except Exception as e:
            print(f"基础编码示例失败: {e}")
            print("注意: 需要真实的GPT-SoVITS模型文件")


async def example_dataset_analysis():
    """数据集分析示例"""
    print("\n=== 数据集分析示例 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建较大的示例数据集
        list_file, cnhubert_dir = create_sample_dataset(temp_dir, 8)
        
        # 分析数据集
        analysis = SemanticEncodingUtils.analyze_input_data(list_file, cnhubert_dir)
        
        if "error" not in analysis:
            print(f"数据集分析结果:")
            print(f"  总行数: {analysis['total_lines']}")
            print(f"  有效文件: {analysis['valid_lines']}")
            print(f"  无效文件: {analysis['invalid_lines']}")
            print(f"  缺失CNHubert: {analysis['missing_cnhubert']}")
            print(f"  说话人: {', '.join(analysis['speakers'])}")
            print(f"  语言: {', '.join(analysis['languages'])}")
            
            # CNHubert特征统计
            if 'cnhubert_stats' in analysis:
                stats = analysis['cnhubert_stats']
                print(f"  CNHubert统计:")
                print(f"    总大小: {stats['total_size_mb']:.1f}MB")
                print(f"    平均大小: {stats['avg_size_kb']:.1f}KB")
                print(f"    大小范围: {stats['min_size_kb']:.1f}-{stats['max_size_kb']:.1f}KB")
            
            # 文本统计
            if 'text_stats' in analysis:
                stats = analysis['text_stats']
                print(f"  文本统计:")
                print(f"    平均长度: {stats['avg_length']:.1f}字符")
                print(f"    长度范围: {stats['min_length']}-{stats['max_length']}字符")
                
        else:
            print(f"分析失败: {analysis['error']}")


def example_config_optimization():
    """配置优化示例"""
    print("\n=== 配置优化示例 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建不同规模的数据集
        datasets = {
            "小数据集": 3,
            "中数据集": 15,
            "大数据集": 50
        }
        
        for dataset_name, num_files in datasets.items():
            list_file, cnhubert_dir = create_sample_dataset(temp_dir, num_files)
            
            # 获取优化配置
            config = SemanticEncodingUtils.suggest_processing_config(
                list_file, cnhubert_dir,
                target_processing_time=120.0,  # 目标2分钟内完成
                available_memory_gb=8.0        # 8GB内存
            )
            
            print(f"{dataset_name} ({num_files}文件):")
            print(f"  建议设备: {config.device}")
            print(f"  建议并行数: {config.n_parts}")
            print(f"  半精度: {config.is_half}")
            print(f"  输出格式: {config.output_format}")


def example_input_validation():
    """输入验证示例"""
    print("\n=== 输入验证示例 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建有问题的数据集
        list_file, cnhubert_dir = create_sample_dataset(temp_dir, 5)
        
        # 删除一个CNHubert文件模拟缺失
        cnhubert_files = os.listdir(cnhubert_dir)
        if cnhubert_files:
            os.remove(os.path.join(cnhubert_dir, cnhubert_files[0]))
            print(f"删除文件 {cnhubert_files[0]} 模拟缺失情况")
        
        # 验证输入
        validation = SemanticEncodingUtils.validate_input_files(
            list_file, cnhubert_dir,
            check_model_files=False  # 跳过模型文件检查
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
            print(f"  统计:")
            print(f"    总行数: {stats.get('total_lines', 0)}")
            print(f"    有效文件: {stats.get('valid_files', 0)}")
            print(f"    无效文件: {stats.get('invalid_files', 0)}")


def example_time_estimation():
    """处理时间估算示例"""
    print("\n=== 处理时间估算示例 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        list_file, cnhubert_dir = create_sample_dataset(temp_dir, 10)
        
        # 不同配置的时间估算
        configs = {
            "CPU单进程": SemanticEncodingConfig(device="cpu", n_parts=1, is_half=False),
            "CPU多进程": SemanticEncodingConfig(device="cpu", n_parts=4, is_half=False),
            "GPU半精度": SemanticEncodingConfig(device="cuda", n_parts=1, is_half=True),
            "GPU多进程": SemanticEncodingConfig(device="cuda", n_parts=2, is_half=True)
        }
        
        for config_name, config in configs.items():
            time_estimate = SemanticEncodingUtils.estimate_processing_time(
                list_file, cnhubert_dir, config
            )
            
            if "error" not in time_estimate:
                print(f"{config_name}:")
                print(f"  总时间: {time_estimate['estimated_total_time']:.1f}秒")
                print(f"  处理时间: {time_estimate['processing_time']:.1f}秒")
                print(f"  I/O时间: {time_estimate['io_time']:.1f}秒")
                print(f"  并行加速: {time_estimate['parallel_speedup']:.1f}x")
                print(f"  预计完成: {time_estimate['estimated_completion']}")
            else:
                print(f"{config_name}: 估算失败 - {time_estimate['error']}")


async def example_version_comparison():
    """版本对比示例"""
    print("\n=== 版本对比示例 ===")
    
    # 获取支持的版本信息
    versions = SemanticEncodingUtils.get_supported_versions()
    
    print("支持的GPT-SoVITS版本:")
    for version, info in versions.items():
        print(f"  {version}:")
        print(f"    描述: {info['description']}")
        print(f"    模型类: {info['model_class']}")
        print(f"    典型大小: {info['typical_size_mb']}MB")
        print(f"    特性: {', '.join(info['features'])}")


def example_output_validation():
    """输出验证示例"""
    print("\n=== 输出验证示例 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        list_file, cnhubert_dir = create_sample_dataset(temp_dir, 5)
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # 模拟部分输出文件
        # TSV格式
        tsv_file = os.path.join(output_dir, "6-name2semantic.tsv")
        tsv_content = """sample_001	1 2 3 4 5 6 7 8 9 10
sample_002	2 3 4 5 6 7 8 9 10 11
sample_003	3 4 5 6 7 8 9 10 11 12"""
        
        with open(tsv_file, "w", encoding="utf8") as f:
            f.write(tsv_content)
        
        # JSON格式
        json_file = os.path.join(output_dir, "6-name2semantic.json")
        json_data = {
            "sample_001": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "sample_002": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "sample_003": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        }
        
        with open(json_file, "w", encoding="utf8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        # 验证输出完整性
        expected_files = [f"sample_{i+1:03d}" for i in range(5)]  # 期望5个文件
        
        for output_format in ["tsv", "json"]:
            completeness = SemanticEncodingUtils.check_output_completeness(
                output_dir, expected_files, output_format
            )
            
            print(f"{output_format.upper()}格式完整性检查:")
            print(f"  完整: {completeness['complete']}")
            print(f"  缺失文件: {len(completeness['missing_files'])}")
            
            if completeness['missing_files']:
                print(f"  缺失: {completeness['missing_files']}")
            
            # 统计信息
            if 'statistics' in completeness:
                stats = completeness['statistics']
                print(f"  统计: 期望{stats['expected_count']}, 找到{stats['found_count']}")
                print(f"  完成率: {stats['completion_rate']:.1%}")


async def example_batch_processing():
    """批量处理示例"""
    print("\n=== 批量处理示例 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建多个实验的数据
        experiments = ["exp1", "exp2", "exp3"]
        
        for exp_name in experiments:
            exp_dir = os.path.join(temp_dir, exp_name)
            os.makedirs(exp_dir, exist_ok=True)
            
            list_file, cnhubert_dir = create_sample_dataset(exp_dir, 4)
            
            print(f"实验: {exp_name}")
            
            # 分析并建议配置
            config = SemanticEncodingUtils.suggest_processing_config(list_file, cnhubert_dir)
            
            # 估算时间
            time_est = SemanticEncodingUtils.estimate_processing_time(list_file, cnhubert_dir, config)
            
            print(f"  文件数: 4")
            print(f"  建议设备: {config.device}")
            print(f"  建议并行数: {config.n_parts}")
            if "error" not in time_est:
                print(f"  预计时间: {time_est.get('estimated_total_time', 0):.1f}秒")
        
        # 批量分析整个目录
        print(f"\n批量目录分析:")
        batch_result = SemanticEncodingUtils.batch_analyze_directory(temp_dir)
        
        if "error" not in batch_result:
            print(f"  总文件数: {batch_result['total_files']}")
            print(f"  有效文件: {batch_result['valid_files']}")
            print(f"  无效文件: {batch_result['invalid_files']}")
            
            summary = batch_result['summary']
            print(f"  汇总统计:")
            print(f"    总行数: {summary['total_lines']}")
            print(f"    有效行数: {summary['valid_lines']}")
            print(f"    说话人: {len(summary['speakers'])}个")
            print(f"    语言: {len(summary['languages'])}种")


async def run_all_examples():
    """运行所有示例"""
    print("🧠 GPT-SoVITS语义编码模块使用示例")
    print("=" * 60)
    
    try:
        await example_basic_encoding()
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
        print("2. 确保CNHubert特征文件已正确生成")
        print("3. 根据硬件配置调整设备和并行数设置")
        print("4. 大数据集建议先进行分析和时间估算")
        print("5. 定期检查输出文件完整性")
        
    except Exception as e:
        print(f"\n❌ 示例运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_all_examples())