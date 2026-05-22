#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS GPT训练 API 使用示例

演示如何使用GPT训练API进行模型训练
"""

import asyncio
import time
from pathlib import Path

from .service import GPTTrainingService, GPTTrainingRequest, GPTTrainingConfig


async def basic_gpt_training_example():
    """基础GPT训练示例"""
    print("=== GPT训练基础示例 ===")
    
    # 初始化API
    api = GPTTrainingService()
    
    # 创建训练配置
    config = GPTTrainingConfig(
        batch_size=8,
        total_epoch=10,  # 测试用较少epoch
        save_every_epoch=3,
        learning_rate=0.01,
        gpu_numbers="0",
        if_save_latest=True,
        if_save_every_weights=True
    )
    
    # 创建训练请求
    request = GPTTrainingRequest(
        exp_name="example_gpt_model",
        exp_root="/path/to/your/experiments",  # 请修改为实际路径
        config=config
    )
    
    print(f"实验名称: {request.exp_name}")
    print(f"实验目录: {request.exp_root}")
    print(f"批次大小: {config.batch_size}")
    print(f"训练轮数: {config.total_epoch}")
    print(f"学习率: {config.learning_rate}")
    
    # 验证配置
    validation = api.validate_config(config)
    if not validation["valid"]:
        print(f"配置验证失败: {validation['issues']}")
        return
    
    print("✓ 配置验证通过")
    
    # 启动训练（注意：这需要实际的训练数据）
    try:
        response = await api.start_training(request)
        
        if response.success:
            print(f"✓ 训练启动成功")
            print(f"  任务ID: {response.job_id}")
            print(f"  配置文件: {response.config_file}")
            print(f"  日志目录: {response.log_dir}")
            print(f"  模型目录: {response.model_dir}")
            
            # 监控训练状态
            job_id = response.job_id
            while True:
                status = api.get_training_status(job_id)
                if status:
                    print(f"训练状态: {status.status}")
                    print(f"当前轮次: {status.current_epoch}/{status.total_epochs}")
                    print(f"当前损失: {status.current_loss}")
                    print(f"Top-3准确率: {status.top3_accuracy}")
                    
                    if status.status in ["completed", "failed", "stopped"]:
                        break
                
                await asyncio.sleep(10)  # 每10秒检查一次
            
            print(f"训练结束，最终状态: {status.status}")
            
        else:
            print(f"✗ 训练启动失败: {response.message}")
            
    except Exception as e:
        print(f"✗ 训练过程出错: {e}")


def multi_gpu_gpt_training_example():
    """多GPU GPT训练示例"""
    print("\n=== 多GPU GPT训练示例 ===")
    
    # 多GPU配置
    config = GPTTrainingConfig(
        batch_size=12,  # 多GPU可以用更大的batch_size
        total_epoch=20,
        gpu_numbers="0-1-2-3",  # 使用4张GPU
        precision="16-mixed",  # 使用混合精度训练
        learning_rate=0.008,  # 多GPU时可以适当提高学习率
        warmup_steps=3000,
        decay_steps=50000
    )
    
    print(f"GPU配置: {config.gpu_numbers}")
    print(f"批次大小: {config.batch_size}")
    print(f"训练精度: {config.precision}")
    print(f"学习率: {config.learning_rate}")
    print(f"预热步数: {config.warmup_steps}")


def advanced_gpt_config_example():
    """高级GPT配置示例"""
    print("\n=== 高级GPT配置示例 ===")
    
    # 高级配置
    config = GPTTrainingConfig(
        batch_size=16,
        total_epoch=25,
        save_every_epoch=5,
        
        # 学习率调度
        learning_rate=0.015,
        warmup_steps=4000,
        decay_steps=60000,
        
        # 模型架构
        vocab_size=1025,
        phoneme_vocab_size=512,
        embedding_dim=768,  # 增大嵌入维度
        hidden_dim=768,
        n_layer=32,  # 增加层数
        n_head=24,   # 增加注意力头数
        
        # 训练选项
        if_dpo=True,  # 启用DPO训练
        precision="16-mixed",
        gradient_clip=0.8,
        
        # 数据配置
        max_sec=60,  # 增加最大音频长度
        num_workers=6,
        
        # 预训练模型
        pretrained_s1="/path/to/pretrained/gpt_model.ckpt",
        
        gpu_numbers="0-1"
    )
    
    print(f"嵌入维度: {config.embedding_dim}")
    print(f"隐藏维度: {config.hidden_dim}")
    print(f"Transformer层数: {config.n_layer}")
    print(f"注意力头数: {config.n_head}")
    print(f"DPO训练: {config.if_dpo}")
    print(f"最大音频长度: {config.max_sec}秒")


async def gpt_training_monitoring_example():
    """GPT训练监控示例"""
    print("\n=== GPT训练监控示例 ===")
    
    api = GPTTrainingService()
    
    # 假设有一个正在运行的训练任务
    jobs = api.list_training_jobs()
    print(f"当前训练任务: {jobs}")
    
    if jobs:
        job_id = jobs[0]
        print(f"监控任务: {job_id}")
        
        # 获取详细状态
        status = api.get_training_status(job_id)
        if status:
            print(f"状态: {status.status}")
            print(f"进度: {status.current_epoch}/{status.total_epochs}")
            print(f"当前损失: {status.current_loss}")
            print(f"最佳损失: {status.best_loss}")
            print(f"Top-3准确率: {status.top3_accuracy}")
            print(f"学习率: {status.learning_rate}")
            print(f"开始时间: {status.start_time}")
            print(f"日志文件: {status.log_file}")


def gpt_config_optimization_example():
    """GPT配置优化示例"""
    print("\n=== GPT配置优化示例 ===")
    
    from .utils import optimize_gpt_config_for_hardware, estimate_gpt_training_time
    
    # 原始配置
    original_config = {
        "batch_size": 16,
        "precision": "32",
        "max_sec": 60,
        "num_workers": 8
    }
    
    # 针对8GB显存优化
    optimized_config = optimize_gpt_config_for_hardware(original_config, gpu_memory_gb=8)
    print(f"8GB显存优化配置: {optimized_config}")
    
    # 针对6GB显存优化
    optimized_config_6gb = optimize_gpt_config_for_hardware(original_config, gpu_memory_gb=6)
    print(f"6GB显存优化配置: {optimized_config_6gb}")
    
    # 估算训练时间
    config = {"batch_size": 8, "total_epoch": 15, "gpu_numbers": "0"}
    time_estimate = estimate_gpt_training_time(config, 2000)
    print(f"训练时间估算: {time_estimate}")


def data_validation_example():
    """数据验证示例"""
    print("\n=== 数据验证示例 ===")
    
    from .utils import validate_gpt_training_data
    
    # 验证训练数据
    exp_dir = "/path/to/your/experiment"  # 请修改为实际路径
    validation_result = validate_gpt_training_data(exp_dir)
    
    print(f"数据验证结果: {validation_result}")
    
    if validation_result["valid"]:
        print("✓ 训练数据验证通过")
        print(f"  文本样本数: {validation_result['data_stats'].get('text_samples', 0)}")
        print(f"  语义样本数: {validation_result['data_stats'].get('semantic_samples', 0)}")
    else:
        print("✗ 训练数据验证失败")
        print(f"  缺少文件: {validation_result['missing_files']}")
        print(f"  问题: {validation_result.get('issues', [])}")


if __name__ == "__main__":
    print("GPT-SoVITS GPT训练API使用示例")
    print("=" * 50)
    
    # 运行异步示例
    # asyncio.run(basic_gpt_training_example())
    
    # 运行同步示例
    multi_gpu_gpt_training_example()
    advanced_gpt_config_example()
    gpt_config_optimization_example()
    data_validation_example()
    
    # asyncio.run(gpt_training_monitoring_example())
    
    print("\n注意：实际训练需要准备好训练数据")
    print("请根据实际情况修改路径和配置参数")