#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS SoVITS训练 API 使用示例

演示如何使用SoVITS训练API进行模型训练
"""

import asyncio
import time
from pathlib import Path

from .service import SoVITSTrainingService, SoVITSTrainingRequest, SoVITSTrainingConfig


async def basic_training_example():
    """基础训练示例"""
    print("=== SoVITS训练基础示例 ===")
    
    # 初始化API
    api = SoVITSTrainingService()
    
    # 创建训练配置
    config = SoVITSTrainingConfig(
        version="v2Pro",
        batch_size=16,  # 根据显存调整
        total_epoch=5,  # 测试用较少epoch
        save_every_epoch=2,
        gpu_numbers="0",
        if_save_latest=True,
        if_save_every_weights=True
    )
    
    # 创建训练请求
    request = SoVITSTrainingRequest(
        exp_name="example_voice_model",
        exp_root="/path/to/your/experiments",  # 请修改为实际路径
        config=config
    )
    
    print(f"实验名称: {request.exp_name}")
    print(f"实验目录: {request.exp_root}")
    print(f"模型版本: {config.version}")
    print(f"批次大小: {config.batch_size}")
    print(f"训练轮数: {config.total_epoch}")
    
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
                    
                    if status.status in ["completed", "failed", "stopped"]:
                        break
                
                await asyncio.sleep(10)  # 每10秒检查一次
            
            print(f"训练结束，最终状态: {status.status}")
            
        else:
            print(f"✗ 训练启动失败: {response.message}")
            
    except Exception as e:
        print(f"✗ 训练过程出错: {e}")


def multi_gpu_training_example():
    """多GPU训练示例"""
    print("\n=== 多GPU训练示例 ===")
    
    # 多GPU配置
    config = SoVITSTrainingConfig(
        version="v2Pro",
        batch_size=24,  # 多GPU可以用更大的batch_size
        total_epoch=10,
        gpu_numbers="0-1-2-3",  # 使用4张GPU
        fp16_run=True,  # 启用半精度训练节省显存
        if_grad_ckpt=True  # 启用梯度检查点
    )
    
    print(f"GPU配置: {config.gpu_numbers}")
    print(f"批次大小: {config.batch_size}")
    print(f"半精度训练: {config.fp16_run}")
    print(f"梯度检查点: {config.if_grad_ckpt}")


def advanced_config_example():
    """高级配置示例"""
    print("\n=== 高级配置示例 ===")
    
    # 高级配置
    config = SoVITSTrainingConfig(
        version="v2ProPlus",
        batch_size=32,
        total_epoch=20,
        save_every_epoch=5,
        
        # 学习率配置
        text_low_lr_rate=0.3,  # 降低文本模块学习率
        learning_rate=0.00008,  # 降低基础学习率
        lr_decay=0.9998,  # 调整衰减率
        
        # 损失权重调整
        c_mel=50.0,  # 增加Mel损失权重提高音质
        c_kl=1.2,    # 增加KL损失权重
        
        # 数据配置
        segment_size=16384,  # 减小片段大小节省显存
        sampling_rate=32000,
        
        # 预训练模型
        pretrained_s2G="/path/to/pretrained/s2G.pth",
        pretrained_s2D="/path/to/pretrained/s2D.pth",
        
        gpu_numbers="0-1"
    )
    
    print(f"模型版本: {config.version}")
    print(f"文本学习率权重: {config.text_low_lr_rate}")
    print(f"基础学习率: {config.learning_rate}")
    print(f"Mel损失权重: {config.c_mel}")
    print(f"KL损失权重: {config.c_kl}")
    print(f"音频片段大小: {config.segment_size}")


async def training_monitoring_example():
    """训练监控示例"""
    print("\n=== 训练监控示例 ===")
    
    api = SoVITSTrainingService()
    
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
            print(f"开始时间: {status.start_time}")
            print(f"日志文件: {status.log_file}")
        
        # 演示停止训练
        # stop_success = api.stop_training(job_id)
        # print(f"停止训练: {stop_success}")


def config_optimization_example():
    """配置优化示例"""
    print("\n=== 配置优化示例 ===")
    
    from .utils import optimize_config_for_hardware, get_model_size_info
    
    # 获取模型信息
    model_info = get_model_size_info("v2Pro")
    print(f"v2Pro模型信息: {model_info}")
    
    # 原始配置
    original_config = {
        "batch_size": 48,
        "segment_size": 20480,
        "fp16_run": False
    }
    
    # 针对8GB显存优化
    optimized_config = optimize_config_for_hardware(original_config, gpu_memory_gb=8)
    print(f"8GB显存优化配置: {optimized_config}")
    
    # 针对6GB显存优化
    optimized_config_6gb = optimize_config_for_hardware(original_config, gpu_memory_gb=6)
    print(f"6GB显存优化配置: {optimized_config_6gb}")


if __name__ == "__main__":
    print("GPT-SoVITS SoVITS训练API使用示例")
    print("=" * 50)
    
    # 运行异步示例
    # asyncio.run(basic_training_example())
    
    # 运行同步示例
    multi_gpu_training_example()
    advanced_config_example()
    config_optimization_example()
    
    # asyncio.run(training_monitoring_example())
    
    print("\n注意：实际训练需要准备好训练数据和预训练模型")
    print("请根据实际情况修改路径和配置参数")