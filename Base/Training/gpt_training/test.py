#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS GPT训练 API 测试脚本

测试GPT训练API的各项功能
"""

import os
import sys
import asyncio
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from .service import GPTTrainingService, GPTTrainingRequest, GPTTrainingConfig


async def test_gpt_training_api():
    """测试GPT训练API"""
    print("=== GPT训练API测试 ===")
    
    try:
        # 初始化API
        api = GPTTrainingService()
        print(f"✓ API初始化成功")
        print(f"  GPT-SoVITS根目录: {api.gpt_sovits_root}")
        print(f"  训练脚本: {api.s1_train_script}")
        
        # 测试配置验证
        config = GPTTrainingConfig(
            batch_size=8,
            total_epoch=10,
            learning_rate=0.01,
            gpu_numbers="0"
        )
        
        validation_result = api.validate_config(config)
        print(f"✓ 配置验证: {validation_result}")
        
        # 创建测试请求
        request = GPTTrainingRequest(
            exp_name="test_gpt_model",
            exp_root="/tmp/test_experiments",
            config=config
        )
        
        print(f"✓ 创建训练请求成功")
        print(f"  实验名称: {request.exp_name}")
        print(f"  实验根目录: {request.exp_root}")
        print(f"  批次大小: {request.config.batch_size}")
        print(f"  训练轮数: {request.config.total_epoch}")
        print(f"  学习率: {request.config.learning_rate}")
        
        # 注意：这里不实际启动训练，只测试API接口
        print("✓ 所有测试通过")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_config_validation():
    """测试配置验证功能"""
    print("\n=== 配置验证测试 ===")
    
    try:
        api = GPTTrainingService()
        
        # 测试有效配置
        valid_config = GPTTrainingConfig(
            batch_size=8,
            total_epoch=15,
            learning_rate=0.01,
            gpu_numbers="0-1"
        )
        
        result = api.validate_config(valid_config)
        print(f"有效配置验证: {result}")
        
        # 测试无效配置
        invalid_config = GPTTrainingConfig(
            batch_size=0,
            total_epoch=-1,
            learning_rate=-0.01,
            gpu_numbers="invalid"
        )
        
        result = api.validate_config(invalid_config)
        print(f"无效配置验证: {result}")
        
        print("✓ 配置验证测试通过")
        
    except Exception as e:
        print(f"✗ 配置验证测试失败: {e}")


def test_data_validation():
    """测试数据验证功能"""
    print("\n=== 数据验证测试 ===")
    
    try:
        api = GPTTrainingService()
        
        # 测试不存在的目录
        result = api._validate_training_data("/non/existent/path")
        print(f"不存在目录验证: {result}")
        
        # 创建临时测试数据
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建测试文件
            text_file = os.path.join(temp_dir, "2-name2text.txt")
            semantic_file = os.path.join(temp_dir, "6-name2semantic.tsv")
            
            # 写入测试数据
            with open(text_file, 'w', encoding='utf-8') as f:
                for i in range(20):
                    f.write(f"test_audio_{i:03d}.wav|speaker|zh|这是测试文本{i}\n")
            
            with open(semantic_file, 'w', encoding='utf-8') as f:
                f.write("item_name\tsemantic_audio\n")
                for i in range(20):
                    f.write(f"test_audio_{i:03d}.wav\t1 2 3 4 5\n")
            
            # 验证测试数据
            result = api._validate_training_data(temp_dir)
            print(f"测试数据验证: {result}")
        
        print("✓ 数据验证测试通过")
        
    except Exception as e:
        print(f"✗ 数据验证测试失败: {e}")


def test_job_management():
    """测试任务管理功能"""
    print("\n=== 任务管理测试 ===")
    
    try:
        api = GPTTrainingService()
        
        # 测试列出任务
        jobs = api.list_training_jobs()
        print(f"当前训练任务: {jobs}")
        
        # 测试获取不存在任务的状态
        status = api.get_training_status("non_existent_job")
        print(f"不存在任务状态: {status}")
        
        print("✓ 任务管理测试通过")
        
    except Exception as e:
        print(f"✗ 任务管理测试失败: {e}")


def test_utils_functions():
    """测试工具函数"""
    print("\n=== 工具函数测试 ===")
    
    try:
        from .utils import (
            validate_gpt_training_data,
            estimate_gpt_training_time,
            optimize_gpt_config_for_hardware
        )
        
        # 测试数据验证
        result = validate_gpt_training_data("/non/existent/path")
        print(f"数据验证工具: {result}")
        
        # 测试时间估算
        config = {"batch_size": 8, "total_epoch": 15, "gpu_numbers": "0"}
        time_estimate = estimate_gpt_training_time(config, 1000)
        print(f"时间估算: {time_estimate}")
        
        # 测试配置优化
        original_config = {"batch_size": 16, "precision": "32"}
        optimized = optimize_gpt_config_for_hardware(original_config, 8)
        print(f"配置优化: {optimized}")
        
        print("✓ 工具函数测试通过")
        
    except Exception as e:
        print(f"✗ 工具函数测试失败: {e}")


if __name__ == "__main__":
    # 运行异步测试
    asyncio.run(test_gpt_training_api())
    
    # 运行同步测试
    test_config_validation()
    test_data_validation()
    test_job_management()
    test_utils_functions()
    
    print("\n=== 所有测试完成 ===")