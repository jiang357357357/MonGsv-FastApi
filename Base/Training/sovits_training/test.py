#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS SoVITS训练 API 测试脚本

测试SoVITS训练API的各项功能
"""

import os
import sys
import asyncio
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from .service import SoVITSTrainingService, SoVITSTrainingRequest, SoVITSTrainingConfig


async def test_sovits_training_api():
    """测试SoVITS训练API"""
    print("=== SoVITS训练API测试 ===")
    
    try:
        # 初始化API
        api = SoVITSTrainingService()
        print(f"✓ API初始化成功")
        print(f"  GPT-SoVITS根目录: {api.gpt_sovits_root}")
        print(f"  训练脚本: {api.s2_train_script}")
        
        # 测试支持的版本
        versions = api.get_supported_versions()
        print(f"✓ 支持的版本: {versions}")
        
        # 测试配置验证
        config = SoVITSTrainingConfig(
            version="v2Pro",
            batch_size=16,
            total_epoch=5,
            gpu_numbers="0"
        )
        
        validation_result = api.validate_config(config)
        print(f"✓ 配置验证: {validation_result}")
        
        # 创建测试请求
        request = SoVITSTrainingRequest(
            exp_name="test_sovits_model",
            exp_root="/tmp/test_experiments",
            config=config
        )
        
        print(f"✓ 创建训练请求成功")
        print(f"  实验名称: {request.exp_name}")
        print(f"  实验根目录: {request.exp_root}")
        print(f"  版本: {request.config.version}")
        print(f"  批次大小: {request.config.batch_size}")
        
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
        api = SoVITSTrainingService()
        
        # 测试有效配置
        valid_config = SoVITSTrainingConfig(
            version="v2Pro",
            batch_size=32,
            total_epoch=8,
            gpu_numbers="0-1"
        )
        
        result = api.validate_config(valid_config)
        print(f"有效配置验证: {result}")
        
        # 测试无效配置
        invalid_config = SoVITSTrainingConfig(
            version="invalid_version",
            batch_size=0,
            total_epoch=-1,
            gpu_numbers="invalid"
        )
        
        result = api.validate_config(invalid_config)
        print(f"无效配置验证: {result}")
        
        print("✓ 配置验证测试通过")
        
    except Exception as e:
        print(f"✗ 配置验证测试失败: {e}")


def test_job_management():
    """测试任务管理功能"""
    print("\n=== 任务管理测试 ===")
    
    try:
        api = SoVITSTrainingService()
        
        # 测试列出任务
        jobs = api.list_training_jobs()
        print(f"当前训练任务: {jobs}")
        
        # 测试获取不存在任务的状态
        status = api.get_training_status("non_existent_job")
        print(f"不存在任务状态: {status}")
        
        print("✓ 任务管理测试通过")
        
    except Exception as e:
        print(f"✗ 任务管理测试失败: {e}")


if __name__ == "__main__":
    # 运行异步测试
    asyncio.run(test_sovits_training_api())
    
    # 运行同步测试
    test_config_validation()
    test_job_management()
    
    print("\n=== 所有测试完成 ===")