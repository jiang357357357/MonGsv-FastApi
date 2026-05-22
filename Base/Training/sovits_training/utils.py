#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS SoVITS训练工具函数

提供训练相关的辅助功能
"""

import os
import re
import json
import yaml
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def parse_training_log(log_file: str) -> Dict:
    """
    解析训练日志文件
    
    Args:
        log_file: 日志文件路径
        
    Returns:
        Dict: 解析结果
    """
    if not os.path.exists(log_file):
        return {"error": "日志文件不存在"}
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析epoch信息
        epoch_pattern = r'Epoch (\d+)/(\d+)'
        epochs = re.findall(epoch_pattern, content)
        
        # 解析损失信息
        loss_pattern = r'Loss: ([\d.]+)'
        losses = [float(loss) for loss in re.findall(loss_pattern, content)]
        
        # 解析学习率信息
        lr_pattern = r'LR: ([\d.e-]+)'
        learning_rates = [float(lr) for lr in re.findall(lr_pattern, content)]
        
        return {
            "epochs": epochs,
            "losses": losses,
            "learning_rates": learning_rates,
            "current_epoch": epochs[-1][0] if epochs else 0,
            "total_epochs": epochs[-1][1] if epochs else 0,
            "latest_loss": losses[-1] if losses else 0.0,
            "min_loss": min(losses) if losses else 0.0
        }
    except Exception as e:
        return {"error": f"解析日志失败: {str(e)}"}


def validate_experiment_data(exp_dir: str) -> Dict:
    """
    验证实验数据完整性
    
    Args:
        exp_dir: 实验目录路径
        
    Returns:
        Dict: 验证结果
    """
    required_files = [
        "2-name2text.txt",
        "3-bert",
        "4-cnhubert", 
        "5-wav32k",
        "6-name2semantic.tsv"
    ]
    
    missing_files = []
    existing_files = []
    
    for file_name in required_files:
        file_path = os.path.join(exp_dir, file_name)
        if os.path.exists(file_path):
            existing_files.append(file_name)
        else:
            missing_files.append(file_name)
    
    # 检查数据量
    data_stats = {}
    
    # 检查文本文件
    text_file = os.path.join(exp_dir, "2-name2text.txt")
    if os.path.exists(text_file):
        with open(text_file, 'r', encoding='utf-8') as f:
            text_lines = len(f.readlines())
        data_stats["text_samples"] = text_lines
    
    # 检查语义文件
    semantic_file = os.path.join(exp_dir, "6-name2semantic.tsv")
    if os.path.exists(semantic_file):
        with open(semantic_file, 'r', encoding='utf-8') as f:
            semantic_lines = len(f.readlines()) - 1  # 减去标题行
        data_stats["semantic_samples"] = semantic_lines
    
    # 检查音频文件
    wav_dir = os.path.join(exp_dir, "5-wav32k")
    if os.path.exists(wav_dir):
        wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
        data_stats["audio_samples"] = len(wav_files)
    
    return {
        "valid": len(missing_files) == 0,
        "existing_files": existing_files,
        "missing_files": missing_files,
        "data_stats": data_stats
    }


def estimate_training_time(config: Dict, data_samples: int) -> Dict:
    """
    估算训练时间
    
    Args:
        config: 训练配置
        data_samples: 数据样本数
        
    Returns:
        Dict: 时间估算结果
    """
    batch_size = config.get("batch_size", 32)
    total_epochs = config.get("total_epoch", 8)
    gpu_count = len(config.get("gpu_numbers", "0").split("-"))
    
    # 每个epoch的步数
    steps_per_epoch = max(1, data_samples // (batch_size * gpu_count))
    total_steps = steps_per_epoch * total_epochs
    
    # 估算每步时间（秒）- 基于经验值
    seconds_per_step = 2.0  # 假设每步2秒
    
    total_seconds = total_steps * seconds_per_step
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    
    return {
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "estimated_hours": hours,
        "estimated_minutes": minutes,
        "estimated_total_seconds": total_seconds
    }


def get_model_size_info(version: str) -> Dict:
    """
    获取模型大小信息
    
    Args:
        version: 模型版本
        
    Returns:
        Dict: 模型信息
    """
    model_info = {
        "v1": {
            "parameters": "~100M",
            "vram_requirement": "6GB+",
            "recommended_batch_size": 16
        },
        "v2": {
            "parameters": "~120M", 
            "vram_requirement": "8GB+",
            "recommended_batch_size": 24
        },
        "v2Pro": {
            "parameters": "~150M",
            "vram_requirement": "10GB+", 
            "recommended_batch_size": 32
        },
        "v2ProPlus": {
            "parameters": "~180M",
            "vram_requirement": "12GB+",
            "recommended_batch_size": 24
        },
        "v4": {
            "parameters": "~200M",
            "vram_requirement": "16GB+",
            "recommended_batch_size": 16
        }
    }
    
    return model_info.get(version, {
        "parameters": "Unknown",
        "vram_requirement": "Unknown", 
        "recommended_batch_size": 16
    })


def optimize_config_for_hardware(config: Dict, gpu_memory_gb: int) -> Dict:
    """
    根据硬件配置优化训练参数
    
    Args:
        config: 原始配置
        gpu_memory_gb: GPU显存大小(GB)
        
    Returns:
        Dict: 优化后的配置
    """
    optimized_config = config.copy()
    
    # 根据显存调整batch_size
    if gpu_memory_gb <= 6:
        optimized_config["batch_size"] = min(config.get("batch_size", 32), 8)
        optimized_config["fp16_run"] = True
    elif gpu_memory_gb <= 8:
        optimized_config["batch_size"] = min(config.get("batch_size", 32), 16)
        optimized_config["fp16_run"] = True
    elif gpu_memory_gb <= 12:
        optimized_config["batch_size"] = min(config.get("batch_size", 32), 24)
    else:
        # 高端显卡可以使用更大的batch_size
        optimized_config["batch_size"] = min(config.get("batch_size", 32), 48)
    
    # 根据显存调整segment_size
    if gpu_memory_gb <= 8:
        optimized_config["segment_size"] = 10240  # 减小音频片段大小
    
    return optimized_config


def create_training_script(config_file: str, output_script: str) -> str:
    """
    创建训练启动脚本
    
    Args:
        config_file: 配置文件路径
        output_script: 输出脚本路径
        
    Returns:
        str: 脚本路径
    """
    script_content = f"""#!/bin/bash
# GPT-SoVITS SoVITS训练启动脚本
# 自动生成于 {os.path.basename(__file__)}

set -e

echo "开始SoVITS训练..."
echo "配置文件: {config_file}"
echo "时间: $(date)"

# 检查CUDA可用性
if command -v nvidia-smi &> /dev/null; then
    echo "GPU信息:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
else
    echo "警告: 未检测到NVIDIA GPU"
fi

# 启动训练
python GPT_SoVITS/s2_train.py --config "{config_file}"

echo "训练完成: $(date)"
"""
    
    with open(output_script, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # 添加执行权限
    os.chmod(output_script, 0o755)
    
    return output_script


def monitor_gpu_usage() -> Dict:
    """
    监控GPU使用情况
    
    Returns:
        Dict: GPU使用信息
    """
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 5:
                    gpu_info.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_used": int(parts[2]),
                        "memory_total": int(parts[3]),
                        "utilization": int(parts[4])
                    })
            
            return {"gpus": gpu_info, "available": True}
        else:
            return {"error": "无法获取GPU信息", "available": False}
    
    except Exception as e:
        return {"error": str(e), "available": False}


def cleanup_old_checkpoints(checkpoint_dir: str, keep_last_n: int = 3) -> List[str]:
    """
    清理旧的检查点文件
    
    Args:
        checkpoint_dir: 检查点目录
        keep_last_n: 保留最新的N个检查点
        
    Returns:
        List[str]: 被删除的文件列表
    """
    if not os.path.exists(checkpoint_dir):
        return []
    
    # 获取所有检查点文件
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pth') or file.endswith('.ckpt'):
            file_path = os.path.join(checkpoint_dir, file)
            mtime = os.path.getmtime(file_path)
            checkpoint_files.append((file_path, mtime))
    
    # 按修改时间排序
    checkpoint_files.sort(key=lambda x: x[1], reverse=True)
    
    # 删除多余的文件
    deleted_files = []
    for file_path, _ in checkpoint_files[keep_last_n:]:
        try:
            os.remove(file_path)
            deleted_files.append(file_path)
        except Exception as e:
            print(f"删除文件失败 {file_path}: {e}")
    
    return deleted_files