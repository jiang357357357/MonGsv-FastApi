#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS GPT训练工具函数

提供GPT训练相关的辅助功能
"""

import os
import re
import yaml
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def parse_gpt_training_log(log_file: str) -> Dict:
    """
    解析GPT训练日志文件
    
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
        loss_pattern = r'train_loss: ([\d.]+)'
        losses = [float(loss) for loss in re.findall(loss_pattern, content)]
        
        # 解析准确率信息
        acc_pattern = r'top_3_acc: ([\d.]+)'
        accuracies = [float(acc) for acc in re.findall(acc_pattern, content)]
        
        # 解析学习率信息
        lr_pattern = r'lr: ([\d.e-]+)'
        learning_rates = [float(lr) for lr in re.findall(lr_pattern, content)]
        
        return {
            "epochs": epochs,
            "losses": losses,
            "accuracies": accuracies,
            "learning_rates": learning_rates,
            "current_epoch": epochs[-1][0] if epochs else 0,
            "total_epochs": epochs[-1][1] if epochs else 0,
            "latest_loss": losses[-1] if losses else 0.0,
            "min_loss": min(losses) if losses else 0.0,
            "latest_accuracy": accuracies[-1] if accuracies else 0.0,
            "max_accuracy": max(accuracies) if accuracies else 0.0
        }
    except Exception as e:
        return {"error": f"解析日志失败: {str(e)}"}


def validate_gpt_training_data(exp_dir: str) -> Dict:
    """
    验证GPT训练数据完整性
    
    Args:
        exp_dir: 实验目录路径
        
    Returns:
        Dict: 验证结果
    """
    required_files = [
        "2-name2text.txt",
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
    
    # 检查数据量和质量
    data_stats = {}
    issues = []
    
    # 检查文本文件
    text_file = os.path.join(exp_dir, "2-name2text.txt")
    if os.path.exists(text_file):
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                text_lines = f.readlines()
            
            data_stats["text_samples"] = len(text_lines)
            
            # 检查文本格式
            if len(text_lines) < 10:
                issues.append("文本样本数量过少（建议至少100条）")
            
            # 检查文本内容
            empty_lines = sum(1 for line in text_lines if not line.strip())
            if empty_lines > 0:
                issues.append(f"发现{empty_lines}行空文本")
                
        except Exception as e:
            issues.append(f"读取文本文件失败: {e}")
    
    # 检查语义文件
    semantic_file = os.path.join(exp_dir, "6-name2semantic.tsv")
    if os.path.exists(semantic_file):
        try:
            with open(semantic_file, 'r', encoding='utf-8') as f:
                semantic_lines = f.readlines()
            
            # 减去标题行
            semantic_samples = len(semantic_lines) - 1 if len(semantic_lines) > 0 else 0
            data_stats["semantic_samples"] = semantic_samples
            
            if semantic_samples < 10:
                issues.append("语义样本数量过少（建议至少100条）")
            
            # 检查格式
            if len(semantic_lines) > 1:
                header = semantic_lines[0].strip()
                if header != "item_name\tsemantic_audio":
                    issues.append("语义文件格式错误，标题行应为'item_name\\tsemantic_audio'")
                
                # 检查数据行格式
                for i, line in enumerate(semantic_lines[1:6], 1):  # 检查前5行
                    parts = line.strip().split('\t')
                    if len(parts) != 2:
                        issues.append(f"语义文件第{i+1}行格式错误")
                        break
                        
        except Exception as e:
            issues.append(f"读取语义文件失败: {e}")
    
    # 检查数据一致性
    if "text_samples" in data_stats and "semantic_samples" in data_stats:
        if abs(data_stats["text_samples"] - data_stats["semantic_samples"]) > 5:
            issues.append("文本样本数与语义样本数不匹配")
    
    return {
        "valid": len(missing_files) == 0 and len(issues) == 0,
        "existing_files": existing_files,
        "missing_files": missing_files,
        "data_stats": data_stats,
        "issues": issues
    }


def estimate_gpt_training_time(config: Dict, data_samples: int) -> Dict:
    """
    估算GPT训练时间
    
    Args:
        config: 训练配置
        data_samples: 数据样本数
        
    Returns:
        Dict: 时间估算结果
    """
    batch_size = config.get("batch_size", 8)
    total_epochs = config.get("total_epoch", 15)
    gpu_count = len(config.get("gpu_numbers", "0").split("-"))
    
    # 每个epoch的步数
    steps_per_epoch = max(1, data_samples // (batch_size * gpu_count))
    total_steps = steps_per_epoch * total_epochs
    
    # 估算每步时间（秒）- GPT训练相对较快
    seconds_per_step = 1.5  # 假设每步1.5秒
    
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


def optimize_gpt_config_for_hardware(config: Dict, gpu_memory_gb: int) -> Dict:
    """
    根据硬件配置优化GPT训练参数
    
    Args:
        config: 原始配置
        gpu_memory_gb: GPU显存大小(GB)
        
    Returns:
        Dict: 优化后的配置
    """
    optimized_config = config.copy()
    
    # 根据显存调整batch_size
    if gpu_memory_gb <= 6:
        optimized_config["batch_size"] = min(config.get("batch_size", 8), 4)
        optimized_config["precision"] = "16-mixed"
    elif gpu_memory_gb <= 8:
        optimized_config["batch_size"] = min(config.get("batch_size", 8), 6)
        optimized_config["precision"] = "16-mixed"
    elif gpu_memory_gb <= 12:
        optimized_config["batch_size"] = min(config.get("batch_size", 8), 8)
    else:
        # 高端显卡可以使用更大的batch_size
        optimized_config["batch_size"] = min(config.get("batch_size", 8), 16)
    
    # 根据显存调整模型参数
    if gpu_memory_gb <= 8:
        optimized_config["max_sec"] = 30  # 减小最大音频长度
        optimized_config["num_workers"] = 2  # 减少数据加载进程
    
    return optimized_config


def create_gpt_training_script(config_file: str, output_script: str) -> str:
    """
    创建GPT训练启动脚本
    
    Args:
        config_file: 配置文件路径
        output_script: 输出脚本路径
        
    Returns:
        str: 脚本路径
    """
    script_content = f"""#!/bin/bash
# GPT-SoVITS GPT训练启动脚本
# 自动生成于 {os.path.basename(__file__)}

set -e

echo "开始GPT训练..."
echo "配置文件: {config_file}"
echo "时间: $(date)"

# 检查CUDA可用性
if command -v nvidia-smi &> /dev/null; then
    echo "GPU信息:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
else
    echo "警告: 未检测到NVIDIA GPU"
fi

# 设置环境变量
export MASTER_ADDR="localhost"
export USE_LIBUV="0"

# 启动训练
python GPT_SoVITS/s1_train.py --config_file "{config_file}"

echo "训练完成: $(date)"
"""
    
    with open(output_script, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # 添加执行权限
    os.chmod(output_script, 0o755)
    
    return output_script


def analyze_gpt_model_convergence(log_file: str) -> Dict:
    """
    分析GPT模型收敛情况
    
    Args:
        log_file: 日志文件路径
        
    Returns:
        Dict: 收敛分析结果
    """
    log_data = parse_gpt_training_log(log_file)
    
    if "error" in log_data:
        return log_data
    
    losses = log_data.get("losses", [])
    accuracies = log_data.get("accuracies", [])
    
    if not losses or not accuracies:
        return {"error": "日志中缺少损失或准确率数据"}
    
    # 分析损失趋势
    loss_trend = "unknown"
    if len(losses) >= 10:
        recent_losses = losses[-10:]
        early_losses = losses[:10]
        
        if sum(recent_losses) / len(recent_losses) < sum(early_losses) / len(early_losses):
            loss_trend = "decreasing"
        else:
            loss_trend = "stable_or_increasing"
    
    # 分析准确率趋势
    acc_trend = "unknown"
    if len(accuracies) >= 10:
        recent_accs = accuracies[-10:]
        early_accs = accuracies[:10]
        
        if sum(recent_accs) / len(recent_accs) > sum(early_accs) / len(early_accs):
            acc_trend = "increasing"
        else:
            acc_trend = "stable_or_decreasing"
    
    # 判断收敛状态
    convergence_status = "unknown"
    if loss_trend == "decreasing" and acc_trend == "increasing":
        convergence_status = "converging"
    elif loss_trend == "stable_or_increasing" and acc_trend == "stable_or_decreasing":
        convergence_status = "converged_or_overfitting"
    else:
        convergence_status = "training"
    
    # 提供建议
    suggestions = []
    
    if convergence_status == "converged_or_overfitting":
        suggestions.append("模型可能已收敛或开始过拟合，建议停止训练")
    
    if log_data.get("max_accuracy", 0) < 0.7:
        suggestions.append("准确率较低，建议检查数据质量或调整学习率")
    
    if len(losses) > 5:
        recent_loss_std = sum((x - sum(losses[-5:]) / 5) ** 2 for x in losses[-5:]) / 5
        if recent_loss_std > 0.1:
            suggestions.append("损失波动较大，建议降低学习率")
    
    return {
        "loss_trend": loss_trend,
        "accuracy_trend": acc_trend,
        "convergence_status": convergence_status,
        "suggestions": suggestions,
        "current_loss": log_data.get("latest_loss", 0),
        "current_accuracy": log_data.get("latest_accuracy", 0),
        "best_loss": log_data.get("min_loss", 0),
        "best_accuracy": log_data.get("max_accuracy", 0)
    }


def cleanup_gpt_checkpoints(checkpoint_dir: str, keep_best_n: int = 3) -> List[str]:
    """
    清理GPT检查点文件，保留最佳的N个
    
    Args:
        checkpoint_dir: 检查点目录
        keep_best_n: 保留最佳的N个检查点
        
    Returns:
        List[str]: 被删除的文件列表
    """
    if not os.path.exists(checkpoint_dir):
        return []
    
    # 获取所有检查点文件
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.ckpt') and 'epoch' in file:
            file_path = os.path.join(checkpoint_dir, file)
            
            # 尝试从文件名提取epoch信息
            epoch_match = re.search(r'epoch[_-]?(\d+)', file)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                checkpoint_files.append((file_path, epoch))
    
    # 按epoch排序，保留最新的
    checkpoint_files.sort(key=lambda x: x[1], reverse=True)
    
    # 删除多余的文件
    deleted_files = []
    for file_path, epoch in checkpoint_files[keep_best_n:]:
        try:
            os.remove(file_path)
            deleted_files.append(file_path)
        except Exception as e:
            print(f"删除文件失败 {file_path}: {e}")
    
    return deleted_files