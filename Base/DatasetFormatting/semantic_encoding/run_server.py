#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语义编码API启动脚本

提供便捷的启动方式和配置选项
"""

import os
import sys
import argparse
from pathlib import Path

import uvicorn


def find_gpt_sovits_root():
    """查找GPT-SoVITS项目根目录"""
    current_dir = Path(__file__).parent
    
    # 向上查找包含GPT_SoVITS目录的路径
    for parent in current_dir.parents:
        gpt_sovits_dir = parent / "文档" / "GPT-SoVITS-main"
        if gpt_sovits_dir.exists():
            return str(gpt_sovits_dir)
    
    # 尝试相对路径
    possible_paths = [
        "../../../../文档/GPT-SoVITS-main",
        "../../../GPT-SoVITS-main", 
        "../../GPT-SoVITS-main"
    ]
    
    for path in possible_paths:
        abs_path = current_dir / path
        if abs_path.exists() and (abs_path / "GPT_SoVITS").exists():
            return str(abs_path.resolve())
    
    return None


def check_dependencies():
    """检查依赖包"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'torch',
        'numpy',
        'pydantic'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n安装命令:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


def check_model_files(gpt_sovits_root: str):
    """检查模型文件"""
    model_files = {
        "预训练模型": "GPT_SoVITS/pretrained_models/s2G2333k.pth",
        "配置文件": "GPT_SoVITS/configs/s2.json"
    }
    
    missing_files = []
    for name, path in model_files.items():
        full_path = os.path.join(gpt_sovits_root, path)
        if not os.path.exists(full_path):
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        print("⚠️ 以下模型文件不存在:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n请确保GPT-SoVITS模型文件已正确下载")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="启动语义编码API服务")
    parser.add_argument("--host", default="0.0.0.0", help="服务主机地址")
    parser.add_argument("--port", type=int, default=8005, help="服务端口")
    parser.add_argument("--reload", action="store_true", help="开启热重载（开发模式）")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"], help="日志级别")
    parser.add_argument("--gpt-sovits-root", help="GPT-SoVITS项目根目录路径")
    parser.add_argument("--check-only", action="store_true", help="仅检查环境，不启动服务")
    parser.add_argument("--skip-model-check", action="store_true", help="跳过模型文件检查")
    
    args = parser.parse_args()
    
    print("🧠 GPT-SoVITS 语义编码API启动器")
    print("=" * 50)
    
    # 检查依赖
    print("📦 检查依赖包...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ 依赖包检查通过")
    
    # 检查GPT-SoVITS路径
    print("\n📁 检查GPT-SoVITS路径...")
    gpt_sovits_root = args.gpt_sovits_root or find_gpt_sovits_root()
    
    if not gpt_sovits_root:
        print("❌ 未找到GPT-SoVITS项目目录")
        print("请使用 --gpt-sovits-root 参数指定路径")
        sys.exit(1)
    
    print(f"✅ GPT-SoVITS路径: {gpt_sovits_root}")
    
    # 检查模型文件
    if not args.skip_model_check:
        print("\n🤖 检查模型文件...")
        if not check_model_files(gpt_sovits_root):
            print("⚠️ 模型文件检查失败，但仍可启动服务")
            print("使用 --skip-model-check 跳过此检查")
        else:
            print("✅ 模型文件检查通过")
    
    # 设置环境变量
    if gpt_sovits_root:
        os.environ["GPT_SOVITS_ROOT"] = gpt_sovits_root
    
    # 检查CUDA
    print("\n🔧 检查计算环境...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            print("⚠️ CUDA不可用，将使用CPU")
    except ImportError:
        print("❌ PyTorch未安装")
        sys.exit(1)
    
    if args.check_only:
        print("\n✅ 环境检查完成，所有依赖都已满足")
        return
    
    # 启动服务
    print(f"\n🚀 启动API服务...")
    print(f"   主机: {args.host}")
    print(f"   端口: {args.port}")
    print(f"   访问地址: http://{args.host}:{args.port}")
    print(f"   API文档: http://{args.host}:{args.port}/docs")
    print(f"   热重载: {'开启' if args.reload else '关闭'}")
    print(f"   日志级别: {args.log_level}")
    
    print("\n📋 API端点:")
    print(f"   POST /encode - 语义编码")
    print(f"   POST /analyze - 数据集分析")
    print(f"   POST /suggest-config - 配置建议")
    print(f"   POST /validate - 输入验证")
    print(f"   GET /versions - 支持的版本")
    
    try:
        uvicorn.run(
            "semantic_encoding.server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except Exception as e:
        print(f"\n❌ 服务启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()