#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 推理服务启动脚本

启动推理API服务器
"""

import os
import sys
import argparse
import uvicorn
from pathlib import Path

from Code.runtime_env import ensure_audio_runtime

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="GPT-SoVITS 推理 API 服务器")
    
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0",
        help="服务器主机地址 (默认: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8009,
        help="服务器端口 (默认: 8009)"
    )
    
    parser.add_argument(
        "--workers", 
        type=int, 
        default=1,
        help="工作进程数 (默认: 1)"
    )
    
    parser.add_argument(
        "--reload", 
        action="store_true",
        help="启用自动重载 (开发模式)"
    )
    
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="日志级别 (默认: info)"
    )
    
    parser.add_argument(
        "--access-log", 
        action="store_true",
        help="启用访问日志"
    )
    
    parser.add_argument(
        "--gpt-sovits-root", 
        type=str,
        help="GPT-SoVITS项目根目录路径"
    )
    
    parser.add_argument(
        "--models-dir", 
        type=str,
        help="模型存储目录路径"
    )
    
    return parser.parse_args()


def setup_environment(args):
    """设置环境变量"""
    if args.gpt_sovits_root:
        os.environ["GPT_SOVITS_ROOT"] = args.gpt_sovits_root
        print(f"设置GPT-SoVITS根目录: {args.gpt_sovits_root}")
    
    if args.models_dir:
        os.environ["MODELS_DIR"] = args.models_dir
        print(f"设置模型目录: {args.models_dir}")


def check_dependencies():
    """检查依赖项"""
    required_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "numpy",
        "torch",
        "librosa",
        "soundfile"
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
        print("\n请安装缺少的依赖包:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ 所有依赖包检查通过")
    return True


def test_api_initialization():
    """测试API初始化"""
    try:
        from Code.FastApi.Base.Inference.server import inference_api, model_manager
        
        if inference_api:
            print("✅ 推理API初始化成功")
            model_info = inference_api.get_model_info()
            print(f"   设备: {model_info.get('device', 'unknown')}")
            print(f"   模型已加载: {model_info.get('models_loaded', False)}")
        else:
            print("⚠️ 推理API初始化失败，但服务仍可启动")
        
        if model_manager:
            print("✅ 模型管理器初始化成功")
            stats = model_manager.get_model_statistics()
            print(f"   注册模型数: {stats.get('total_models', 0)}")
        else:
            print("⚠️ 模型管理器初始化失败")
        
        return True
        
    except Exception as e:
        print(f"⚠️ API初始化测试失败: {e}")
        print("服务仍将启动，但某些功能可能不可用")
        return True  # 仍然允许启动服务


def print_startup_info(args):
    """打印启动信息"""
    print("\n" + "="*60)
    print("🚀 GPT-SoVITS 推理 API 服务器")
    print("="*60)
    print(f"📡 服务地址: http://{args.host}:{args.port}")
    print(f"📖 API文档: http://{args.host}:{args.port}/docs")
    print(f"🔧 工作进程: {args.workers}")
    print(f"📝 日志级别: {args.log_level}")
    print(f"🔄 自动重载: {'启用' if args.reload else '禁用'}")
    print(f"📊 访问日志: {'启用' if args.access_log else '禁用'}")
    
    # 显示环境信息
    gpt_sovits_root = os.environ.get("GPT_SOVITS_ROOT")
    models_dir = os.environ.get("MODELS_DIR")
    
    if gpt_sovits_root:
        print(f"📁 GPT-SoVITS根目录: {gpt_sovits_root}")
    if models_dir:
        print(f"🗂️ 模型目录: {models_dir}")
    
    print("\n📋 可用的API端点:")
    endpoints = [
        ("GET", "/", "服务状态"),
        ("GET", "/health", "健康检查"),
        ("POST", "/inference", "语音合成推理"),
        ("POST", "/inference/file", "文件上传推理"),
        ("POST", "/models/load", "加载模型"),
        ("GET", "/models/info", "模型信息"),
        ("GET", "/models/list", "模型列表"),
        ("POST", "/models/register", "注册模型"),
        ("POST", "/models/switch/{model_name}", "切换模型"),
        ("GET", "/utils/languages", "支持的语言"),
        ("GET", "/utils/formats", "支持的格式")
    ]
    
    for method, path, desc in endpoints:
        print(f"   {method:6} {path:30} - {desc}")
    
    print("\n💡 使用提示:")
    print("1. 首次使用需要先注册和加载模型")
    print("2. 访问 /docs 查看完整的API文档")
    print("3. 使用 Ctrl+C 停止服务")
    print("="*60)


def main():
    """主函数"""
    args = parse_args()
    ensure_audio_runtime(root=Path(__file__).resolve(), verify_torchcodec=True, strict=True, verbose=True)
    
    print("🔍 检查运行环境...")
    
    # 检查依赖项
    if not check_dependencies():
        sys.exit(1)
    
    # 设置环境变量
    setup_environment(args)
    
    # 测试API初始化
    test_api_initialization()
    
    # 打印启动信息
    print_startup_info(args)
    
    try:
        # 启动服务器
        uvicorn.run(
            "server:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=args.reload,
            log_level=args.log_level,
            access_log=args.access_log
        )
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"\n❌ 服务器启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
