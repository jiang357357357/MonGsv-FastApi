#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频特征提取服务启动脚本

启动 GPT-SoVITS 音频特征提取 FastAPI 服务
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

from Code.FastApi.Base.DatasetFormatting.audio_features.server import app


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="GPT-SoVITS 音频特征提取服务")
    
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0",
        help="服务器主机地址 (默认: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8003,
        help="服务器端口 (默认: 8003)"
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
        "--workers", 
        type=int, 
        default=1,
        help="工作进程数 (默认: 1)"
    )
    
    parser.add_argument(
        "--access-log", 
        action="store_true",
        help="启用访问日志"
    )
    
    return parser.parse_args()


def check_dependencies():
    """检查依赖项"""
    missing_deps = []
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import librosa
        print(f"✓ Librosa: {librosa.__version__}")
    except ImportError:
        missing_deps.append("librosa")
    
    try:
        import soundfile
        print(f"✓ SoundFile: {soundfile.__version__}")
    except ImportError:
        missing_deps.append("soundfile")
    
    try:
        import fastapi
        print(f"✓ FastAPI: {fastapi.__version__}")
    except ImportError:
        missing_deps.append("fastapi")
    
    try:
        import uvicorn
        print(f"✓ Uvicorn: {uvicorn.__version__}")
    except ImportError:
        missing_deps.append("uvicorn")
    
    if missing_deps:
        print(f"\n❌ 缺少依赖项: {', '.join(missing_deps)}")
        print("请安装缺少的依赖项:")
        print(f"pip install {' '.join(missing_deps)}")
        return False
    
    return True


def check_gpt_sovits_path():
    """检查GPT-SoVITS路径"""
    # 检查相对路径
    gpt_sovits_paths = [
        current_dir.parent.parent.parent.parent / "文档" / "GPT-SoVITS-main",
        Path("GPT-SoVITS"),
        Path("../GPT-SoVITS"),
        Path("../../GPT-SoVITS"),
    ]
    
    for path in gpt_sovits_paths:
        if path.exists():
            print(f"✓ 找到GPT-SoVITS路径: {path}")
            return True
    
    print("⚠️  未找到GPT-SoVITS路径，某些功能可能不可用")
    print("请确保GPT-SoVITS在以下位置之一:")
    for path in gpt_sovits_paths:
        print(f"  - {path}")
    
    return False


def print_startup_info(host: str, port: int):
    """打印启动信息"""
    print("\n" + "=" * 60)
    print("🎵 GPT-SoVITS 音频特征提取服务")
    print("=" * 60)
    print(f"服务地址: http://{host}:{port}")
    print(f"API文档: http://{host}:{port}/docs")
    print(f"健康检查: http://{host}:{port}/health")
    print("\n主要端点:")
    print(f"  POST /extract          - 同步特征提取")
    print(f"  POST /extract_async    - 异步特征提取")
    print(f"  GET  /status/{{task_id}} - 查询任务状态")
    print(f"  POST /analyze          - 数据集分析")
    print(f"  GET  /config/suggest   - 配置建议")
    print(f"  GET  /models/info      - 模型信息")
    print("=" * 60)


def main():
    """主函数"""
    args = parse_args()
    ensure_audio_runtime(root=Path(__file__).resolve(), verify_torchcodec=True, strict=True, verbose=True)
    
    print("启动音频特征提取服务...")
    
    # 检查依赖项
    if not check_dependencies():
        sys.exit(1)
    
    # 检查GPT-SoVITS路径
    check_gpt_sovits_path()
    
    # 打印启动信息
    print_startup_info(args.host, args.port)
    
    # 配置uvicorn参数
    uvicorn_config = {
        "app": app,
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level,
        "access_log": args.access_log,
    }
    
    # 开发模式配置
    if args.reload:
        uvicorn_config["reload"] = True
        uvicorn_config["reload_dirs"] = [str(current_dir)]
        print("🔄 开发模式: 启用自动重载")
    else:
        uvicorn_config["workers"] = args.workers
        if args.workers > 1:
            print(f"👥 生产模式: {args.workers} 个工作进程")
    
    try:
        # 启动服务器
        uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except Exception as e:
        print(f"\n❌ 服务启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
