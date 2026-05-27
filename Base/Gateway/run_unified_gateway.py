#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 统一网关启动脚本

启动统一网关服务，整合所有模块功能。
"""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from Code.FastApi.Base.monconfig import MonConfig
from Code.runtime_env import ensure_audio_runtime


current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="GPT-SoVITS 统一网关服务")

    parser.add_argument("--host", type=str, default=None, help="服务器主机地址")
    parser.add_argument("--port", type=int, default=None, help="服务器端口")
    parser.add_argument("--workers", type=int, default=None, help="工作进程数")
    parser.add_argument(
        "--reload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否启用自动重载",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="日志级别",
    )
    parser.add_argument(
        "--access-log",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否启用访问日志",
    )
    parser.add_argument(
        "--enable-auth",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否启用 API 认证",
    )
    parser.add_argument("--api-key", type=str, default=None, help="API 密钥")
    parser.add_argument("--max-concurrent", type=int, default=None, help="最大并发任务数")
    return parser.parse_args()


def configure_stdio():
    """Ensure Windows console output does not fail on Unicode text."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def load_monconfig() -> MonConfig:
    """加载工作区 `.monconfig`。"""
    return MonConfig(start_path=Path(__file__).resolve())


def _pick(cli_value, config_value, default_value):
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default_value


def build_runtime_args(cli_args, mon_config: MonConfig):
    """合并 CLI、.monconfig 和默认值。"""
    runtime_args = SimpleNamespace(
        host=_pick(cli_args.host, mon_config.get("server", "HOST"), "0.0.0.0"),
        port=_pick(cli_args.port, mon_config.get("server", "PORT", cast=int), 8000),
        workers=_pick(
            cli_args.workers,
            mon_config.get("process", "WORKERS", cast=int),
            1,
        ),
        reload=_pick(
            cli_args.reload,
            mon_config.get("server", "RELOAD", cast=bool),
            False,
        ),
        log_level=_pick(
            cli_args.log_level,
            mon_config.get("log", "LEVEL"),
            "info",
        ),
        access_log=_pick(
            cli_args.access_log,
            mon_config.get("log", "ACCESS_LOG", cast=bool),
            False,
        ),
        enable_auth=_pick(
            cli_args.enable_auth,
            mon_config.get("server", "ENABLE_AUTH", cast=bool),
            False,
        ),
        api_key=_pick(
            cli_args.api_key,
            mon_config.get("server", "API_KEY"),
            None,
        ),
        max_concurrent=_pick(
            cli_args.max_concurrent,
            mon_config.get("server", "MAX_CONCURRENT_JOBS", cast=int),
            10,
        ),
    )

    if runtime_args.enable_auth and not runtime_args.api_key:
        print("错误: 启用认证时必须提供 API 密钥")
        sys.exit(1)

    return runtime_args


def check_dependencies():
    """检查依赖项。"""
    required_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "python-multipart",
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("错误: 缺少以下依赖包:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n请安装缺少的依赖包:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    print("依赖检查通过")
    return True


def setup_environment(args, mon_config: MonConfig):
    """将 `.monconfig` 注入统一环境变量。"""
    workspace_root = mon_config.workspace_root()
    if workspace_root:
        os.environ["MON_WORKSPACE_ROOT"] = str(workspace_root)

    loaded_files = mon_config.loaded_files()
    if loaded_files:
        os.environ["MON_CONFIG_FILE"] = str(loaded_files[0])

    if args.enable_auth:
        os.environ["ENABLE_AUTH"] = "true"
        os.environ["API_KEY"] = args.api_key
        print("已启用 API 认证")
    else:
        os.environ["ENABLE_AUTH"] = "false"
        os.environ.pop("API_KEY", None)

    os.environ["MAX_CONCURRENT_JOBS"] = str(args.max_concurrent)
    os.environ["LOG_LEVEL"] = str(args.log_level)
    os.environ["SERVER_PORT"] = str(args.port)

    path_env_map = {
        "GPT_SOVITS_ROOT": workspace_root,
        "MODELS_DIR": mon_config.resolve_path("paths", "MODELS_DIR"),
        "OUTPUT_DIR": mon_config.resolve_path("paths", "OUTPUT_DIR"),
        "TEMP_DIR": mon_config.resolve_path("paths", "TEMP_DIR"),
        "PRETRAINED_MODELS_DIR": mon_config.resolve_path("model", "PRETRAINED_MODELS_DIR"),
        "UVR5_WEIGHTS_DIR": mon_config.resolve_path("model", "UVR5_WEIGHTS_DIR"),
    }
    for env_key, path_value in path_env_map.items():
        if path_value:
            os.environ[env_key] = str(path_value)

    gpu_enabled = mon_config.get("gpu", "ENABLED", cast=bool)
    if gpu_enabled is not None:
        os.environ["USE_GPU"] = "true" if gpu_enabled else "false"

    gpu_device = mon_config.get("gpu", "DEVICE")
    if gpu_device:
        os.environ["GPU_DEVICE"] = gpu_device

    half_precision = mon_config.get("gpu", "HALF_PRECISION", cast=bool)
    if half_precision is not None:
        os.environ["is_half"] = "True" if half_precision else "False"

    register_host = mon_config.get("monhub", "REGISTER_HOST")
    if not register_host:
        register_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host

    monhub_env = {
        "MONHUB_ENABLED": "true" if mon_config.get("monhub", "ENABLED", False, cast=bool) else "false",
        "MONHUB_HOST": mon_config.get("monhub", "HOST", "127.0.0.1"),
        "MONHUB_PORT": str(mon_config.get("monhub", "PORT", 40051, cast=int)),
        "MONHUB_DISCOVERY_ENABLED": "true" if mon_config.get("monhub", "DISCOVERY_ENABLED", True, cast=bool) else "false",
        "MONHUB_DISCOVERY_PORT": str(mon_config.get("monhub", "DISCOVERY_PORT", 40053, cast=int)),
        "MONHUB_SERVICE_ID": mon_config.get("monhub", "SERVICE_ID", "MonGsvFastapi"),
        "MONHUB_SERVICE_NAME": mon_config.get("monhub", "SERVICE_NAME", mon_config.get("service", "NAME", "MonGsvFastapi")),
        "MONHUB_SERVICE_TYPE": mon_config.get("monhub", "SERVICE_TYPE", "tts_service"),
        "MONHUB_SERVICE_VERSION": mon_config.get("service", "VERSION", "1.0.0"),
        "MONHUB_SERVICE_DESCRIPTION": mon_config.get("service", "DESCRIPTION", "GPT-SoVITS FastAPI voice service"),
        "MONHUB_HEARTBEAT_INTERVAL": str(mon_config.get("monhub", "HEARTBEAT_INTERVAL", 30, cast=int)),
        "MONHUB_REGISTER_HOST": register_host,
        "MONHUB_REGISTER_PORT": str(mon_config.get("monhub", "REGISTER_PORT", args.port, cast=int)),
    }
    for env_key, env_value in monhub_env.items():
        os.environ[env_key] = str(env_value)

    print(f"最大并发任务数: {args.max_concurrent}")


def test_services(service_manager):
    """测试服务加载。"""
    try:
        total_services = len(service_manager.services)
        print(f"成功加载 {total_services} 个服务:")
        for name, service_info in service_manager.services.items():
            print(f"   - {name:20} -> {service_info['prefix']}")

        if total_services == 0:
            print("警告: 没有加载任何服务")
            return False
        return True
    except Exception as exc:
        print(f"错误: 服务加载测试失败: {exc}")
        return False


def print_startup_info(args, mon_config: MonConfig):
    """打印启动信息。"""
    docs_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    workspace_root = mon_config.workspace_root()

    print("\n" + "=" * 70)
    print("GPT-SoVITS 统一网关服务")
    print("=" * 70)
    print(f"监听地址: http://{args.host}:{args.port}")
    print(f"文档地址: http://{docs_host}:{args.port}/docs")
    print(f"工作进程: {args.workers}")
    print(f"日志级别: {args.log_level}")
    print(f"自动重载: {'启用' if args.reload else '禁用'}")
    print(f"访问日志: {'启用' if args.access_log else '禁用'}")
    print(f"API认证: {'启用' if args.enable_auth else '禁用'}")
    monhub_enabled = mon_config.get("monhub", "ENABLED", False, cast=bool)
    if monhub_enabled:
        monhub_host = mon_config.get("monhub", "HOST", "127.0.0.1")
        monhub_port = mon_config.get("monhub", "PORT", 40051, cast=int)
        monhub_discovery = mon_config.get("monhub", "DISCOVERY_ENABLED", True, cast=bool)
        print(f"MonHub注册: 启用 ({monhub_host}:{monhub_port}, UDP发现: {'启用' if monhub_discovery else '禁用'})")
    else:
        print("MonHub注册: 禁用")
    if workspace_root:
        print(f"工作区根目录: {workspace_root}")
    for config_file in mon_config.loaded_files():
        print(f"配置文件: {config_file}")

    print("\n统一 API 端点:")
    endpoints = [
        ("数据准备", [("POST", "/data-prep/audio-slice/process", "音频切分"), ("POST", "/data-prep/asr/recognize", "ASR识别")]),
        ("数据格式化", [("POST", "/dataset/text/extract", "文本特征提取"), ("POST", "/dataset/audio/extract", "音频特征提取"), ("POST", "/dataset/semantic/encode", "语义编码")]),
        ("模型训练", [("POST", "/training/gpt/start", "GPT训练"), ("POST", "/training/sovits/start", "SoVITS训练"), ("GET", "/training/status/{job_id}", "训练状态")]),
        ("推理服务", [("POST", "/inference/tts", "文本转语音"), ("WS", "/ws/tts/stream", "流式文本转语音")]),
        ("实时语音", [("WS", "/ws/asr/final", "推荐：VAD断句最终识别"), ("WS", "/ws/asr/transcribe", "2-pass流式识别"), ("WS", "/ws/asr/vad", "VAD检测")]),
        ("工作流", [("POST", "/workflow/complete", "完整流程/可选带训练"), ("POST", "/workflow/training/full", "预处理到训练的引导流程"), ("POST", "/batch/projects", "批量处理")]),
        ("管理监控", [("GET", "/health", "健康检查"), ("GET", "/services/status", "服务状态"), ("GET", "/monhub/status", "MonHub注册状态")]),
    ]
    for category, apis in endpoints:
        print(f"\n{category}:")
        for method, path, desc in apis:
            print(f"   {method:6} {path:35} - {desc}")

    print("\n快速开始:")
    print("1. 访问 /docs 查看完整API文档")
    print("2. 使用 /workflow/complete 执行完整流程")
    print("3. 使用 /workflow/training/full 执行预处理到训练的引导流程")
    print("4. 使用 /health 检查服务状态")
    print("5. 使用 Ctrl+C 停止服务")
    print("=" * 70)


def main():
    """主函数。"""
    configure_stdio()
    cli_args = parse_args()
    mon_config = load_monconfig()
    args = build_runtime_args(cli_args, mon_config)
    ensure_audio_runtime(root=mon_config.workspace_root() or Path(__file__).resolve(), verify_torchcodec=True, strict=True, verbose=True)

    print("检查运行环境...")
    if not check_dependencies():
        sys.exit(1)

    setup_environment(args, mon_config)

    from Code.FastApi.Base.Gateway.unified_gateway import app, service_manager

    if not test_services(service_manager):
        print("警告: 服务加载存在问题，但仍将启动服务")

    print_startup_info(args, mon_config)

    try:
        import uvicorn

        uvicorn_kwargs = {
            "host": args.host,
            "port": args.port,
            "reload": args.reload,
            "log_level": args.log_level,
            "access_log": args.access_log,
        }

        # Keep single-process startup on the in-memory app object to avoid
        # a second import of `unified_gateway`, which would reload all services.
        if args.reload or args.workers > 1:
            uvicorn.run(
                "Code.FastApi.Base.Gateway.unified_gateway:app",
                workers=args.workers,
                **uvicorn_kwargs,
            )
        else:
            uvicorn.run(
                app,
                **uvicorn_kwargs,
            )
    except KeyboardInterrupt:
        print("\n统一网关服务已停止")
    except Exception as exc:
        print(f"\n错误: 服务器启动失败: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
