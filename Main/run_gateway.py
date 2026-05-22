#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastApi 主运行入口。

统一从这里启动主网关服务。
"""

import sys
from pathlib import Path


def _setup_path():
    current_dir = Path(__file__).resolve().parent
    repo_root = current_dir.parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def main():
    repo_root = _setup_path()

    from Code.runtime_env import ensure_audio_runtime

    ensure_audio_runtime(root=repo_root, verify_torchcodec=True, strict=True, verbose=True)

    if len(sys.argv) > 1 and sys.argv[1] in {"status", "stop", "cleanup"}:
        from Code.FastApi.Tool.gateway_tools import main as tool_main

        raise SystemExit(tool_main(sys.argv[1:]))

    if len(sys.argv) > 1 and sys.argv[1] == "start":
        sys.argv.pop(1)

    from Code.FastApi.Tool.gateway_tools import gateway_port, get_gateway_status, stop_gateway

    port = gateway_port()
    status = get_gateway_status(port)
    if status.occupied:
        process_text = ", ".join(f"{item.name}(pid={item.pid})" for item in status.processes)
        print(f"检测到端口 {port} 已被占用: {process_text}")
        print("正在自动停止占用进程...")
        status = stop_gateway(port)
        if status.occupied:
            process_text = ", ".join(f"{item.name}(pid={item.pid})" for item in status.processes)
            print(f"自动清理失败，端口 {port} 仍被占用: {process_text}")
            raise SystemExit(1)
        print(f"端口 {port} 清理完成")

    from Code.FastApi.Base.Gateway.run_unified_gateway import main as run_main

    run_main()


if __name__ == "__main__":
    main()
