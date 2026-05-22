#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gateway runtime tools for Code/FastApi."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from Code.FastApi.Base.monconfig import MonConfig


@dataclass
class GatewayProcessInfo:
    pid: int
    name: str


@dataclass
class GatewayStatus:
    port: int
    occupied: bool
    processes: List[GatewayProcessInfo]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_monconfig() -> MonConfig:
    return MonConfig(start_path=_repo_root() / "Code" / "FastApi" / "Main" / "run_gateway.py")


def gateway_port(config: Optional[MonConfig] = None) -> int:
    config = config or load_monconfig()
    return config.get("server", "PORT", 40032, cast=int)


def gateway_temp_dir(config: Optional[MonConfig] = None) -> Path:
    config = config or load_monconfig()
    temp_dir = config.resolve_path("paths", "TEMP_DIR", "Data/Temp")
    assert temp_dir is not None
    return temp_dir


def _port_processes_windows(port: int) -> List[GatewayProcessInfo]:
    if os.name != "nt":
        raise NotImplementedError("This tool currently supports Windows only.")

    result = subprocess.run(
        ["netstat", "-ano", "-p", "tcp"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode != 0 or not result.stdout:
        return []

    process_ids: List[int] = []
    port_suffix = f":{port}"
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("TCP"):
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        local_address = parts[1]
        state = parts[3].upper()
        pid_text = parts[4]

        if not local_address.endswith(port_suffix):
            continue

        # 只有 LISTENING 才会真正阻止服务重新绑定端口。
        # FIN_WAIT_2 / CLOSE_WAIT / TIME_WAIT 等都是连接收尾态，不应视为网关占用。
        if state != "LISTENING":
            continue
        if not pid_text.isdigit():
            continue

        pid = int(pid_text)
        if pid <= 0:
            continue

        if pid not in process_ids:
            process_ids.append(pid)

    processes: List[GatewayProcessInfo] = []
    for pid in process_ids:
        try:
            name = _process_name_windows(pid)
            processes.append(GatewayProcessInfo(pid=pid, name=name))
        except Exception:
            processes.append(GatewayProcessInfo(pid=pid, name="unknown"))
    return processes


def _process_name_windows(pid: int) -> str:
    if pid <= 0:
        return "system"

    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return "unknown"

    first_line = result.stdout.splitlines()[0].strip()
    if not first_line or first_line.startswith("INFO:"):
        return "unknown"

    if first_line.startswith("\""):
        parts = [item.strip("\"") for item in first_line.split("\",\"")]
        if parts and parts[0]:
            return Path(parts[0]).stem

    name = result.stdout.strip()
    return name or "unknown"


def get_gateway_status(port: Optional[int] = None) -> GatewayStatus:
    port = port or gateway_port()
    processes = _port_processes_windows(port)
    return GatewayStatus(port=port, occupied=bool(processes), processes=processes)


def stop_gateway(port: Optional[int] = None) -> GatewayStatus:
    status = get_gateway_status(port)
    if not status.occupied:
        return status

    if os.name != "nt":
        raise NotImplementedError("This tool currently supports Windows only.")

    target_pids = [process.pid for process in status.processes if process.pid > 0]

    for process in status.processes:
        if process.pid <= 0:
            continue
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"Stop-Process -Id {process.pid} -Force -ErrorAction SilentlyContinue",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )

    for _ in range(10):
        time.sleep(0.5)
        current_status = get_gateway_status(status.port)
        remaining_pids = {process.pid for process in current_status.processes}
        if not any(pid in remaining_pids for pid in target_pids):
            return current_status

    for pid in target_pids:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/F", "/T"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )

    for _ in range(10):
        time.sleep(0.5)
        current_status = get_gateway_status(status.port)
        remaining_pids = {process.pid for process in current_status.processes}
        if not any(pid in remaining_pids for pid in target_pids):
            return current_status

    return get_gateway_status(status.port)


def cleanup_gateway(port: Optional[int] = None, keep_temp: bool = False) -> dict:
    before = get_gateway_status(port)
    after_stop = stop_gateway(before.port)

    temp_dir = gateway_temp_dir()
    cleaned = False
    removed_entries = 0

    if not keep_temp and temp_dir.exists():
        for child in list(temp_dir.iterdir()):
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=False)
            else:
                child.unlink(missing_ok=True)
            removed_entries += 1
        cleaned = True

    return {
        "port": before.port,
        "was_occupied": before.occupied,
        "processes_before": before.processes,
        "processes_after": after_stop.processes,
        "temp_dir": str(temp_dir),
        "temp_cleaned": cleaned,
        "removed_entries": removed_entries,
    }


def print_status(status: GatewayStatus):
    print(f"Gateway port: {status.port}")
    print(f"Occupied: {'yes' if status.occupied else 'no'}")
    for process in status.processes:
        print(f"Process: pid={process.pid} name={process.name}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Code/FastApi gateway tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("status", help="Show gateway port status")
    status_parser.add_argument("--port", type=int, help="Override gateway port")

    stop_parser = subparsers.add_parser("stop", help="Stop gateway process on the configured port")
    stop_parser.add_argument("--port", type=int, help="Override gateway port")

    cleanup_parser = subparsers.add_parser("cleanup", help="Stop gateway and clean TEMP_DIR")
    cleanup_parser.add_argument("--port", type=int, help="Override gateway port")
    cleanup_parser.add_argument("--keep-temp", action="store_true", help="Skip TEMP_DIR cleanup")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "status":
        print_status(get_gateway_status(args.port))
        return 0

    if args.command == "stop":
        before = get_gateway_status(args.port)
        print_status(before)
        after = stop_gateway(args.port)
        print("After stop:")
        print_status(after)
        return 0

    if args.command == "cleanup":
        result = cleanup_gateway(args.port, keep_temp=args.keep_temp)
        print(f"Gateway port: {result['port']}")
        print(f"Was occupied: {'yes' if result['was_occupied'] else 'no'}")
        print(f"Temp dir: {result['temp_dir']}")
        print(f"Temp cleaned: {'yes' if result['temp_cleaned'] else 'no'}")
        print(f"Removed entries: {result['removed_entries']}")
        return 0

    parser.error("Unknown command")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
