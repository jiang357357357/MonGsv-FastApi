#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight MonHub service registration bridge.

The reference MonHub project is intentionally kept independent.  This module
speaks the small service-management protocol directly so the FastAPI gateway can
register itself without importing MonHub's Python 3.12 package tree.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"true", "yes", "1", "on", "enabled"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class MonHubBridgeConfig:
    enabled: bool = False
    hub_host: str = "127.0.0.1"
    hub_port: int = 40051
    discovery_enabled: bool = True
    discovery_port: int = 40053
    service_id: str = "MonGsvFastapi"
    service_name: str = "MonGsvFastapi"
    service_type: str = "tts_service"
    service_version: str = "1.0.0"
    description: str = "GPT-SoVITS FastAPI voice service"
    register_host: str = "127.0.0.1"
    register_port: int = 40302
    heartbeat_interval: int = 30


class MonHubBridge:
    """Register this gateway in MonHub and maintain heartbeats."""

    def __init__(self, config: MonHubBridgeConfig):
        self.config = config
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._status_lock = threading.Lock()
        self._connected = False
        self._registered = False
        self._hub_address: Optional[str] = None
        self._last_error: Optional[str] = None
        self._last_reply: Optional[dict[str, Any]] = None
        self._last_heartbeat_at: Optional[str] = None
        self._started_at: Optional[str] = None

    def start(self) -> None:
        if not self.config.enabled:
            self._set_status(error=None, connected=False, registered=False)
            return
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._started_at = _utc_now()
        self._thread = threading.Thread(target=self._run, name="monhub-bridge", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._set_status(connected=False, registered=False)

    def status(self) -> dict[str, Any]:
        with self._status_lock:
            return {
                "enabled": self.config.enabled,
                "connected": self._connected,
                "registered": self._registered,
                "hub_address": self._hub_address,
                "service_id": self.config.service_id,
                "service_name": self.config.service_name,
                "service_type": self.config.service_type,
                "register_host": self.config.register_host,
                "register_port": self.config.register_port,
                "heartbeat_interval": self.config.heartbeat_interval,
                "last_heartbeat_at": self._last_heartbeat_at,
                "last_error": self._last_error,
                "last_reply": self._last_reply,
                "started_at": self._started_at,
            }

    def _set_status(
        self,
        *,
        connected: Optional[bool] = None,
        registered: Optional[bool] = None,
        hub_address: Optional[str] = None,
        error: Optional[str] = None,
        reply: Optional[dict[str, Any]] = None,
        heartbeat_at: Optional[str] = None,
    ) -> None:
        with self._status_lock:
            if connected is not None:
                self._connected = connected
            if registered is not None:
                self._registered = registered
            if hub_address is not None:
                self._hub_address = hub_address
            if error is not None:
                self._last_error = error
            if reply is not None:
                self._last_reply = reply
            if heartbeat_at is not None:
                self._last_heartbeat_at = heartbeat_at

    def _run(self) -> None:
        try:
            import zmq  # type: ignore
        except ImportError:
            self._set_status(error="pyzmq is not installed", connected=False, registered=False)
            return

        context = None
        socket_obj = None
        registered = False
        try:
            hub_address = self._resolve_hub_address()
            self._set_status(hub_address=hub_address, error="")

            context = zmq.Context()
            socket_obj = context.socket(zmq.DEALER)
            socket_obj.setsockopt_string(zmq.IDENTITY, self.config.service_id)
            socket_obj.setsockopt(zmq.LINGER, 500)
            socket_obj.connect(hub_address)
            poller = zmq.Poller()
            poller.register(socket_obj, zmq.POLLIN)

            self._send(socket_obj, "SERVICE_REGISTER", self._service_payload())
            registered = True
            self._set_status(connected=True, registered=True)

            next_heartbeat = time.monotonic()
            while not self._stop_event.is_set():
                events = dict(poller.poll(timeout=200))
                if socket_obj in events:
                    self._receive(socket_obj)

                now = time.monotonic()
                if now >= next_heartbeat:
                    heartbeat_at = _utc_now()
                    self._send(socket_obj, "SERVICE_HEARTBEAT", self._heartbeat_payload())
                    self._set_status(heartbeat_at=heartbeat_at)
                    next_heartbeat = now + max(5, self.config.heartbeat_interval)

        except Exception as exc:
            self._set_status(error=str(exc), connected=False, registered=False)
        finally:
            if socket_obj is not None and registered:
                try:
                    self._send(socket_obj, "SERVICE_UNREGISTER", {"service_id": self.config.service_id})
                except Exception:
                    pass
            if socket_obj is not None:
                try:
                    socket_obj.close()
                except Exception:
                    pass
            if context is not None:
                try:
                    context.term()
                except Exception:
                    pass
            self._set_status(connected=False, registered=False)

    def _resolve_hub_address(self) -> str:
        if self.config.discovery_enabled:
            discovered = self._discover_hub_address()
            if discovered:
                return discovered
        return f"tcp://{self.config.hub_host}:{self.config.hub_port}"

    def _discover_hub_address(self) -> Optional[str]:
        request = {
            "type": "SERVICE_DISCOVER",
            "source": self.config.service_id,
            "timestamp": _utc_now(),
            "payload": {},
        }
        data = json.dumps(request, ensure_ascii=False).encode("utf-8")
        targets = [
            ("255.255.255.255", self.config.discovery_port),
            ("127.0.0.1", self.config.discovery_port),
        ]

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
            udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            udp_socket.settimeout(1.0)
            for target in targets:
                try:
                    udp_socket.sendto(data, target)
                except OSError:
                    continue
            end_at = time.monotonic() + 2.0
            while time.monotonic() < end_at:
                try:
                    response_data, address = udp_socket.recvfrom(65535)
                except socket.timeout:
                    break
                try:
                    response = json.loads(response_data.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if response.get("type") != "DISCOVERY_RESPONSE":
                    continue
                payload = response.get("payload", {}) or {}
                registry_info = payload.get("registry_info", {}) or {}
                hub_port = registry_info.get("hub_zmq_port") or self.config.hub_port
                hub_ip = registry_info.get("registry_ip") or address[0]
                return f"tcp://{hub_ip}:{hub_port}"
        return None

    def _send(self, socket_obj: Any, message_type: str, payload: dict[str, Any]) -> None:
        message = {
            "protocol": "MonHub",
            "version": "2.0.0",
            "msg_id": str(uuid.uuid4()),
            "type": message_type,
            "source": self.config.service_id,
            "target": "MonHub",
            "timestamp": _utc_now(),
            "payload": payload,
        }
        socket_obj.send_multipart([b"", json.dumps(message, ensure_ascii=False).encode("utf-8")])

    def _receive(self, socket_obj: Any) -> None:
        frames = socket_obj.recv_multipart()
        if not frames:
            return
        try:
            reply = json.loads(frames[-1].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        self._set_status(reply=reply)

    def _service_payload(self) -> dict[str, Any]:
        base_url = f"http://{self.config.register_host}:{self.config.register_port}"
        return {
            "service_id": self.config.service_id,
            "service_name": self.config.service_name,
            "service_type": self.config.service_type,
            "version": self.config.service_version,
            "status": "online",
            "description": self.config.description,
            "endpoints": [
                {"name": "api", "protocol": "http", "host": self.config.register_host, "port": self.config.register_port, "path": "/", "primary": True},
                {"name": "docs", "protocol": "http", "host": self.config.register_host, "port": self.config.register_port, "path": "/docs"},
                {"name": "health", "protocol": "http", "host": self.config.register_host, "port": self.config.register_port, "path": "/health"},
                {"name": "tts", "protocol": "http", "host": self.config.register_host, "port": self.config.register_port, "path": "/inference/tts"},
                {"name": "transcribe", "protocol": "http", "host": self.config.register_host, "port": self.config.register_port, "path": "/inference/transcribe"},
            ],
            "capabilities": [
                "http_api",
                "tts",
                "asr",
                "transcribe",
                "dataset_preparation",
                "training",
                "model_management",
                "openapi",
            ],
            "metadata": {
                "base_url": base_url,
                "docs_url": f"{base_url}/docs",
                "health_url": f"{base_url}/health",
                "framework": "fastapi",
            },
        }

    def _heartbeat_payload(self) -> dict[str, Any]:
        return {
            "service_id": self.config.service_id,
            "service_name": self.config.service_name,
            "status": "online",
            "health": 100,
            "load": 0,
            "timestamp": _utc_now(),
        }


def create_monhub_bridge_from_env() -> MonHubBridge:
    register_port = _env_int("MONHUB_REGISTER_PORT", _env_int("SERVER_PORT", 40302))
    config = MonHubBridgeConfig(
        enabled=_env_bool("MONHUB_ENABLED", False),
        hub_host=os.environ.get("MONHUB_HOST", "127.0.0.1"),
        hub_port=_env_int("MONHUB_PORT", 40051),
        discovery_enabled=_env_bool("MONHUB_DISCOVERY_ENABLED", True),
        discovery_port=_env_int("MONHUB_DISCOVERY_PORT", 40053),
        service_id=os.environ.get("MONHUB_SERVICE_ID", "MonGsvFastapi"),
        service_name=os.environ.get("MONHUB_SERVICE_NAME", "MonGsvFastapi"),
        service_type=os.environ.get("MONHUB_SERVICE_TYPE", "tts_service"),
        service_version=os.environ.get("MONHUB_SERVICE_VERSION", "1.0.0"),
        description=os.environ.get("MONHUB_SERVICE_DESCRIPTION", "GPT-SoVITS FastAPI voice service"),
        register_host=os.environ.get("MONHUB_REGISTER_HOST", "127.0.0.1"),
        register_port=register_port,
        heartbeat_interval=_env_int("MONHUB_HEARTBEAT_INTERVAL", 30),
    )
    return MonHubBridge(config)
