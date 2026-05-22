#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型驻留管理器。

负责追踪模型加载、使用、空闲超时和淘汰候选。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
from typing import Dict, List, Optional


@dataclass
class ResidencyConfig:
    """驻留策略配置。"""

    idle_ttl_seconds: int = 20 * 60
    max_loaded_models: int = 1
    cleanup_interval_seconds: int = 60


@dataclass
class ResidencyRecord:
    """单个模型驻留记录。"""

    model_key: str
    gpt_path: str
    sovits_path: str
    loaded_at: datetime = field(default_factory=datetime.utcnow)
    last_used_at: datetime = field(default_factory=datetime.utcnow)
    active_requests: int = 0
    status: str = "loaded"
    unload_reason: Optional[str] = None

    def touch(self, now: Optional[datetime] = None) -> None:
        self.last_used_at = now or datetime.utcnow()

    def mark_loaded(self, now: Optional[datetime] = None) -> None:
        current = now or datetime.utcnow()
        self.loaded_at = current
        self.last_used_at = current
        self.status = "loaded"
        self.unload_reason = None

    def mark_unloaded(self, reason: str, now: Optional[datetime] = None) -> None:
        self.last_used_at = now or datetime.utcnow()
        self.status = "unloaded"
        self.unload_reason = reason
        self.active_requests = 0

    def is_idle_expired(self, config: ResidencyConfig, now: Optional[datetime] = None) -> bool:
        if self.active_requests > 0 or self.status != "loaded":
            return False
        current = now or datetime.utcnow()
        return current - self.last_used_at >= timedelta(seconds=config.idle_ttl_seconds)


class ModelResidencyManager:
    """管理模型在内存/显存中的驻留状态。"""

    def __init__(self, config: Optional[ResidencyConfig] = None):
        self.config = config or ResidencyConfig()
        self._records: Dict[str, ResidencyRecord] = {}
        self._current_model_key: Optional[str] = None
        self._last_cleanup_at: Optional[datetime] = None
        self._lock = RLock()

    @staticmethod
    def build_model_key(gpt_path: str, sovits_path: str) -> str:
        return f"{gpt_path}::{sovits_path}"

    def register_loaded(self, gpt_path: str, sovits_path: str, now: Optional[datetime] = None) -> ResidencyRecord:
        with self._lock:
            current = now or datetime.utcnow()
            model_key = self.build_model_key(gpt_path, sovits_path)
            record = self._records.get(model_key)
            if record is None:
                record = ResidencyRecord(model_key=model_key, gpt_path=gpt_path, sovits_path=sovits_path)
                self._records[model_key] = record
            record.mark_loaded(current)
            self._current_model_key = model_key
            return record

    def mark_used(self, model_key: Optional[str] = None, now: Optional[datetime] = None) -> Optional[ResidencyRecord]:
        with self._lock:
            key = model_key or self._current_model_key
            if not key or key not in self._records:
                return None
            record = self._records[key]
            record.touch(now)
            return record

    def begin_request(self, model_key: Optional[str] = None, now: Optional[datetime] = None) -> Optional[ResidencyRecord]:
        with self._lock:
            record = self.mark_used(model_key, now)
            if record is None:
                return None
            record.active_requests += 1
            return record

    def end_request(self, model_key: Optional[str] = None, now: Optional[datetime] = None) -> Optional[ResidencyRecord]:
        with self._lock:
            key = model_key or self._current_model_key
            if not key or key not in self._records:
                return None
            record = self._records[key]
            if record.active_requests > 0:
                record.active_requests -= 1
            record.touch(now)
            return record

    def mark_unloaded(self, model_key: Optional[str] = None, reason: str = "manual", now: Optional[datetime] = None) -> None:
        with self._lock:
            key = model_key or self._current_model_key
            if not key or key not in self._records:
                return
            self._records[key].mark_unloaded(reason, now)
            if self._current_model_key == key:
                self._current_model_key = None

    def get_current_record(self) -> Optional[ResidencyRecord]:
        with self._lock:
            if not self._current_model_key:
                return None
            return self._records.get(self._current_model_key)

    def list_records(self) -> List[ResidencyRecord]:
        with self._lock:
            return sorted(
                self._records.values(),
                key=lambda item: item.last_used_at,
                reverse=True,
            )

    def get_expired_records(self, now: Optional[datetime] = None) -> List[ResidencyRecord]:
        with self._lock:
            current = now or datetime.utcnow()
            return [
                record
                for record in self._records.values()
                if record.is_idle_expired(self.config, current)
            ]

    def select_eviction_candidates(self, now: Optional[datetime] = None) -> List[ResidencyRecord]:
        with self._lock:
            current = now or datetime.utcnow()
            loaded_records = [
                record for record in self._records.values()
                if record.status == "loaded"
            ]
            loaded_records.sort(key=lambda item: item.last_used_at)

            expired = [
                record for record in loaded_records
                if record.is_idle_expired(self.config, current)
            ]
            if expired:
                return expired

            overflow = max(0, len(loaded_records) - self.config.max_loaded_models)
            if overflow <= 0:
                return []
            return [
                record for record in loaded_records
                if record.active_requests == 0
            ][:overflow]

    def should_run_cleanup(self, now: Optional[datetime] = None) -> bool:
        with self._lock:
            current = now or datetime.utcnow()
            if self._last_cleanup_at is None:
                return True
            return current - self._last_cleanup_at >= timedelta(seconds=self.config.cleanup_interval_seconds)

    def mark_cleanup_run(self, now: Optional[datetime] = None) -> None:
        with self._lock:
            self._last_cleanup_at = now or datetime.utcnow()

    def get_status(self) -> dict:
        with self._lock:
            return {
                "current_model_key": self._current_model_key,
                "config": {
                    "idle_ttl_seconds": self.config.idle_ttl_seconds,
                    "max_loaded_models": self.config.max_loaded_models,
                    "cleanup_interval_seconds": self.config.cleanup_interval_seconds,
                },
                "records": [
                    {
                        "model_key": record.model_key,
                        "gpt_path": record.gpt_path,
                        "sovits_path": record.sovits_path,
                        "loaded_at": record.loaded_at.isoformat(),
                        "last_used_at": record.last_used_at.isoformat(),
                        "active_requests": record.active_requests,
                        "status": record.status,
                        "unload_reason": record.unload_reason,
                    }
                    for record in self.list_records()
                ],
            }
