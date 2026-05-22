"""
Role persistence repository.
"""

from __future__ import annotations

from Code.FastApi.Domain.Role.Models.role_info import RoleInfo
from Code.FastApi.Domain.common import JsonResourceStore


class RoleRepository:
    def __init__(self):
        self._store = JsonResourceStore("roles.json")

    def list_items(self) -> list[RoleInfo]:
        return [RoleInfo.model_validate(item) for item in self._store.load_items()]

    def save_items(self, roles: list[RoleInfo]) -> None:
        self._store.save_items([role.model_dump() for role in roles])

    def next_id(self) -> int:
        return self._store.next_id()

