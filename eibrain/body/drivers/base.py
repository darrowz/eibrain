"""Base driver contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class DriverResult:
    status: str = "ok"
    details: dict[str, Any] = field(default_factory=dict)


class DriverAdapter(Protocol):
    def heartbeat(self) -> str | DriverResult: ...

    def invoke(self, operation: str, payload: dict[str, Any]) -> DriverResult: ...
