"""No-op driver implementation."""

from __future__ import annotations

from .base import DriverResult


class NoopDriver:
    def heartbeat(self) -> DriverResult:
        return DriverResult(status="healthy", details={"driver": "noop"})

    def invoke(self, operation: str, payload: dict[str, object]) -> DriverResult:
        return DriverResult(status="ok", details={"operation": operation, "payload": dict(payload)})
