"""HTTP-backed driver."""

from __future__ import annotations

import json
from urllib import request

from eibrain.infra.config import DriverConfig

from .base import DriverResult


class HttpDriver:
    def __init__(self, config: DriverConfig) -> None:
        self.config = config

    def heartbeat(self) -> DriverResult:
        if not self.config.endpoint:
            return DriverResult(status="unavailable", details={"reason": "missing_endpoint"})
        health_endpoint = str(self.config.extra.get("health_endpoint", ""))
        if not health_endpoint:
            return DriverResult(status="healthy", details={"driver": "http", "endpoint": self.config.endpoint})
        try:
            req = request.Request(health_endpoint, method="GET", headers=self.config.headers)
            with request.urlopen(req, timeout=self.config.timeout_s) as response:
                body = response.read().decode("utf-8")
                return DriverResult(
                    status="healthy" if response.status < 400 else "degraded",
                    details={"driver": "http", "endpoint": health_endpoint, "response": body},
                )
        except Exception as exc:
            return DriverResult(
                status="degraded",
                details={"driver": "http", "endpoint": health_endpoint, "error": str(exc)},
            )

    def invoke(self, operation: str, payload: dict[str, object]) -> DriverResult:
        if not self.config.endpoint:
            return DriverResult(status="unavailable", details={"reason": "missing_endpoint"})
        body = json.dumps({"operation": operation, "payload": payload}).encode("utf-8")
        req = request.Request(
            self.config.endpoint,
            data=body,
            method=self.config.method,
            headers={"Content-Type": "application/json", **self.config.headers},
        )
        with request.urlopen(req, timeout=self.config.timeout_s) as response:
            response_body = response.read().decode("utf-8")
            return DriverResult(
                status="ok" if response.status < 400 else "error",
                details={"operation": operation, "response": response_body},
            )
