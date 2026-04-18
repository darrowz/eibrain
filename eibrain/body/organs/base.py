"""Base organ contract."""

from __future__ import annotations

from eibrain.body.drivers import build_driver
from eibrain.body.drivers.base import DriverResult
from eibrain.body.health.organ_health import OrganHealth
from eibrain.body.health.organ_health import SubfunctionHealth
from eibrain.infra.config import OrganConfig, SubfunctionConfig


class BaseOrgan:
    name = "organ"
    subfunction_names: tuple[str, ...] = ()

    def __init__(self, *, config: OrganConfig | None = None) -> None:
        self.config = config or self.default_config()
        self.drivers = {
            name: build_driver(subfunction.driver)
            for name, subfunction in self.config.subfunctions.items()
        }

    @classmethod
    def default_config(cls) -> OrganConfig:
        return OrganConfig(
            enabled=True,
            subfunctions={name: SubfunctionConfig() for name in cls.subfunction_names},
        )

    def heartbeat(self) -> OrganHealth:
        subfunctions = {
            name: self._subfunction_health(name)
            for name in self.subfunction_names
        }
        statuses = [state.health for state in subfunctions.values()]
        if statuses and all(status == "healthy" for status in statuses):
            health = "healthy"
        elif any(status == "healthy" for status in statuses):
            health = "degraded"
        elif any(status == "degraded" for status in statuses):
            health = "degraded"
        else:
            health = "unavailable"
        return OrganHealth(organ=self.name, health=health, subfunctions=subfunctions)

    def supports_action(self, action) -> bool:
        return False

    def handle_action(self, action):
        return None

    def _subfunction_health(self, name: str) -> SubfunctionHealth:
        driver = self.drivers.get(name)
        if driver is None:
            return SubfunctionHealth(name=name, health="unavailable")
        heartbeat = driver.heartbeat()
        if isinstance(heartbeat, DriverResult):
            return SubfunctionHealth(
                name=name,
                health=self._normalize_status(heartbeat.status),
                details=dict(heartbeat.details),
            )
        return SubfunctionHealth(name=name, health=self._normalize_status(str(heartbeat)))

    @staticmethod
    def _normalize_status(status: str) -> str:
        if status in {"ok", "healthy"}:
            return "healthy"
        if status in {"error", "degraded"}:
            return "degraded"
        return "unavailable"
