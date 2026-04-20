"""Neck organ implementation."""

from __future__ import annotations

import time

from eibrain.body.health.organ_health import OrganHealth, SubfunctionHealth
from eibrain.body.organs.base import BaseOrgan
from eibrain.body.runtime_linux import map_target_x_to_angle
from eibrain.protocol.actions import MoveHeadAction
from eibrain.protocol.outcomes import ActionExecuted


class NeckOrgan(BaseOrgan):
    name = "neck"
    subfunction_names = ("motor", "tracking")

    def __init__(self, *, config=None) -> None:
        super().__init__(config=config)
        self._last_tracking: dict[str, object] | None = None

    def heartbeat(self) -> OrganHealth:
        if self._driver_kind("tracking") == "noop":
            return super().heartbeat()
        motor_state = self._subfunction_health("motor")
        tracking_state = self._tracking_health(motor_state=motor_state)
        subfunctions = {
            "motor": motor_state,
            "tracking": tracking_state,
        }
        statuses = [state.health for state in subfunctions.values()]
        if statuses and all(status == "healthy" for status in statuses):
            health = "healthy"
        elif any(status == "healthy" for status in statuses) or any(status == "degraded" for status in statuses):
            health = "degraded"
        else:
            health = "unavailable"
        return OrganHealth(organ=self.name, health=health, subfunctions=subfunctions)

    def supports_action(self, action) -> bool:
        return isinstance(action, MoveHeadAction)

    def handle_action(self, action):
        if not isinstance(action, MoveHeadAction):
            return None
        config = self.config.subfunctions.get("motor")
        extra = config.driver.extra if config is not None else {}
        pan_min = int(extra.get("pan_min", 40))
        pan_max = int(extra.get("pan_max", 140))
        target_angle = action.target_angle
        if target_angle is None and action.target_x is not None:
            target_angle = map_target_x_to_angle(target_x=action.target_x, pan_min=pan_min, pan_max=pan_max)
        result = self.drivers["motor"].invoke(
            "move_head",
            {
                "target_id": action.target_id,
                "target_name": action.target_name,
                "target_x": action.target_x,
                "target_angle": target_angle,
            },
        )
        self._last_tracking = {
            "target_id": action.target_id,
            "target_name": action.target_name,
            "target_x": action.target_x,
            "target_angle": target_angle,
            "tracked_at_ts": action.ts or time.time(),
            "status": result.status,
        }
        return ActionExecuted(
            ts=action.ts,
            source="neck.motor",
            status=result.status,
            session_id=action.session_id,
            actor_id=action.actor_id,
            target_id=action.target_id,
            action_kind=action.kind,
            details=result.details,
        )

    def _tracking_health(self, *, motor_state: SubfunctionHealth) -> SubfunctionHealth:
        probe = self.drivers["tracking"].heartbeat()
        details = self._merge_probe_details(dict(probe.details))
        if self._last_tracking is not None:
            details.update(self._last_tracking)
        if motor_state.health == "unavailable":
            health = "unavailable"
            details["status"] = "motor_unavailable"
            details["error"] = "motor_unavailable"
        elif motor_state.health == "degraded":
            health = "degraded"
            details["status"] = "motor_degraded"
            details["error"] = "motor_degraded"
        elif self._last_tracking is not None:
            last_status = str(self._last_tracking.get("status", probe.status))
            health = self._normalize_status(last_status)
            details["status"] = "tracking_target" if health == "healthy" else last_status
        else:
            health = self._normalize_status(probe.status)
            details["status"] = "tracking_ready"
        return SubfunctionHealth(name="tracking", health=health, details=details)

    def _driver_kind(self, name: str) -> str:
        config = self.config.subfunctions.get(name)
        if config is None:
            return "noop"
        return str(config.driver.kind)

    @staticmethod
    def _merge_probe_details(probe: dict[str, object]) -> dict[str, object]:
        merged = dict(probe)
        merged["driver"] = merged.get("driver", "command")
        nested = merged.get("details", {})
        if not isinstance(nested, dict):
            nested = {}
        merged["details"] = nested
        return merged
