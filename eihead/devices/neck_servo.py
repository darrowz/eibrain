"""Servo command adapter for the honjia neck yaw axis.

The adapter owns no hardware policy.  It only translates an already-made yaw
decision into the tiny driver call expected by the current honjia servo driver.
"""

from __future__ import annotations

from typing import Any, Protocol


class ServoDriver(Protocol):
    def ctrl_servo(self, angle: int, servo_id: int | None = None) -> Any:
        """Move one servo and return the driver's payload/status."""


class NeckServoCommandAdapter:
    """Apply yaw decisions to an injected servo driver."""

    def __init__(self, driver: ServoDriver, *, servo_id: int = 1) -> None:
        self._driver = driver
        self.servo_id = int(servo_id)

    def apply_decision(self, decision: Any) -> dict[str, Any]:
        angle = int(decision.angle)
        if not bool(decision.should_command):
            return {
                "status": "suppressed",
                "reason": str(getattr(decision, "reason", "") or ""),
                "angle": angle,
            }

        payload = self._driver.ctrl_servo(angle, self.servo_id)
        return {
            "status": "ok",
            "servo_id": self.servo_id,
            "angle": angle,
            "payload": payload,
        }
