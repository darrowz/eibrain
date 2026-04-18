"""Minimal Raspbot servo driver for honjia gimbal control."""

from __future__ import annotations

import os


class RaspbotDriver:
    def __init__(self, *, bus: int = 1, addr: int = 0x2B, servo_id: int = 1, enabled: bool = True, mock: bool = False) -> None:
        self.bus = bus
        self.addr = addr
        self.servo_id = servo_id
        self.enabled = enabled
        self.mock = mock
        self.device_path = f"/dev/i2c-{self.bus}"
        self.last_command: tuple[int, int] | None = None

    def ctrl_servo(self, angle: int, servo_id: int | None = None) -> list[int]:
        target_servo = self.servo_id if servo_id is None else servo_id
        self.last_command = (target_servo, angle)
        payload = [0xFF, self.addr & 0xFF, target_servo & 0xFF, angle & 0xFF]
        if self.mock or not self.enabled:
            return payload
        if not os.path.exists(self.device_path):
            raise RuntimeError(f"missing i2c device: {self.device_path}")
        try:
            import smbus2  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on honjia packages
            raise RuntimeError(f"smbus2 unavailable: {exc}") from exc
        bus = smbus2.SMBus(self.bus)
        try:
            bus.write_i2c_block_data(self.addr, 0x06 + target_servo * 4, [0, 0, angle & 0xFF, 0])
        finally:
            bus.close()
        return payload
