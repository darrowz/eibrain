"""Runtime lifecycle manager for in-process kernel components."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


LifecycleHook = Callable[[], object]
HealthHook = Callable[[], dict[str, object]]


@dataclass(slots=True)
class _LifecycleComponent:
    name: str
    start: LifecycleHook | None = None
    stop: LifecycleHook | None = None
    health: HealthHook | None = None
    state: str = "registered"
    error: str = ""

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"state": self.state}
        if self.error:
            payload["error"] = self.error
        if self.health is not None and self.state == "running":
            payload["health"] = dict(self.health())
        return payload


class LifecycleManager:
    """Start and stop runtime components in deterministic order."""

    def __init__(self) -> None:
        self._components: list[_LifecycleComponent] = []
        self._state = "created"

    def register(
        self,
        name: str,
        *,
        start: LifecycleHook | None = None,
        stop: LifecycleHook | None = None,
        health: HealthHook | None = None,
    ) -> None:
        if any(component.name == name for component in self._components):
            raise ValueError(f"component already registered: {name}")
        self._components.append(_LifecycleComponent(name=name, start=start, stop=stop, health=health))
        if self._state == "created":
            self._state = "registered"

    def start(self) -> None:
        self._state = "starting"
        for component in self._components:
            try:
                if component.start is not None:
                    component.start()
                component.state = "running"
                component.error = ""
            except Exception as exc:
                component.state = "failed"
                component.error = str(exc)
                self._state = "failed"
                raise
        self._state = "running"

    def stop(self) -> None:
        self._state = "stopping"
        for component in reversed(self._components):
            if component.state != "running":
                continue
            try:
                if component.stop is not None:
                    component.stop()
                component.state = "stopped"
                component.error = ""
            except Exception as exc:
                component.state = "failed"
                component.error = str(exc)
                self._state = "failed"
                raise
        if self._state != "failed":
            self._state = "stopped"

    def status(self) -> dict[str, object]:
        return {
            "state": self._state,
            "component_count": len(self._components),
            "components": {component.name: component.to_dict() for component in self._components},
        }
