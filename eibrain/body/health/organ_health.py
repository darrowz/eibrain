"""Organ health models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class SubfunctionHealth:
    name: str
    health: str = "healthy"
    details: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class OrganHealth:
    organ: str
    health: str = "healthy"
    subfunctions: dict[str, SubfunctionHealth] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "organ": self.organ,
            "health": self.health,
            "subfunctions": {name: asdict(state) for name, state in self.subfunctions.items()},
        }
