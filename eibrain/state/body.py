"""Body state models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class BodyState:
    organs: dict[str, dict[str, str]] = field(default_factory=dict)
