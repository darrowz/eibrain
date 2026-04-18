"""Self state models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SelfState:
    mode: str = "idle"
