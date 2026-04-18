"""Engagement state models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EngagementState:
    phase: str = "idle"
