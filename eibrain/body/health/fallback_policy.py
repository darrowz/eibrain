"""Fallback policy placeholder."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FallbackPolicy:
    mode: str = "normal"
