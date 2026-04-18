"""Adaptation engine."""

from __future__ import annotations


class AdaptationEngine:
    """Phase 5 adaptation helper."""

    def decide(self, score: float) -> str:
        return "keep_policy" if score >= 1.0 else "adjust_policy"
