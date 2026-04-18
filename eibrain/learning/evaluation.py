"""Evaluation engine."""

from __future__ import annotations


class EvaluationEngine:
    """Phase 5 evaluation helper."""

    def score_review(self, review_item: dict[str, str]) -> float:
        base = 1.0 if review_item.get("reply") else 0.2
        if review_item.get("outcome", "").endswith("completed"):
            base += 0.5
        return base
