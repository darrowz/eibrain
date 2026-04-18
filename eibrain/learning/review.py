"""Self-review engine."""

from __future__ import annotations


class SelfReviewEngine:
    """Phase 5 self-review helper."""

    def review_turn(self, *, event_type: str, transcript: str, reply: str, outcome: str) -> dict[str, str]:
        quality = "ok" if reply else "needs_attention"
        return {
            "event_type": event_type,
            "transcript": transcript,
            "reply": reply,
            "outcome": outcome,
            "quality": quality,
            "recommendation": "keep_policy" if quality == "ok" else "adjust_policy",
        }
