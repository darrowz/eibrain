"""Semantic memory store."""

from __future__ import annotations


class SemanticMemoryStore:
    """Phase 4 in-memory semantic memory store."""

    def __init__(self) -> None:
        self._profiles: dict[str, dict[str, str]] = {}

    def remember_profile(self, *, actor_id: str, profile: dict[str, str]) -> None:
        self._profiles[actor_id] = dict(profile)

    def load_actor_profile(self, actor_id: str) -> dict[str, str] | None:
        profile = self._profiles.get(actor_id)
        return dict(profile) if profile is not None else None
