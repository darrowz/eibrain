"""Episodic memory store."""

from __future__ import annotations


class EpisodicMemoryStore:
    """Phase 4 in-memory episodic memory store."""

    def __init__(self) -> None:
        self._episodes: dict[str, str] = {}

    def remember_episode(self, *, session_id: str, summary: str) -> None:
        self._episodes[session_id] = summary

    def summarize_session(self, session_id: str) -> str:
        return self._episodes.get(session_id, "")
