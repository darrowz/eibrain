"""Working memory store."""

from __future__ import annotations

from collections import defaultdict


class WorkingMemoryStore:
    """Phase 2 in-memory working memory store."""

    def __init__(self) -> None:
        self._turns: dict[str, list[str]] = defaultdict(list)

    def remember_turn(self, *, session_id: str, text: str) -> None:
        self._turns[session_id].append(text)
        self._turns[session_id] = self._turns[session_id][-10:]

    def recent_turns(self, session_id: str) -> list[str]:
        return list(self._turns.get(session_id, []))
