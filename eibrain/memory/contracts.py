"""Memory contracts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class MemoryQuery:
    query: str
    session_id: str | None = None
    actor_id: str | None = None


@dataclass(slots=True)
class MemoryResult:
    summary: str = ""
    relevant_memories: list[str] = field(default_factory=list)
    actor_profile: dict[str, str] = field(default_factory=dict)
    session_summary: str = ""
