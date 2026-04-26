"""Base memory adapter contract."""

from __future__ import annotations

from typing import Any, Protocol

from eibrain.memory.contracts import MemoryQuery, MemoryResult


class MemoryAdapter(Protocol):
    def retrieve_context(self, query: MemoryQuery) -> MemoryResult: ...

    def remember_episode(
        self,
        *,
        session_id: str,
        summary: str,
        actor_id: str | None = None,
        title: str = "",
        memory_type: str = "conversation",
        source: str = "eibrain.dialogue",
        modality: str = "text",
        organ: str = "cognition",
        outcome: dict[str, object] | None = None,
        content: dict[str, object] | None = None,
        meta: dict[str, object] | None = None,
        tags: list[str] | None = None,
        evidence: list[dict[str, object]] | None = None,
        links: list[dict[str, object]] | None = None,
    ) -> None: ...

    def remember_world_observation(
        self,
        *,
        session_id: str,
        summary: str,
        actor_id: str | None = None,
        content: dict[str, object] | None = None,
        meta: dict[str, object] | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.remember_episode(
            session_id=session_id,
            actor_id=actor_id,
            summary=summary,
            title=str(kwargs.get("title") or "Visual world observation"),
            memory_type="world_observation",
            source="eibrain.visual_world",
            modality="vision",
            organ="eye",
            content=content,
            meta=meta,
            tags=tags,
        )
