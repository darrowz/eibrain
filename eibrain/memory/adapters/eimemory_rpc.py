"""eimemory RPC adapter placeholder."""

from __future__ import annotations

from eibrain.infra.config import OpenClawConfig
from eibrain.memory.contracts import MemoryQuery, MemoryResult


class EIMemoryRPCAdapter:
    def __init__(self, config: OpenClawConfig | None = None) -> None:
        self.config = config or OpenClawConfig(provider="eimemory_rpc")
        self._profiles: dict[str, dict[str, str]] = {}
        self._sessions: dict[str, str] = {}

    def retrieve_context(self, query: MemoryQuery) -> MemoryResult:
        session_summary = self.summarize_session(query.session_id) if query.session_id else ""
        actor_profile = self.load_actor_profile(query.actor_id) if query.actor_id else None
        summary_parts = [f"eimemory-context:{query.query}"]
        if session_summary:
            summary_parts.append(f"session:{session_summary}")
        if actor_profile:
            summary_parts.append(f"profile:{actor_profile}")
        return MemoryResult(
            summary=" | ".join(summary_parts),
            actor_profile=actor_profile or {},
            session_summary=session_summary,
        )

    def remember_episode(self, *, session_id: str, summary: str) -> None:
        self._sessions[session_id] = summary

    def remember_preference(self, *, actor_id: str, profile: dict[str, str]) -> None:
        self._profiles[actor_id] = dict(profile)

    def load_actor_profile(self, actor_id: str | None) -> dict[str, str] | None:
        if actor_id is None:
            return None
        profile = self._profiles.get(actor_id)
        return dict(profile) if profile is not None else None

    def summarize_session(self, session_id: str | None) -> str:
        if session_id is None:
            return ""
        return self._sessions.get(session_id, "")
