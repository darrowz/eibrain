"""eimemory RPC adapter."""

from __future__ import annotations

import json
from urllib import request
from urllib.error import URLError

from eibrain.infra.config import OpenClawConfig
from eibrain.memory.contracts import MemoryQuery, MemoryResult


class EIMemoryRPCAdapter:
    def __init__(self, config: OpenClawConfig | None = None) -> None:
        self.config = config or OpenClawConfig(provider="eimemory_rpc")
        self._profiles: dict[str, dict[str, str]] = {}
        self._sessions: dict[str, str] = {}

    def retrieve_context(self, query: MemoryQuery) -> MemoryResult:
        payload = {
            "method": "memory.recall",
            "params": {
                "query": query.query,
                "limit": 8,
                "scope": self._scope(query),
                "task_context": {
                    "task_type": "brain.respond",
                    "goal": "retrieve memory for embodied response",
                },
            },
        }
        try:
            result = self._post_json(payload)
            return self._map_result(result)
        except (URLError, OSError, ValueError, TypeError, KeyError):
            return self._fallback_result(query)

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
    ) -> None:
        cleaned_summary = str(summary or "").strip()
        if not cleaned_summary:
            return
        self._sessions[session_id] = cleaned_summary
        if not self.config.endpoint:
            return
        payload = {
            "method": "memory.ingest",
            "params": {
                "text": cleaned_summary,
                "title": title or "Embodied episode",
                "memory_type": memory_type,
                "source": source,
                "scope": self._scope_from_ids(session_id=session_id, actor_id=actor_id),
                "organ": organ,
                "modality": modality,
            },
        }
        try:
            self._post_json(payload)
        except (URLError, OSError, ValueError, TypeError, KeyError):
            return

    def remember_preference(
        self,
        *,
        actor_id: str,
        profile: dict[str, str],
        title: str = "",
        source: str = "eibrain.preference",
        modality: str = "text",
        organ: str = "cognition",
    ) -> None:
        normalized_profile = {str(key): str(value) for key, value in dict(profile or {}).items()}
        self._profiles[actor_id] = dict(normalized_profile)
        if not normalized_profile or not self.config.endpoint:
            return
        profile_summary = ", ".join(
            f"{key}={value}"
            for key, value in sorted(normalized_profile.items())
            if key.strip() and value.strip()
        )
        if not profile_summary:
            return
        payload = {
            "method": "memory.ingest",
            "params": {
                "text": f"Preference profile for {actor_id}: {profile_summary}",
                "title": title or f"Preference profile: {actor_id}",
                "memory_type": "preference",
                "source": source,
                "scope": self._scope_from_ids(actor_id=actor_id),
                "organ": organ,
                "modality": modality,
            },
        }
        try:
            self._post_json(payload)
        except (URLError, OSError, ValueError, TypeError, KeyError):
            return

    def load_actor_profile(self, actor_id: str | None) -> dict[str, str] | None:
        if actor_id is None:
            return None
        profile = self._profiles.get(actor_id)
        return dict(profile) if profile is not None else None

    def summarize_session(self, session_id: str | None) -> str:
        if session_id is None:
            return ""
        return self._sessions.get(session_id, "")

    def _scope(self, query: MemoryQuery) -> dict[str, str]:
        return self._scope_from_ids(session_id=query.session_id, actor_id=query.actor_id)

    def _scope_from_ids(self, *, session_id: str | None = None, actor_id: str | None = None) -> dict[str, str]:
        scope: dict[str, str] = {}
        if self.config.agent_id:
            scope["agent_id"] = self.config.agent_id
        if self.config.workspace_id:
            scope["workspace_id"] = self.config.workspace_id
        if session_id:
            scope["session_id"] = session_id
        if actor_id:
            scope["actor_id"] = actor_id
        return scope

    def _post_json(self, payload: dict[str, object]) -> dict[str, object]:
        if not self.config.endpoint:
            raise ValueError("eimemory RPC endpoint is not configured")
        req = request.Request(
            self.config.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers=self._headers(),
        )
        with request.urlopen(req, timeout=self.config.timeout_s) as response:
            result = json.loads(response.read().decode("utf-8"))
        if not isinstance(result, dict) or result.get("ok") is False:
            raise ValueError("eimemory RPC request failed")
        return result

    def _map_result(self, payload: dict[str, object]) -> MemoryResult:
        result = dict(payload.get("result", {}))
        items = [dict(item) for item in result.get("items", [])]
        rules = [dict(rule) for rule in result.get("rules", [])]
        explanation = dict(result.get("explanation", {}))
        relevant_memories = []
        for item in items:
            title = str(item.get("title", "")).strip()
            summary = str(item.get("summary", "")).strip() or str(dict(item.get("content", {})).get("text", "")).strip()
            if title and summary:
                relevant_memories.append(f"{title}: {summary}")
            elif summary:
                relevant_memories.append(summary)
            elif title:
                relevant_memories.append(title)
        summary_parts = [str(item.get("summary", "")).strip() for item in items[:2]]
        rule_hint = str(rules[0].get("title", "")).strip() if rules else ""
        if rule_hint:
            summary_parts.append(rule_hint)
        return MemoryResult(
            summary=" | ".join(part for part in summary_parts if part),
            relevant_memories=relevant_memories,
            actor_profile={},
            session_summary=str(explanation.get("session_summary", "")),
        )

    def _fallback_result(self, query: MemoryQuery) -> MemoryResult:
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

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers
