"""eimemory RPC adapter."""

from __future__ import annotations

import json
from urllib import request

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
        result = self._post_json(payload)
        return self._map_result(result)

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

    def _scope(self, query: MemoryQuery) -> dict[str, str]:
        scope: dict[str, str] = {}
        if self.config.agent_id:
            scope["agent_id"] = self.config.agent_id
        if self.config.workspace_id:
            scope["workspace_id"] = self.config.workspace_id
        if query.session_id:
            scope["session_id"] = query.session_id
        if query.actor_id:
            scope["actor_id"] = query.actor_id
        return scope

    def _post_json(self, payload: dict[str, object]) -> dict[str, object]:
        req = request.Request(
            self.config.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers=self._headers(),
        )
        with request.urlopen(req, timeout=self.config.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

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

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers
