"""OpenClaw memory adapter boundary."""

from __future__ import annotations

import json
from urllib import parse, request

from eibrain.infra.config import OpenClawConfig
from eibrain.memory.contracts import MemoryQuery, MemoryResult


class OpenClawMemoryAdapter:
    def __init__(self, config: OpenClawConfig | None = None) -> None:
        self.config = config or OpenClawConfig()
        self._profiles: dict[str, dict[str, str]] = {}
        self._sessions: dict[str, str] = {}

    def retrieve_context(self, query: MemoryQuery) -> MemoryResult:
        if self.config.provider == "http_json" and self.config.endpoint:
            return self._retrieve_via_http(query)
        session_summary = self.summarize_session(query.session_id) if query.session_id else ""
        actor_profile = self.load_actor_profile(query.actor_id) if query.actor_id else None
        summary_parts = [f"openclaw-context:{query.query}"]
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
        if self.config.provider == "http_json" and self.config.endpoint:
            self._post_json("/remember_episode", {"session_id": session_id, "summary": summary})
            return
        self._sessions[session_id] = summary

    def remember_preference(self, *, actor_id: str, profile: dict[str, str]) -> None:
        if self.config.provider == "http_json" and self.config.endpoint:
            self._post_json("/remember_preference", {"actor_id": actor_id, "profile": profile})
            return
        self._profiles[actor_id] = dict(profile)

    def load_actor_profile(self, actor_id: str | None) -> dict[str, str] | None:
        if actor_id is None:
            return None
        if self.config.provider == "http_json" and self.config.endpoint:
            payload = self._get_json("/actor_profile", {"actor_id": actor_id})
            profile = payload.get("profile")
            return dict(profile) if isinstance(profile, dict) else None
        profile = self._profiles.get(actor_id)
        return dict(profile) if profile is not None else None

    def summarize_session(self, session_id: str | None) -> str:
        if session_id is None:
            return ""
        if self.config.provider == "http_json" and self.config.endpoint:
            payload = self._get_json("/session_summary", {"session_id": session_id})
            return str(payload.get("summary", ""))
        return self._sessions.get(session_id, "")

    def _retrieve_via_http(self, query: MemoryQuery) -> MemoryResult:
        payload = self._post_json(
            "/retrieve_context",
            {"query": query.query, "session_id": query.session_id, "actor_id": query.actor_id},
        )
        return MemoryResult(
            summary=str(payload.get("summary", "")),
            relevant_memories=[str(item) for item in payload.get("relevant_memories", [])],
            actor_profile={
                str(key): str(value)
                for key, value in dict(payload.get("actor_profile", {})).items()
            },
            session_summary=str(payload.get("session_summary", "")),
        )

    def _post_json(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        req = request.Request(
            self._url(path),
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers=self._headers(),
        )
        with request.urlopen(req, timeout=self.config.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def _get_json(self, path: str, params: dict[str, str]) -> dict[str, object]:
        query = parse.urlencode(params)
        req = request.Request(f"{self._url(path)}?{query}", method="GET", headers=self._headers())
        with request.urlopen(req, timeout=self.config.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def _url(self, path: str) -> str:
        return f"{self.config.endpoint.rstrip('/')}{path}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers
