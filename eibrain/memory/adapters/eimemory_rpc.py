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
        self.last_recall_diagnostics: dict[str, object] = {}
        self.last_writeback_status: dict[str, object] = {}

    def retrieve_context(self, query: MemoryQuery) -> MemoryResult:
        task_context = {
            "task_type": "brain.respond",
            "goal": "retrieve memory for embodied response",
        }
        task_context.update(dict(query.task_context or {}))
        payload = {
            "method": "memory.recall",
            "params": {
                "query": query.query,
                "limit": 8,
                "scope": self._scope(query),
                "task_context": task_context,
            },
        }
        try:
            result = self._post_json(payload)
            memory_result = self._map_result(result)
            self.last_recall_diagnostics = dict(memory_result.recall_diagnostics)
            return memory_result
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
        outcome: dict[str, object] | None = None,
        content: dict[str, object] | None = None,
        meta: dict[str, object] | None = None,
        tags: list[str] | None = None,
        evidence: list[dict[str, object]] | None = None,
        links: list[dict[str, object]] | None = None,
    ) -> None:
        cleaned_summary = str(summary or "").strip()
        if not cleaned_summary:
            return
        self._sessions[session_id] = cleaned_summary
        if not self.config.endpoint:
            self.last_writeback_status = self._writeback_status(
                status="error",
                source=source,
                memory_type=memory_type,
                modality=modality,
                organ=organ,
                title=title or "Embodied episode",
                meta=meta,
                error="eimemory RPC endpoint is not configured",
            )
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
                "outcome": dict(outcome or {}),
            },
        }
        params = payload["params"]
        if content is not None:
            params["content"] = dict(content)
        if meta is not None:
            params["meta"] = dict(meta)
        if tags is not None:
            params["tags"] = [str(tag) for tag in tags]
        if evidence is not None:
            params["evidence"] = [dict(item) for item in evidence]
        if links is not None:
            params["links"] = [dict(item) for item in links]
        try:
            response = self._post_json(payload)
            self.last_writeback_status = self._writeback_status(
                status="ok",
                source=source,
                memory_type=memory_type,
                modality=modality,
                organ=organ,
                title=title or "Embodied episode",
                meta=meta,
                response=response,
            )
            self._observe_failed_outcome(
                outcome=outcome,
                session_id=session_id,
                actor_id=actor_id,
                source=source,
                modality=modality,
                organ=organ,
            )
        except (URLError, OSError, ValueError, TypeError, KeyError) as exc:
            self.last_writeback_status = self._writeback_status(
                status="error",
                source=source,
                memory_type=memory_type,
                modality=modality,
                organ=organ,
                title=title or "Embodied episode",
                meta=meta,
                error=f"{type(exc).__name__}: {exc}",
            )
            return

    def remember_world_observation(
        self,
        *,
        session_id: str,
        summary: str,
        actor_id: str | None = None,
        content: dict[str, object] | None = None,
        meta: dict[str, object] | None = None,
        tags: list[str] | None = None,
        evidence: list[dict[str, object]] | None = None,
        links: list[dict[str, object]] | None = None,
        title: str = "Visual world observation",
    ) -> None:
        observation_tags = self._merge_tags(tags, ["world_observation", "vision"])
        self.remember_episode(
            session_id=session_id,
            actor_id=actor_id,
            summary=summary,
            title=title,
            memory_type="world_observation",
            source="eibrain.visual_world",
            modality="vision",
            organ="eye",
            content=content,
            meta=meta,
            tags=observation_tags,
            evidence=evidence,
            links=links,
        )

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

    def observe_outcome(
        self,
        *,
        signal_type: str,
        payload: dict[str, object],
        session_id: str | None = None,
        actor_id: str | None = None,
    ) -> None:
        if not self.config.endpoint:
            return
        body = {
            "method": "evolution.observe",
            "params": {
                "signal_type": signal_type,
                "payload": dict(payload or {}),
                "scope": self._scope_from_ids(session_id=session_id, actor_id=actor_id),
            },
        }
        try:
            self._post_json(body)
        except (URLError, OSError, ValueError, TypeError, KeyError):
            return

    def observe(
        self,
        signal_type: str,
        payload: dict[str, object],
        *,
        session_id: str | None = None,
        actor_id: str | None = None,
    ) -> dict[str, object]:
        if not self.config.endpoint:
            return {}
        body = {
            "method": "evolution.observe",
            "params": {
                "signal_type": signal_type,
                "payload": dict(payload or {}),
                "scope": self._scope_from_ids(session_id=session_id, actor_id=actor_id),
            },
        }
        try:
            return self._post_json(body)
        except (URLError, OSError, ValueError, TypeError, KeyError):
            return {}

    def record_skill_trace(
        self,
        payload: dict[str, object],
        *,
        session_id: str | None = None,
        actor_id: str | None = None,
    ) -> dict[str, object]:
        if not self.config.endpoint:
            return {}
        body = {
            "method": "experience.record_skill_trace",
            "params": {
                "payload": dict(payload or {}),
                "scope": self._scope_from_ids(session_id=session_id, actor_id=actor_id),
            },
        }
        try:
            return self._post_json(body)
        except (URLError, OSError, ValueError, TypeError, KeyError):
            return {}

    def record_memory_trace(
        self,
        payload: dict[str, object],
        *,
        session_id: str | None = None,
        actor_id: str | None = None,
    ) -> dict[str, object]:
        if not self.config.endpoint:
            return {}
        body = {
            "method": "experience.record_memory_trace",
            "params": {
                "payload": dict(payload or {}),
                "scope": self._scope_from_ids(session_id=session_id, actor_id=actor_id),
            },
        }
        try:
            return self._post_json(body)
        except (URLError, OSError, ValueError, TypeError, KeyError):
            return {}

    def get_active_policy(
        self,
        *,
        task_type: str = "brain.respond",
        session_id: str | None = None,
        actor_id: str | None = None,
    ) -> dict[str, object]:
        if not self.config.endpoint:
            return {}
        body = {
            "method": "evolution.get_active_policy",
            "params": {
                "task_type": task_type,
                "scope": self._scope_from_ids(session_id=session_id, actor_id=actor_id),
            },
        }
        try:
            result = self._post_json(body)
        except (URLError, OSError, ValueError, TypeError, KeyError):
            return {}
        policy = result.get("result", {})
        return dict(policy) if isinstance(policy, dict) else {}

    def _observe_failed_outcome(
        self,
        *,
        outcome: dict[str, object] | None,
        session_id: str,
        actor_id: str | None,
        source: str,
        modality: str,
        organ: str,
    ) -> None:
        payload = dict(outcome or {})
        status = str(payload.get("status") or "").lower()
        failed = payload.get("success") is False or status in {"error", "failed"}
        if not failed:
            return
        payload.update({"source": source, "modality": modality, "organ": organ})
        self.observe_outcome(
            signal_type="incident",
            payload=payload,
            session_id=session_id,
            actor_id=actor_id,
        )

    def _writeback_status(
        self,
        *,
        status: str,
        source: str,
        memory_type: str,
        modality: str,
        organ: str,
        title: str,
        meta: dict[str, object] | None,
        response: dict[str, object] | None = None,
        error: str = "",
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": status,
            "source": source,
            "memory_type": memory_type,
            "modality": modality,
            "organ": organ,
            "title": title,
        }
        if error:
            payload["error"] = error
        result = dict((response or {}).get("result", {}) or {})
        record_id = result.get("record_id") or result.get("id")
        if record_id:
            payload["record_id"] = str(record_id)
        metadata = dict(meta or {})
        for key in (
            "trace_id",
            "source_event_id",
            "candidate_types",
            "identity_memory",
            "persona_memory",
            "retention",
            "promotion_status",
            "training_candidate",
        ):
            if key in metadata:
                payload[key] = metadata[key]
        return payload

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

    def _merge_tags(self, tags: list[str] | None, required_tags: list[str]) -> list[str]:
        merged: list[str] = []
        for tag in [*(tags or []), *required_tags]:
            cleaned = str(tag or "").strip()
            if cleaned and cleaned not in merged:
                merged.append(cleaned)
        return merged

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
            recall_diagnostics={
                "query": explanation.get("query", ""),
                "task_context": explanation.get("task_context", {}),
                "recall_profile": explanation.get("recall_profile", ""),
                "selected_count": explanation.get("selected_count", 0),
                "source_composition": explanation.get("source_composition", {}),
                "selected_records": explanation.get("selected_records", []),
                "recall_filters": explanation.get("recall_filters", {}),
            },
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
