"""eimemory RPC adapter."""

from __future__ import annotations

import json
from typing import Mapping
from urllib import request
from urllib.error import URLError

from eibrain.infra.config import OpenClawConfig
from eibrain.memory.contracts import MemoryQuery, MemoryResult
from eibrain.memory.scoring_compat import merge_memory_metadata, normalize_memory_metadata, score_meta_from_recall_entry
from eibrain.memory.subject import classify_memory_layer, normalize_hongtu_scope, subject_context


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
        if query.session_id:
            task_context["session_id"] = query.session_id
        self._attach_subject_context(
            task_context,
            actor_id=query.actor_id,
            session_id=query.session_id,
            source=self._clean_text(task_context.get("source")),
            memory_type=self._clean_text(task_context.get("memory_type")),
        )
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
        idempotency_key: str | None = None,
        source_event_id: str | None = None,
        conflict: dict[str, object] | None = None,
        persona_snapshot: dict[str, object] | None = None,
    ) -> None:
        cleaned_summary = str(summary or "").strip()
        if not cleaned_summary:
            return
        self._sessions[session_id] = cleaned_summary
        normalized_meta = normalize_memory_metadata(meta)
        governance = self._governance_fields(
            meta=normalized_meta,
            idempotency_key=idempotency_key,
            source_event_id=source_event_id,
            conflict=conflict,
            persona_snapshot=persona_snapshot,
        )
        request_meta = self._meta_with_session(
            meta=normalized_meta,
            session_id=session_id,
            actor_id=actor_id,
            source=source,
            memory_type=memory_type,
        )
        status_meta = dict(request_meta)
        status_meta.update(governance)
        if not self.config.endpoint:
            self.last_writeback_status = self._writeback_status(
                status="error",
                source=source,
                memory_type=memory_type,
                modality=modality,
                organ=organ,
                title=title or "Embodied episode",
                meta=status_meta,
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
                "scope": self._scope_from_ids(actor_id=actor_id),
                "organ": organ,
                "modality": modality,
                "outcome": dict(outcome or {}),
            },
        }
        params = payload["params"]
        if content is not None:
            params["content"] = dict(content)
        if request_meta:
            params["meta"] = dict(request_meta)
        params.update(governance)
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
                meta=status_meta,
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
                meta=status_meta,
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
        idempotency_key: str | None = None,
        source_event_id: str | None = None,
        conflict: dict[str, object] | None = None,
        persona_snapshot: dict[str, object] | None = None,
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
            idempotency_key=idempotency_key,
            source_event_id=source_event_id,
            conflict=conflict,
            persona_snapshot=persona_snapshot,
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
        request_meta = self._meta_with_session(
            meta=None,
            session_id=None,
            actor_id=actor_id,
            source=source,
            memory_type="preference",
        )
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
                "meta": request_meta,
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
                "scope": self._scope_from_ids(actor_id=actor_id),
            },
        }
        meta = self._meta_with_session(
            meta=None,
            session_id=session_id,
            actor_id=actor_id,
            source=self._clean_text((payload or {}).get("source")),
            memory_type=self._clean_text((payload or {}).get("memory_type")),
        )
        if meta:
            body["params"]["meta"] = meta
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
                "scope": self._scope_from_ids(actor_id=actor_id),
            },
        }
        meta = self._meta_with_session(
            meta=None,
            session_id=session_id,
            actor_id=actor_id,
            source=self._clean_text((payload or {}).get("source")),
            memory_type=self._clean_text((payload or {}).get("memory_type")),
        )
        if meta:
            body["params"]["meta"] = meta
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
                "scope": self._scope_from_ids(actor_id=actor_id),
            },
        }
        meta = self._meta_with_session(
            meta=None,
            session_id=session_id,
            actor_id=actor_id,
            source=self._clean_text((payload or {}).get("source")) or "eibrain.skill_trace",
            memory_type="trace",
        )
        if meta:
            body["params"]["meta"] = meta
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
        trace_payload = dict(payload or {})
        body = {
            "method": "experience.record_memory_trace",
            "params": {
                "payload": trace_payload,
                "scope": self._scope_from_ids(actor_id=actor_id),
            },
        }
        trace_meta = {
            key: trace_payload[key]
            for key in ("source", "source_channel", "channel_id", "subject_context", "memory_layer", "memory_type")
            if key in trace_payload
        }
        meta = self._meta_with_session(
            meta=trace_meta,
            session_id=session_id,
            actor_id=actor_id,
            source=self._clean_text(trace_payload.get("source")) or self._clean_text(trace_payload.get("schema")) or "eibrain.memory_trace",
            memory_type=self._clean_text(trace_payload.get("memory_type")) or "trace",
        )
        if meta:
            body["params"]["meta"] = meta
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
                "scope": self._scope_from_ids(actor_id=actor_id),
            },
        }
        meta = self._meta_with_session(meta=None, session_id=session_id, actor_id=actor_id)
        if meta:
            body["params"]["meta"] = meta
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
            "idempotency_key",
            "conflict",
            "conflict_resolution",
            "persona_snapshot",
            "candidate_types",
            "identity_memory",
            "persona_memory",
            "retention",
            "promotion_status",
            "training_candidate",
            "subject_context",
            "memory_layer",
            "source_channel",
            "raw_actor_id",
            "actor_alias",
        ):
            if key in metadata:
                payload[key] = metadata[key]
        return payload

    def _governance_fields(
        self,
        *,
        meta: dict[str, object] | None,
        idempotency_key: str | None,
        source_event_id: str | None,
        conflict: dict[str, object] | None,
        persona_snapshot: dict[str, object] | None,
    ) -> dict[str, object]:
        metadata = dict(meta or {})
        fields: dict[str, object] = {}
        resolved_idempotency_key = idempotency_key or metadata.get("idempotency_key")
        if resolved_idempotency_key:
            fields["idempotency_key"] = str(resolved_idempotency_key)
        resolved_source_event_id = source_event_id or metadata.get("source_event_id")
        if resolved_source_event_id:
            fields["source_event_id"] = str(resolved_source_event_id)
        resolved_conflict = conflict or metadata.get("conflict") or metadata.get("conflict_resolution")
        if isinstance(resolved_conflict, dict):
            fields["conflict"] = dict(resolved_conflict)
        resolved_persona_snapshot = persona_snapshot or metadata.get("persona_snapshot")
        if isinstance(resolved_persona_snapshot, dict):
            fields["persona_snapshot"] = dict(resolved_persona_snapshot)
        return fields

    def _scope(self, query: MemoryQuery) -> dict[str, str]:
        return self._scope_from_ids(actor_id=query.actor_id)

    def _scope_from_ids(self, *, session_id: str | None = None, actor_id: str | None = None) -> dict[str, str]:
        del session_id
        return normalize_hongtu_scope(
            {
                "tenant_id": self.config.tenant_id or "default",
                "agent_id": self.config.agent_id,
                "workspace_id": self.config.workspace_id,
                "user_id": actor_id or "",
            }
        )

    def _meta_with_session(
        self,
        *,
        meta: dict[str, object] | None,
        session_id: str | None,
        actor_id: str | None = None,
        source: str | None = None,
        memory_type: str | None = None,
    ) -> dict[str, object]:
        request_meta = dict(meta or {})
        if session_id:
            request_meta["session_id"] = session_id
        resolved_source = self._clean_text(source or request_meta.get("source"))
        resolved_memory_type = self._clean_text(memory_type or request_meta.get("memory_type"))
        channel_id = self._channel_id(context=request_meta, source=resolved_source)
        memory_layer = self._memory_layer(
            source=resolved_source,
            memory_type=resolved_memory_type,
            context=request_meta,
        )
        request_meta["subject_context"] = self._subject_context(
            context=request_meta,
            channel_id=channel_id,
            actor_id=actor_id,
            session_id=session_id,
            source=resolved_source,
            memory_type=resolved_memory_type,
        )
        request_meta["source_channel"] = channel_id
        if memory_layer:
            request_meta["memory_layer"] = memory_layer
        if actor_alias := self._clean_text(actor_id):
            request_meta["raw_actor_id"] = actor_alias
            request_meta["actor_alias"] = actor_alias
        return request_meta

    def _attach_subject_context(
        self,
        task_context: dict[str, object],
        *,
        actor_id: str | None,
        session_id: str | None,
        source: str | None,
        memory_type: str | None,
    ) -> None:
        channel_id = self._channel_id(context=task_context, source=source)
        memory_layer = self._memory_layer(source=source, memory_type=memory_type, context=task_context)
        task_context.setdefault("channel_id", channel_id)
        task_context["subject_context"] = self._subject_context(
            context=task_context,
            channel_id=channel_id,
            actor_id=actor_id,
            session_id=session_id,
            source=source,
            memory_type=memory_type,
        )
        if memory_layer:
            task_context["memory_layer"] = memory_layer

    def _subject_context(
        self,
        *,
        context: dict[str, object] | None,
        channel_id: str,
        actor_id: str | None,
        session_id: str | None,
        source: str | None,
        memory_type: str | None,
    ) -> dict[str, object]:
        existing = (context or {}).get("subject_context")
        merged = dict(existing) if isinstance(existing, dict) else {}
        resolved_source = self._clean_text(source)
        merged.update(
            subject_context(
                channel_id=channel_id,
                actor_id=actor_id,
                session_id=session_id,
                source=resolved_source or None,
            )
        )
        memory_layer = self._memory_layer(
            source=resolved_source,
            memory_type=memory_type,
            context=context,
        )
        if memory_layer:
            merged["memory_layer"] = memory_layer
        return merged

    def _memory_layer(
        self,
        *,
        source: str | None,
        memory_type: str | None,
        context: dict[str, object] | None = None,
    ) -> str:
        existing = self._clean_text((context or {}).get("memory_layer"))
        if existing:
            return existing
        resolved_source = self._clean_text(source)
        resolved_memory_type = self._clean_text(memory_type)
        if not resolved_source and not resolved_memory_type:
            return ""
        return classify_memory_layer(resolved_source or None, resolved_memory_type or None)

    def _channel_id(self, *, context: dict[str, object] | None = None, source: str | None = None) -> str:
        payload = dict(context or {})
        for key in ("channel_id", "source_channel"):
            channel_id = self._clean_text(payload.get(key))
            if channel_id:
                return channel_id
        subject = payload.get("subject_context")
        if isinstance(subject, dict):
            channel_id = self._clean_text(subject.get("channel_id"))
            if channel_id:
                return channel_id
        source_id = self._clean_text(source or payload.get("source")).lower()
        if "feishu" in source_id:
            return "feishu"
        if "visual" in source_id or "vision" in source_id:
            return "vision"
        if source_id.startswith("openclaw."):
            return "openclaw"
        if self._clean_text(self.config.workspace_id) == "honjia":
            return "voice.honjia"
        return "voice"

    @staticmethod
    def _clean_text(value: object) -> str:
        if value is None:
            return ""
        return str(value).strip()

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
        scored_records = self._selected_records_with_scoring(items=items, explanation=explanation)
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
                "quality_summary": explanation.get("quality_summary", {}),
                "source_composition": explanation.get("source_composition", {}),
                "selected_records": scored_records,
                "scoring": self._scoring_rows(explanation),
                "recall_filters": explanation.get("recall_filters", {}),
            },
        )

    def _selected_records_with_scoring(
        self,
        *,
        items: list[dict[str, object]],
        explanation: Mapping[str, object],
    ) -> list[dict[str, object]]:
        item_by_id = {
            str(item.get("record_id") or item.get("id") or "").strip(): item
            for item in items
            if str(item.get("record_id") or item.get("id") or "").strip()
        }
        scoring_by_id = {
            str(entry.get("record_id") or "").strip(): entry
            for entry in explanation.get("scoring", [])
            if isinstance(entry, dict) and str(entry.get("record_id") or "").strip()
        }
        selected_records: list[dict[str, object]] = []
        for record in explanation.get("selected_records", []):
            if not isinstance(record, dict):
                continue
            record_id = str(record.get("record_id") or record.get("id") or "").strip()
            item_meta = dict(item_by_id.get(record_id, {}).get("meta", {})) if record_id else {}
            record_meta = dict(record.get("meta", {})) if isinstance(record.get("meta"), dict) else {}
            upstream_meta = normalize_memory_metadata(merge_memory_metadata(item_meta, record_meta))
            merged_meta = upstream_meta
            if not _meta_has_memory_score(upstream_meta):
                scoring_meta = score_meta_from_recall_entry(scoring_by_id.get(record_id))
                merged_meta = normalize_memory_metadata(merge_memory_metadata(scoring_meta, item_meta, record_meta))
            payload = dict(record)
            if merged_meta:
                payload["meta"] = merged_meta
            selected_records.append(payload)
        return selected_records

    def _scoring_rows(self, explanation: Mapping[str, object]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for entry in explanation.get("scoring", []):
            if not isinstance(entry, dict):
                continue
            payload = dict(entry)
            compat_meta = score_meta_from_recall_entry(payload)
            if compat_meta:
                score = dict(dict(compat_meta.get("scoring", {})).get("memory_score_v1", {}))
                quality = dict(compat_meta.get("quality", {}))
                if score and not isinstance(payload.get("memory_score_v1"), Mapping):
                    payload["memory_score_v1"] = score
                if quality and not isinstance(payload.get("quality"), Mapping):
                    payload["quality"] = quality
            rows.append(payload)
        return rows

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


def _meta_has_memory_score(meta: Mapping[str, object] | None) -> bool:
    payload = dict(meta) if isinstance(meta, Mapping) else {}
    scoring = payload.get("scoring")
    if not isinstance(scoring, Mapping):
        return False
    return isinstance(scoring.get("memory_score_v1"), Mapping)


class FakeEIMemoryRPCAdapter:
    """In-memory test double for closed-loop memory tests."""

    def __init__(self, recall_result: MemoryResult | None = None) -> None:
        self.recall_result = recall_result or MemoryResult()
        self.queries: list[MemoryQuery] = []
        self.episodes: list[dict[str, object]] = []
        self.world_observations: list[dict[str, object]] = []
        self.memory_traces: list[dict[str, object]] = []
        self.last_recall_diagnostics: dict[str, object] = dict(self.recall_result.recall_diagnostics)
        self.last_writeback_status: dict[str, object] = {"status": "idle"}

    def retrieve_context(self, query: MemoryQuery) -> MemoryResult:
        self.queries.append(query)
        self.last_recall_diagnostics = dict(self.recall_result.recall_diagnostics)
        return self.recall_result

    def remember_episode(self, **kwargs: object) -> dict[str, object]:
        payload = dict(kwargs)
        self.episodes.append(payload)
        record_id = f"episode_{len(self.episodes)}"
        self.last_writeback_status = {
            "status": "ok",
            "source": payload.get("source"),
            "memory_type": payload.get("memory_type"),
            "modality": payload.get("modality"),
            "organ": payload.get("organ"),
            "record_id": record_id,
        }
        return {"record_id": record_id}

    def remember_world_observation(self, **kwargs: object) -> dict[str, object]:
        payload = dict(kwargs)
        self.world_observations.append(payload)
        record_id = f"world_{len(self.world_observations)}"
        self.last_writeback_status = {
            "status": "ok",
            "source": "eibrain.visual_world",
            "memory_type": "world_observation",
            "modality": "vision",
            "organ": "eye",
            "record_id": record_id,
        }
        return {"record_id": record_id}

    def record_memory_trace(
        self,
        payload: dict[str, object],
        *,
        session_id: str | None = None,
        actor_id: str | None = None,
    ) -> dict[str, object]:
        self.memory_traces.append(
            {
                "payload": dict(payload),
                "session_id": session_id,
                "actor_id": actor_id,
            }
        )
        return {"ok": True, "result": {"record_id": f"memory_trace_{len(self.memory_traces)}"}}
