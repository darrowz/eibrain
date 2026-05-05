"""Inert memory proposal orchestration for realtime cognition."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Mapping


CLOSED_LOOP_TRACE_SCHEMA = "eibrain.memory.closed_loop_trace.v1"


def _json_ready(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_ready(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _value(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _memory_query(
    *,
    query: str,
    session_id: str | None,
    actor_id: str | None,
    task_context: Mapping[str, Any],
) -> Any:
    try:
        from eibrain.memory.contracts import MemoryQuery

        return MemoryQuery(
            query=query,
            session_id=session_id,
            actor_id=actor_id,
            task_context=task_context,
        )
    except ModuleNotFoundError:
        return SimpleNamespace(
            query=query,
            session_id=session_id,
            actor_id=actor_id,
            task_context=task_context,
        )


class MemoryOrchestrator:
    """Build round-scoped recall/writeback proposals without I/O side effects."""

    def __init__(
        self,
        *,
        memory_service: Any | None = None,
        default_channels: Iterable[str] = ("voice",),
        default_priority: str = "normal",
    ) -> None:
        self.memory_service = memory_service
        self.default_channels = self._channels(default_channels)
        self.default_priority = _clean_text(default_priority) or "normal"

    def build_recall_request(
        self,
        turn: Any,
        *,
        query: str,
        channels: Iterable[str] | None = None,
        priority: str | int | float | None = None,
        reason: str,
        limit: int = 3,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self._base_payload(
            turn,
            kind="recall_request",
            query=query,
            channels=channels,
            priority=priority,
            reason=reason,
            metadata=metadata,
        )
        payload.update(
            {
                "limit": int(limit),
                "requester": "realtime_memory_orchestrator",
                "external_call": False,
                "stable": False,
            }
        )
        return self._record(turn, payload)

    def build_writeback_proposal(
        self,
        turn: Any,
        *,
        query: str,
        channels: Iterable[str] | None = None,
        priority: str | int | float | None = None,
        reason: str,
        summary: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        requires_commit: bool = True,
    ) -> dict[str, Any]:
        payload = self._base_payload(
            turn,
            kind="writeback_proposal",
            query=query,
            channels=channels,
            priority=priority,
            reason=reason,
            metadata=metadata,
        )
        payload.update(
            {
                "summary": _clean_text(summary) or _clean_text(query),
                "external_call": False,
                "requires_commit": bool(requires_commit),
                "stable": False,
            }
        )
        return self._record(turn, payload)

    def commit_candidates(
        self,
        turn: Any,
        *,
        session_id: str | None = None,
        actor_id: str | None = None,
        task_context: Mapping[str, Any] | None = None,
        default_modality: str = "audio_text",
        default_organ: str = "ear",
    ) -> dict[str, Any]:
        """Commit inert recall/writeback proposals through the configured memory service."""

        trace: dict[str, Any] = {
            "schema": CLOSED_LOOP_TRACE_SCHEMA,
            "round_id": str(_value(turn, "round_id", "")),
            "cancellation_token": _value(turn, "cancellation_token"),
            "session_id": session_id or "",
            "actor_id": actor_id or "",
            "external_call": self.memory_service is not None,
            "recall": {"count": 0, "items": []},
            "writeback": {"count": 0, "items": []},
            "errors": [],
        }
        candidates = _value(turn, "memory_candidates", [])
        if not isinstance(candidates, list):
            candidates = []
        if self.memory_service is None:
            trace["errors"].append({"error": "memory_service_not_configured"})
            return self._record_trace(turn, trace)

        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                continue
            kind = str(candidate.get("kind") or "")
            if kind == "recall_request":
                self._commit_recall(
                    candidate,
                    trace=trace,
                    index=index,
                    session_id=session_id,
                    actor_id=actor_id,
                    task_context=task_context,
                )
            elif kind == "writeback_proposal":
                self._commit_writeback(
                    candidate,
                    trace=trace,
                    index=index,
                    session_id=session_id,
                    actor_id=actor_id,
                    default_modality=default_modality,
                    default_organ=default_organ,
                )
        return self._record_trace(turn, trace)

    def _base_payload(
        self,
        turn: Any,
        *,
        kind: str,
        query: str,
        channels: Iterable[str] | None,
        priority: str | int | float | None,
        reason: str,
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        cleaned_query = _clean_text(query)
        if not cleaned_query:
            raise ValueError("memory proposal query is required")
        cleaned_reason = _clean_text(reason)
        if not cleaned_reason:
            raise ValueError("memory proposal reason is required")
        return _json_ready(
            {
                "kind": kind,
                "round_id": str(_value(turn, "round_id", "")),
                "cancellation_token": _value(turn, "cancellation_token"),
                "query": cleaned_query,
                "channels": self._channels(channels or self.default_channels),
                "priority": self._priority(priority),
                "reason": cleaned_reason,
                "metadata": dict(metadata or {}),
                "source": "memory_orchestrator",
            }
        )

    def _record(self, turn: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
        proposal = _json_ready(dict(payload))
        if hasattr(turn, "append_memory"):
            return turn.append_memory(proposal)
        current = _value(turn, "memory_candidates")
        if isinstance(current, list):
            current.append(proposal)
        elif isinstance(turn, dict):
            turn.setdefault("memory_candidates", []).append(proposal)
        return proposal

    def _record_trace(self, turn: Any, trace: Mapping[str, Any]) -> dict[str, Any]:
        payload = _json_ready(dict(trace))
        current = _value(turn, "memory_traces")
        if isinstance(current, list):
            current.append(payload)
        elif isinstance(turn, dict):
            turn.setdefault("memory_traces", []).append(payload)
        return payload

    def _commit_recall(
        self,
        candidate: Mapping[str, Any],
        *,
        trace: dict[str, Any],
        index: int,
        session_id: str | None,
        actor_id: str | None,
        task_context: Mapping[str, Any] | None,
    ) -> None:
        retriever = getattr(self.memory_service, "retrieve_context", None)
        if not callable(retriever):
            trace["errors"].append({"candidate_index": index, "kind": "recall_request", "error": "retrieve_context_missing"})
            return
        metadata = dict(candidate.get("metadata") or {}) if isinstance(candidate.get("metadata"), Mapping) else {}
        context = {
            "task_type": str(metadata.get("task_type") or "brain.respond"),
            "goal": str(metadata.get("goal") or "retrieve memory for realtime cognition"),
            "reason": str(candidate.get("reason") or ""),
            "channels": list(candidate.get("channels") or []),
            "priority": candidate.get("priority"),
            "round_id": candidate.get("round_id"),
            "cancellation_token": candidate.get("cancellation_token"),
            **dict(task_context or {}),
            **metadata,
        }
        try:
            result = retriever(
                _memory_query(
                    query=str(candidate.get("query") or ""),
                    session_id=session_id,
                    actor_id=actor_id,
                    task_context=context,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive boundary for injected services
            trace["errors"].append(
                {
                    "candidate_index": index,
                    "kind": "recall_request",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return

        diagnostics = dict(getattr(result, "recall_diagnostics", {}) or {})
        item = {
            "candidate_index": index,
            "kind": "recall_request",
            "status": "ok",
            "query": str(candidate.get("query") or ""),
            "summary": str(getattr(result, "summary", "") or ""),
            "memory_count": len(list(getattr(result, "relevant_memories", []) or [])),
            "selected_count": diagnostics.get("selected_count", 0),
            "selected_records": diagnostics.get("selected_records", []),
            "source_composition": diagnostics.get("source_composition", {}),
        }
        trace["recall"]["items"].append(_json_ready(item))
        trace["recall"]["count"] = len(trace["recall"]["items"])

    def _commit_writeback(
        self,
        candidate: Mapping[str, Any],
        *,
        trace: dict[str, Any],
        index: int,
        session_id: str | None,
        actor_id: str | None,
        default_modality: str,
        default_organ: str,
    ) -> None:
        summary = str(candidate.get("summary") or candidate.get("query") or "").strip()
        if candidate.get("requires_commit") is False:
            trace["writeback"]["items"].append(
                {
                    "candidate_index": index,
                    "kind": "writeback_proposal",
                    "status": "skipped",
                    "reason": "requires_commit_false",
                    "summary": summary,
                }
            )
            return
        writer = getattr(self.memory_service, "remember_episode", None)
        if not callable(writer):
            trace["errors"].append({"candidate_index": index, "kind": "writeback_proposal", "error": "remember_episode_missing"})
            return
        metadata = dict(candidate.get("metadata") or {}) if isinstance(candidate.get("metadata"), Mapping) else {}
        try:
            writer(
                session_id=session_id or str(candidate.get("round_id") or "unknown-session"),
                actor_id=actor_id,
                summary=summary,
                title=str(metadata.get("title") or "Realtime memory writeback"),
                memory_type=str(metadata.get("memory_type") or "conversation"),
                source=str(metadata.get("source") or "eibrain.audio_dialogue"),
                modality=str(metadata.get("modality") or default_modality),
                organ=str(metadata.get("organ") or default_organ),
                outcome=dict(metadata.get("outcome") or {}),
                content=dict(metadata.get("content") or {}),
                meta=dict(metadata.get("meta") or {}),
                tags=[str(tag) for tag in metadata.get("tags", [])],
                evidence=[dict(item) for item in metadata.get("evidence", []) if isinstance(item, Mapping)],
                links=[dict(item) for item in metadata.get("links", []) if isinstance(item, Mapping)],
            )
        except Exception as exc:  # pragma: no cover - defensive boundary for injected services
            trace["errors"].append(
                {
                    "candidate_index": index,
                    "kind": "writeback_proposal",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return
        status = dict(getattr(self.memory_service, "last_writeback_status", {}) or {})
        trace["writeback"]["items"].append(
            _json_ready(
                {
                    "candidate_index": index,
                    "kind": "writeback_proposal",
                    "status": str(status.get("status") or "ok"),
                    "summary": summary,
                    "source": status.get("source") or metadata.get("source") or "eibrain.audio_dialogue",
                    "memory_type": status.get("memory_type") or metadata.get("memory_type") or "conversation",
                    "diagnostics": status,
                }
            )
        )
        trace["writeback"]["count"] = len(
            [item for item in trace["writeback"]["items"] if isinstance(item, Mapping) and item.get("status") != "skipped"]
        )

    def _priority(self, value: str | int | float | None) -> str | int | float:
        if value is None:
            return self.default_priority
        if isinstance(value, (int, float)):
            return value
        return _clean_text(value) or self.default_priority

    @staticmethod
    def _channels(channels: Iterable[str]) -> list[str]:
        if isinstance(channels, str):
            channels = (channels,)
        normalized: list[str] = []
        for channel in channels:
            cleaned = _clean_text(channel)
            if cleaned and cleaned not in normalized:
                normalized.append(cleaned)
        return normalized or ["voice"]


__all__ = ["CLOSED_LOOP_TRACE_SCHEMA", "MemoryOrchestrator"]
