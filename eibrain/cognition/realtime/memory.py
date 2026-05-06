"""Inert memory proposal orchestration for realtime cognition."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

from eibrain.memory.contracts import MemoryTraceSummary


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


def _dict_if_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _unique_texts(values: Iterable[Any]) -> list[str]:
    unique: list[str] = []
    for value in values:
        cleaned = _clean_text(value)
        if cleaned and cleaned not in unique:
            unique.append(cleaned)
    return unique


def _candidate_key(candidate: Mapping[str, Any]) -> str:
    return _clean_text(candidate.get("id") or candidate.get("record_id") or candidate.get("query") or candidate.get("text"))


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
    """Build inert proposals and explicitly commit realtime memory loops."""

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

    def build_visual_recall_request(
        self,
        turn: Any,
        *,
        query: str | None = None,
        visual_context: Mapping[str, Any] | None = None,
        scene: Mapping[str, Any] | None = None,
        observation: Mapping[str, Any] | None = None,
        channels: Iterable[str] | None = ("vision",),
        priority: str | int | float | None = "realtime",
        reason: str = "visual_grounding_recall",
        limit: int = 3,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build an inert visual-grounding recall request from realtime vision context."""

        source = dict(scene or observation or visual_context or {})
        visual = dict(visual_context or {})
        if scene:
            visual.setdefault("scene", dict(scene))
        if observation:
            visual.setdefault("observation", dict(observation))
        summary = _clean_text(query) or _clean_text(source.get("summary")) or self._visual_summary(source)
        recall_metadata = {
            "task_type": "brain.orient",
            "modality": "vision",
            "organ": "eye",
            "visual_context": visual,
            "memory_type": "world_observation",
            **dict(metadata or {}),
        }
        return self.build_recall_request(
            turn,
            query=summary,
            channels=channels,
            priority=priority,
            reason=reason,
            limit=limit,
            metadata=recall_metadata,
        )

    def build_visual_world_writeback(
        self,
        turn: Any,
        *,
        scene: Mapping[str, Any] | None = None,
        observation: Mapping[str, Any] | None = None,
        summary: str | None = None,
        source_event_id: str = "",
        trace_id: str = "",
        frame_ref: str = "",
        channels: Iterable[str] | None = ("vision",),
        priority: str | int | float | None = "normal",
        reason: str = "visual_world_observation",
        evidence: Iterable[Mapping[str, Any]] | None = None,
        links: Iterable[Mapping[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
        requires_commit: bool = True,
    ) -> dict[str, Any]:
        """Build an inert world-observation candidate from visual scene content."""

        source = dict(scene or observation or {})
        objects = _mapping_items(source.get("objects") or source.get("detections") or [])
        resolved_summary = _clean_text(summary) or _clean_text(source.get("summary")) or self._visual_summary(source)
        content: dict[str, Any] = {
            "event_type": "vision_observation",
            "summary": resolved_summary,
            "modality": "vision",
            "organ": "eye",
            "scene": dict(scene or source),
            "objects": objects,
        }
        for key in ("relations", "events", "spatial", "confidence", "source"):
            if key in source:
                content[key] = _json_ready(source[key])
        if frame_ref:
            content["frame_ref"] = frame_ref
        elif source.get("frame_ref") or source.get("frame_path"):
            content["frame_ref"] = str(source.get("frame_ref") or source.get("frame_path"))

        meta = {
            "source_system": "eibrain",
            "source": "eibrain.visual_world",
            "trace_id": trace_id,
            "source_event_id": source_event_id,
            "event_type": "vision_observation",
            "memory_kind": "episodic",
            "retention": "episode",
            "promotion_status": "not_promoted",
            "identity_memory": False,
            "persona_memory": False,
            "privacy": {
                "scope": "situational_awareness",
                "sensitivity": "environmental",
                "allowed_use": "embodied_response",
            },
        }
        object_labels = [item.get("label") for item in objects]
        writeback_metadata = {
            "title": "Visual world observation",
            "memory_type": "world_observation",
            "source": "eibrain.visual_world",
            "modality": "vision",
            "organ": "eye",
            "content": content,
            "meta": meta,
            "outcome": {"success": None, "status": "observed", "modality": "vision", "organ": "eye"},
            "tags": _unique_texts(["world_observation", "vision", "eye", *object_labels]),
            "evidence": _mapping_items(evidence or []),
            "links": _mapping_items(links or []),
            **dict(metadata or {}),
        }
        return self.build_writeback_proposal(
            turn,
            query=resolved_summary,
            channels=channels,
            priority=priority,
            reason=reason,
            summary=resolved_summary,
            metadata=writeback_metadata,
            requires_commit=requires_commit,
        )

    def build_action_outcome_writeback(
        self,
        turn: Any,
        *,
        action: str,
        organ: str,
        success: bool | None,
        status: str,
        outcome: Mapping[str, Any] | None = None,
        modality: str = "multimodal_action",
        source_event_id: str = "",
        trace_id: str = "",
        suggested_adjustment: str = "",
        summary: str | None = None,
        channels: Iterable[str] | None = ("action",),
        priority: str | int | float | None = "normal",
        reason: str = "action_outcome_feedback",
        evidence: Iterable[Mapping[str, Any]] | None = None,
        links: Iterable[Mapping[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
        requires_commit: bool = True,
    ) -> dict[str, Any]:
        """Build an inert action-outcome candidate from embodied execution feedback."""

        cleaned_action = _clean_text(action)
        cleaned_organ = _clean_text(organ) or "cognition"
        cleaned_status = _clean_text(status) or "unknown"
        if not cleaned_action:
            raise ValueError("action outcome writeback action is required")
        resolved_summary = _clean_text(summary) or f"{cleaned_action} on {cleaned_organ} {cleaned_status}"
        outcome_payload = {
            "success": success,
            "status": cleaned_status,
            "modality": modality,
            "organ": cleaned_organ,
            **_dict_if_mapping(outcome),
        }
        content = {
            "event_type": "action_outcome",
            "summary": resolved_summary,
            "action": cleaned_action,
            "modality": modality,
            "organ": cleaned_organ,
            "success": success,
            "status": cleaned_status,
        }
        adjustment = _clean_text(suggested_adjustment)
        if adjustment:
            content["suggested_adjustment"] = adjustment

        writeback_metadata = {
            "title": "Realtime action outcome",
            "memory_type": "action_outcome",
            "source": "eibrain.outcome_feedback",
            "modality": modality,
            "organ": cleaned_organ,
            "content": content,
            "meta": {
                "source_system": "eibrain",
                "source": "eibrain.outcome_feedback",
                "trace_id": trace_id,
                "source_event_id": source_event_id,
                "event_type": "action_outcome",
                "memory_kind": "episodic",
                "retention": "episode",
                "promotion_status": "candidate" if success is False or adjustment else "not_promoted",
                "training_candidate": success is False or bool(adjustment),
                "identity_memory": False,
                "persona_memory": False,
                "privacy": {
                    "scope": "operational_feedback",
                    "sensitivity": "operational",
                    "allowed_use": "embodied_response",
                },
            },
            "outcome": outcome_payload,
            "tags": _unique_texts(["action_outcome", modality, cleaned_organ, cleaned_status]),
            "evidence": _mapping_items(evidence or []),
            "links": _mapping_items(links or []),
            **dict(metadata or {}),
        }
        return self.build_writeback_proposal(
            turn,
            query=resolved_summary,
            channels=channels,
            priority=priority,
            reason=reason,
            summary=resolved_summary,
            metadata=writeback_metadata,
            requires_commit=requires_commit,
        )

    def prefetch_recall(
        self,
        turn: Any,
        *,
        query: str,
        channels: Iterable[str] | None = None,
        priority: str | int | float | None = "realtime",
        reason: str = "prefetch_context_for_fast_lane",
        limit: int = 3,
        metadata: Mapping[str, Any] | None = None,
        session_id: str | None = None,
        actor_id: str | None = None,
        task_context: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Resolve a recall request immediately and attach results to the turn.

        The regular builders remain inert. This method is the explicit realtime
        fast-lane bridge used when an eimemory service has been configured.
        """

        request = self.build_recall_request(
            turn,
            query=query,
            channels=channels,
            priority=priority,
            reason=reason,
            limit=limit,
            metadata=metadata,
        )
        trace = self._new_trace(
            turn,
            session_id=session_id,
            actor_id=actor_id,
            external_call=self.memory_service is not None,
        )
        candidates = _value(turn, "memory_candidates", [])
        try:
            index = list(candidates).index(request) if isinstance(candidates, list) else 0
        except ValueError:
            index = 0
        self._track_prefetch_requested(trace, candidate=request, index=index)
        if self.memory_service is None:
            trace["errors"].append(
                {
                    "candidate_index": index,
                    "kind": "recall_request",
                    "error": "memory_service_not_configured",
                    "query": query,
                }
            )
            return self._record_trace(turn, trace).get("resolved_candidates", [])

        resolved = self._commit_recall(
            request,
            trace=trace,
            index=index,
            session_id=session_id,
            actor_id=actor_id,
            task_context={
                "phase": "fast_prefetch",
                "modality": "audio_text",
                "organ": "ear",
                **dict(task_context or {}),
            },
        )
        if isinstance(request, dict):
            request["committed"] = True
            request["commit_status"] = "ok" if not trace["errors"] else "error"
            request["resolved_count"] = len(resolved)
        if resolved:
            self._extend_turn_memory(turn, resolved)
        trace["resolved_candidates"] = resolved
        self._commit_memory_trace(trace, session_id=session_id, actor_id=actor_id)
        self._record_trace(turn, trace)
        return resolved

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

        trace = self._new_trace(
            turn,
            session_id=session_id,
            actor_id=actor_id,
            external_call=self.memory_service is not None,
        )
        candidates = _value(turn, "memory_candidates", [])
        if not isinstance(candidates, list):
            candidates = []
        if self.memory_service is None:
            trace["errors"].append({"error": "memory_service_not_configured"})
            return self._record_trace(turn, trace)

        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                continue
            if candidate.get("committed") is True:
                continue
            kind = str(candidate.get("kind") or "")
            if kind == "recall_request":
                self._track_prefetch_requested(trace, candidate=candidate, index=index)
                previous_error_count = len(trace["errors"])
                resolved = self._commit_recall(
                    candidate,
                    trace=trace,
                    index=index,
                    session_id=session_id,
                    actor_id=actor_id,
                    task_context=task_context,
                )
                if isinstance(candidate, dict):
                    candidate["committed"] = True
                    candidate["commit_status"] = "ok" if len(trace["errors"]) == previous_error_count else "error"
                    candidate["resolved_count"] = len(resolved)
                if resolved:
                    self._extend_turn_memory(turn, resolved)
            elif kind == "writeback_proposal":
                self._track_write_proposed(trace, candidate=candidate, index=index)
                previous_error_count = len(trace["errors"])
                self._commit_writeback(
                    candidate,
                    trace=trace,
                    index=index,
                    session_id=session_id,
                    actor_id=actor_id,
                    default_modality=default_modality,
                    default_organ=default_organ,
                )
                if isinstance(candidate, dict) and len(trace["errors"]) == previous_error_count:
                    candidate["committed"] = True
                    candidate["commit_status"] = "ok"
        self._commit_memory_trace(trace, session_id=session_id, actor_id=actor_id)
        return self._record_trace(turn, trace)

    def record_reply_memory_usage(
        self,
        turn: Any,
        *,
        reply_text: str,
        used_items: Iterable[Mapping[str, Any] | str] | None = None,
        session_id: str | None = None,
        actor_id: str | None = None,
    ) -> dict[str, Any]:
        trace = self._latest_trace(turn) or self._new_trace(
            turn,
            session_id=session_id,
            actor_id=actor_id,
            external_call=self.memory_service is not None,
        )
        reply = {
            "reply_text": _clean_text(reply_text),
            "used_recall_items": self._mark_recall_items_used(turn, used_items=used_items),
        }
        reply["used_count"] = len(reply["used_recall_items"])
        trace["reply"] = _json_ready(reply)
        self._commit_memory_trace(
            trace,
            session_id=session_id or str(trace.get("session_id") or "") or None,
            actor_id=actor_id or str(trace.get("actor_id") or "") or None,
        )
        return self._record_trace(turn, trace)

    def _new_trace(
        self,
        turn: Any,
        *,
        session_id: str | None,
        actor_id: str | None,
        external_call: bool,
    ) -> dict[str, Any]:
        trace_id = self._trace_id(turn)
        return {
            "schema": CLOSED_LOOP_TRACE_SCHEMA,
            "trace_id": trace_id,
            "round_id": str(_value(turn, "round_id", "")),
            "cancellation_token": _value(turn, "cancellation_token"),
            "session_id": session_id or "",
            "actor_id": actor_id or "",
            "external_call": external_call,
            "prefetch": {"requested": [], "result": []},
            "recall": {"count": 0, "items": []},
            "write": {"proposed": [], "committed": []},
            "writeback": {"count": 0, "items": []},
            "reply": {"reply_text": "", "used_recall_items": [], "used_count": 0},
            "memory_trace_summary": {},
            "errors": [],
        }

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
        payload["memory_trace_summary"] = self._trace_summary(payload)
        current = _value(turn, "memory_traces")
        if isinstance(current, list):
            trace_id = str(payload.get("trace_id") or "")
            if trace_id:
                for index, item in enumerate(current):
                    if isinstance(item, Mapping) and str(item.get("trace_id") or "") == trace_id:
                        current[index] = payload
                        return payload
            current.append(payload)
        elif isinstance(turn, dict):
            traces = turn.setdefault("memory_traces", [])
            trace_id = str(payload.get("trace_id") or "")
            if trace_id:
                for index, item in enumerate(traces):
                    if isinstance(item, Mapping) and str(item.get("trace_id") or "") == trace_id:
                        traces[index] = payload
                        return payload
            traces.append(payload)
        return payload

    def _extend_turn_memory(self, turn: Any, items: Iterable[Mapping[str, Any]]) -> None:
        current = _value(turn, "memory_candidates")
        if not isinstance(current, list):
            if isinstance(turn, dict):
                current = turn.setdefault("memory_candidates", [])
            else:
                return
        seen = {_candidate_key(item) for item in current if isinstance(item, Mapping)}
        for item in items:
            payload = _json_ready(dict(item))
            key = _candidate_key(payload)
            if key in seen:
                continue
            seen.add(key)
            current.append(payload)

    def _commit_memory_trace(
        self,
        trace: dict[str, Any],
        *,
        session_id: str | None,
        actor_id: str | None,
    ) -> None:
        recorder = getattr(self.memory_service, "record_memory_trace", None)
        if not callable(recorder):
            return
        payload = _json_ready(dict(trace))
        payload["memory_trace_summary"] = self._trace_summary(payload)
        try:
            result = recorder(payload, session_id=session_id, actor_id=actor_id)
        except Exception as exc:  # pragma: no cover - defensive boundary for injected services
            error = f"{type(exc).__name__}: {exc}"
            trace["trace_record"] = {"status": "error", "error": error}
            trace["errors"].append({"kind": "memory_trace", "error": error})
            return
        if isinstance(result, Mapping) and result:
            result_payload = dict(result)
            result_body = result_payload.get("result")
            result_record = dict(result_body) if isinstance(result_body, Mapping) else {}
            record_id = result_record.get("record_id") or result_record.get("id")
            trace["trace_record"] = _json_ready(
                {
                    "status": "ok",
                    "record_id": record_id,
                    "diagnostics": result_payload,
                }
            )
        else:
            trace["trace_record"] = {"status": "skipped", "reason": "empty_result"}

    def _commit_recall(
        self,
        candidate: Mapping[str, Any],
        *,
        trace: dict[str, Any],
        index: int,
        session_id: str | None,
        actor_id: str | None,
        task_context: Mapping[str, Any] | None,
    ) -> list[dict[str, Any]]:
        retriever = getattr(self.memory_service, "retrieve_context", None)
        if not callable(retriever):
            trace["errors"].append({"candidate_index": index, "kind": "recall_request", "error": "retrieve_context_missing"})
            return []
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
            return []

        diagnostics = dict(getattr(result, "recall_diagnostics", {}) or {})
        resolved = self._recall_candidates_from_result(result, candidate=candidate, diagnostics=diagnostics)
        self._track_prefetch_result(trace, resolved=resolved)
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
            "resolved_count": len(resolved),
        }
        if resolved:
            item["resolved_candidates"] = resolved
        trace["recall"]["items"].append(_json_ready(item))
        trace["recall"]["count"] = len(trace["recall"]["items"])
        return resolved

    def _recall_candidates_from_result(
        self,
        result: Any,
        *,
        candidate: Mapping[str, Any],
        diagnostics: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        relevant = [str(item) for item in list(getattr(result, "relevant_memories", []) or []) if str(item).strip()]
        summary = str(getattr(result, "summary", "") or "").strip()
        if not relevant and summary:
            relevant = [summary]
        selected_records = _mapping_items(diagnostics.get("selected_records", []))
        try:
            limit = max(1, int(candidate.get("limit") or 3))
        except (TypeError, ValueError):
            limit = 3

        resolved: list[dict[str, Any]] = []
        for item_index, text in enumerate(relevant[:limit]):
            record = selected_records[item_index] if item_index < len(selected_records) else {}
            record_id = _clean_text(record.get("record_id") or record.get("id"))
            source = _clean_text(record.get("source")) or "eimemory_recall"
            score = record.get("score", record.get("confidence"))
            if score is None:
                score = round(max(0.1, 0.92 - item_index * 0.05), 3)
            payload = {
                "id": record_id or f"{candidate.get('round_id', 'round')}:eimemory:{item_index}",
                "record_id": record_id,
                "kind": "recall",
                "query": str(candidate.get("query") or ""),
                "text": text,
                "summary": summary or text,
                "score": score,
                "source": "eimemory_recall",
                "memory_source": source,
                "title": record.get("title", ""),
                "memory_type": record.get("memory_type", record.get("type", "")),
                "stable": False,
                "external_call": True,
                "used_in_reply": False,
                "selected_record": dict(record),
            }
            resolved.append(_json_ready(payload))
        return resolved

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
        metadata = dict(candidate.get("metadata") or {}) if isinstance(candidate.get("metadata"), Mapping) else {}
        source = str(metadata.get("source") or "eibrain.audio_dialogue")
        memory_type = str(metadata.get("memory_type") or "conversation")
        if candidate.get("requires_commit") is False:
            skipped = {
                "candidate_index": index,
                "kind": "writeback_proposal",
                "status": "skipped",
                "reason": "requires_commit_false",
                "summary": summary,
                "source": source,
                "memory_type": memory_type,
            }
            trace["writeback"]["items"].append(skipped)
            trace["write"]["committed"].append(skipped)
            self._refresh_writeback_count(trace)
            return
        world_observation = memory_type == "world_observation" and str(metadata.get("modality") or default_modality) == "vision"
        writer_name = "remember_world_observation" if world_observation else "remember_episode"
        writer = getattr(self.memory_service, writer_name, None)
        if not callable(writer) and world_observation:
            writer_name = "remember_episode"
            writer = getattr(self.memory_service, writer_name, None)
        if not callable(writer):
            error = f"{writer_name}_missing"
            trace["errors"].append({"candidate_index": index, "kind": "writeback_proposal", "error": error})
            failure = {
                "candidate_index": index,
                "kind": "writeback_proposal",
                "status": "error",
                "error": error,
                "summary": summary,
                "source": source,
                "memory_type": memory_type,
            }
            trace["writeback"]["items"].append(failure)
            trace["write"]["committed"].append(failure)
            self._refresh_writeback_count(trace)
            return
        try:
            payload = {
                "session_id": session_id or str(candidate.get("round_id") or "unknown-session"),
                "actor_id": actor_id,
                "summary": summary,
                "title": str(metadata.get("title") or "Realtime memory writeback"),
                "memory_type": memory_type,
                "source": source,
                "modality": str(metadata.get("modality") or default_modality),
                "organ": str(metadata.get("organ") or default_organ),
                "outcome": dict(metadata.get("outcome") or {}),
                "content": dict(metadata.get("content") or {}),
                "meta": dict(metadata.get("meta") or {}),
                "tags": [str(tag) for tag in metadata.get("tags", [])],
                "evidence": [dict(item) for item in metadata.get("evidence", []) if isinstance(item, Mapping)],
                "links": [dict(item) for item in metadata.get("links", []) if isinstance(item, Mapping)],
            }
            if writer_name == "remember_world_observation":
                result = writer(
                    session_id=payload["session_id"],
                    actor_id=payload["actor_id"],
                    summary=payload["summary"],
                    title=payload["title"],
                    content=payload["content"],
                    meta=payload["meta"],
                    tags=payload["tags"],
                    evidence=payload["evidence"],
                    links=payload["links"],
                )
            else:
                result = writer(**payload)
        except Exception as exc:  # pragma: no cover - defensive boundary for injected services
            error = f"{type(exc).__name__}: {exc}"
            trace["errors"].append({"candidate_index": index, "kind": "writeback_proposal", "error": error})
            failure = {
                "candidate_index": index,
                "kind": "writeback_proposal",
                "status": "error",
                "error": error,
                "summary": summary,
                "source": source,
                "memory_type": memory_type,
            }
            trace["writeback"]["items"].append(failure)
            trace["write"]["committed"].append(failure)
            self._refresh_writeback_count(trace)
            return
        status = dict(getattr(self.memory_service, "last_writeback_status", {}) or {})
        record_id = status.get("record_id") or (result.get("record_id") if isinstance(result, Mapping) else None)
        committed = _json_ready(
            {
                "candidate_index": index,
                "kind": "writeback_proposal",
                "status": str(status.get("status") or "ok"),
                "summary": summary,
                "source": status.get("source") or source,
                "memory_type": status.get("memory_type") or memory_type,
                "record_id": record_id,
                "diagnostics": status,
            }
        )
        trace["writeback"]["items"].append(committed)
        trace["write"]["committed"].append(committed)
        self._refresh_writeback_count(trace)

    @staticmethod
    def _refresh_writeback_count(trace: dict[str, Any]) -> None:
        trace["writeback"]["count"] = len(
            [item for item in trace["writeback"]["items"] if isinstance(item, Mapping) and item.get("status") != "skipped"]
        )

    @staticmethod
    def _visual_summary(source: Mapping[str, Any]) -> str:
        objects = _mapping_items(source.get("objects") or source.get("detections") or [])
        labels = _unique_texts(item.get("label") for item in objects)
        if labels:
            return f"Observed {', '.join(labels)}"
        return "Observed visual scene"

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

    def _track_prefetch_requested(self, trace: dict[str, Any], *, candidate: Mapping[str, Any], index: int) -> None:
        request = {
            "candidate_index": index,
            "query": str(candidate.get("query") or ""),
            "reason": str(candidate.get("reason") or ""),
            "channels": list(candidate.get("channels") or []),
            "limit": int(candidate.get("limit") or 3),
        }
        existing = trace["prefetch"]["requested"]
        if request not in existing:
            existing.append(_json_ready(request))

    def _track_prefetch_result(self, trace: dict[str, Any], *, resolved: Iterable[Mapping[str, Any]]) -> None:
        results = trace["prefetch"]["result"]
        seen = {_candidate_key(item) for item in results if isinstance(item, Mapping)}
        for item in resolved:
            payload = {
                "record_id": str(item.get("record_id") or ""),
                "text": str(item.get("text") or ""),
                "title": str(item.get("title") or ""),
                "memory_type": str(item.get("memory_type") or ""),
                "memory_source": str(item.get("memory_source") or ""),
            }
            key = _candidate_key(payload)
            if key in seen:
                continue
            seen.add(key)
            results.append(_json_ready(payload))

    def _track_write_proposed(self, trace: dict[str, Any], *, candidate: Mapping[str, Any], index: int) -> None:
        metadata = dict(candidate.get("metadata") or {}) if isinstance(candidate.get("metadata"), Mapping) else {}
        proposal = {
            "candidate_index": index,
            "summary": str(candidate.get("summary") or candidate.get("query") or ""),
            "source": str(metadata.get("source") or "eibrain.audio_dialogue"),
            "memory_type": str(metadata.get("memory_type") or "conversation"),
        }
        existing = trace["write"]["proposed"]
        if proposal not in existing:
            existing.append(_json_ready(proposal))

    def _mark_recall_items_used(
        self,
        turn: Any,
        *,
        used_items: Iterable[Mapping[str, Any] | str] | None,
    ) -> list[dict[str, Any]]:
        candidates = _value(turn, "memory_candidates", [])
        if not isinstance(candidates, list):
            return []
        requested = list(used_items or [])
        requested_keys = {
            _clean_text(item)
            if isinstance(item, str)
            else _clean_text(item.get("record_id") or item.get("id") or item.get("text"))
            for item in requested
        }
        used: list[dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict) or candidate.get("kind") != "recall":
                continue
            key = _clean_text(candidate.get("record_id") or candidate.get("id") or candidate.get("text"))
            if requested_keys and key not in requested_keys:
                continue
            candidate["used_in_reply"] = True
            used.append(
                _json_ready(
                    {
                        "record_id": candidate.get("record_id", ""),
                        "text": candidate.get("text", ""),
                        "title": candidate.get("title", ""),
                        "memory_type": candidate.get("memory_type", ""),
                        "memory_source": candidate.get("memory_source", ""),
                    }
                )
            )
        return used

    def _latest_trace(self, turn: Any) -> dict[str, Any] | None:
        current = _value(turn, "memory_traces")
        if isinstance(current, list) and current:
            latest = current[-1]
            if isinstance(latest, Mapping):
                return dict(latest)
        return None

    def _trace_id(self, turn: Any) -> str:
        current = _value(turn, "memory_traces")
        count = len(current) if isinstance(current, list) else 0
        round_id = _clean_text(_value(turn, "round_id", "memory-trace")) or "memory-trace"
        return f"{round_id}:memory:{count + 1}"

    def _trace_summary(self, trace: Mapping[str, Any]) -> dict[str, Any]:
        errors = list(trace.get("errors") or [])
        committed = [
            item for item in list(_value(trace, "write", {}).get("committed", []))
            if isinstance(item, Mapping) and item.get("status") == "ok"
        ]
        status = "error" if errors and not committed and not list(trace.get("prefetch", {}).get("result", [])) else "partial" if errors else "ok"
        summary = MemoryTraceSummary(
            trace_id=str(trace.get("trace_id") or ""),
            status=status,
            prefetch_requested=len(list(trace.get("prefetch", {}).get("requested", []))),
            prefetch_result=len(list(trace.get("prefetch", {}).get("result", []))),
            write_proposed=len(list(trace.get("write", {}).get("proposed", []))),
            write_committed=len(committed),
            reply_used=len(list(trace.get("reply", {}).get("used_recall_items", []))),
            queries=_unique_texts(item.get("query") for item in list(trace.get("prefetch", {}).get("requested", [])) if isinstance(item, Mapping)),
            written_memory_types=_unique_texts(
                item.get("memory_type") for item in committed if isinstance(item, Mapping)
            ),
            used_memory_ids=_unique_texts(
                item.get("record_id") for item in list(trace.get("reply", {}).get("used_recall_items", [])) if isinstance(item, Mapping)
            ),
        )
        return _json_ready(asdict(summary))


__all__ = ["CLOSED_LOOP_TRACE_SCHEMA", "MemoryOrchestrator"]
