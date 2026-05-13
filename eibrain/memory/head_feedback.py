"""Builders for head execution feedback memory records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from time import time
from typing import Any, Mapping

from eibrain.memory.scoring_compat import merge_memory_metadata, normalize_memory_metadata


ALLOWED_MEMORY_KINDS = {"working", "episodic", "procedural"}
REQUIRED_FEEDBACK_FIELDS = (
    "what_attempted",
    "planned_by",
    "executed_by",
    "success",
    "latency_ms",
    "error",
    "user_feedback",
    "suggested_adjustment",
    "trace_id",
    "timestamp_ms",
)


@dataclass(slots=True)
class HeadFeedbackRecordBuilder:
    """Pure adapter from head execution signals into memory-ready records."""

    source: str = "eibrain.head_feedback"
    default_planned_by: str = "eibrain"
    default_executed_by: str = "eihead"

    def build(
        self,
        *,
        action: object,
        outcome: object | None = None,
        feedback: object | None = None,
        trace_id: str | None = None,
        timestamp_ms: int | None = None,
        planned_by: str | None = None,
        executed_by: str | None = None,
        memory_kind: str | None = None,
        category: str | None = None,
    ) -> dict[str, object]:
        return build_head_feedback_record(
            action=action,
            outcome=outcome,
            feedback=feedback,
            trace_id=trace_id,
            timestamp_ms=timestamp_ms,
            planned_by=planned_by or self.default_planned_by,
            executed_by=executed_by or self.default_executed_by,
            memory_kind=memory_kind,
            category=category,
            source=self.source,
        )


def build_head_feedback_record(
    *,
    action: object,
    outcome: object | None = None,
    feedback: object | None = None,
    trace_id: str | None = None,
    timestamp_ms: int | None = None,
    planned_by: str | None = None,
    executed_by: str | None = None,
    memory_kind: str | None = None,
    category: str | None = None,
    source: str = "eibrain.head_feedback",
) -> dict[str, object]:
    """Convert ExecutionOutcome/UserFeedback-like values into a plain record.

    The returned dict is intentionally inert: callers may pass it to eimemory or
    eitraining later, but this module never performs RPC/network writes.
    """

    action_map = _to_plain_dict(action)
    outcome_map = _to_plain_dict(outcome)
    feedback_map = _to_plain_dict(feedback)
    details = _to_plain_dict(outcome_map.get("details"))

    attempted = _first_text(
        action_map,
        "what_attempted",
        "action_kind",
        "kind",
        "type",
        "name",
        default=_stringify_compact(action),
    )
    success = _extract_success(outcome_map, details)
    latency_ms = _extract_int(outcome_map, details, "latency_ms", "duration_ms", "elapsed_ms")
    error = _extract_error(outcome_map, details)
    user_feedback = _extract_feedback_text(feedback, feedback_map)
    suggested_adjustment = _extract_suggested_adjustment(feedback_map, outcome_map, action_map)
    resolved_trace_id = _first_text_from_maps(
        ({"trace_id": trace_id} if trace_id else {}),
        outcome_map,
        details,
        action_map,
        feedback_map,
        keys=("trace_id", "request_id", "turn_id"),
        default="",
    )
    resolved_source_event_id = _first_text_from_maps(
        outcome_map,
        details,
        action_map,
        feedback_map,
        keys=("source_event_id", "event_id", "request_id", "turn_id"),
        default="",
    )
    resolved_timestamp_ms = timestamp_ms or _extract_int(
        outcome_map,
        feedback_map,
        action_map,
        "timestamp_ms",
        "ts_ms",
    ) or int(time() * 1000)

    resolved_memory_kind = _normalize_memory_kind(
        memory_kind or _infer_memory_kind(success=success, user_feedback=user_feedback, suggested_adjustment=suggested_adjustment)
    )
    resolved_category = category or _infer_category(
        memory_kind=resolved_memory_kind,
        success=success,
        error=error,
    )
    resolved_planned_by = planned_by or _first_text_from_maps(
        action_map,
        outcome_map,
        keys=("planned_by", "planner", "source"),
        default="eibrain",
    )
    resolved_executed_by = executed_by or _first_text_from_maps(
        outcome_map,
        details,
        action_map,
        keys=("executed_by", "executor", "target_id", "organ"),
        default="eihead",
    )
    resolved_organ = _infer_organ(action_map, outcome_map, details)
    confidence = _extract_float(outcome_map, details, feedback_map, "confidence", "score")

    record: dict[str, object] = {
        "record_type": "head_execution_feedback",
        "memory_type": "head_execution_feedback",
        "memory_kind": resolved_memory_kind,
        "category": resolved_category,
        "source": source,
        "organ": resolved_organ,
        "modality": _infer_modality(action_map, outcome_map, details),
        "confidence": confidence,
        "what_attempted": attempted,
        "planned_by": resolved_planned_by,
        "executed_by": resolved_executed_by,
        "success": success,
        "latency_ms": latency_ms,
        "error": error,
        "user_feedback": user_feedback,
        "suggested_adjustment": suggested_adjustment,
        "trace_id": resolved_trace_id,
        "source_event_id": resolved_source_event_id,
        "timestamp_ms": resolved_timestamp_ms,
        "persona_memory": False,
        "retention": _retention_for(resolved_memory_kind),
        "promotion_status": "candidate" if resolved_memory_kind == "procedural" else "not_promoted",
        "action": action_map,
        "outcome": outcome_map,
        "feedback": feedback_map,
    }
    record["tracking_provenance"] = {
        "trace_id": resolved_trace_id,
        "source_event_id": resolved_source_event_id,
        "planned_by": resolved_planned_by,
        "executed_by": resolved_executed_by,
        "organ": resolved_organ,
    }
    record["writeback"] = _writeback_for(resolved_memory_kind)
    record["summary"] = summarize_head_feedback_record(record)
    record["tags"] = _tags_for(record)
    return record


def build_eimemory_ingest_params(record: Mapping[str, object]) -> dict[str, object]:
    """Build memory.ingest params without calling the eimemory RPC client."""

    cleaned = dict(record)
    content = {field: cleaned.get(field) for field in REQUIRED_FEEDBACK_FIELDS}
    content.update(
        {
            "action": cleaned.get("action", {}),
            "outcome": cleaned.get("outcome", {}),
            "feedback": cleaned.get("feedback", {}),
            "confidence": cleaned.get("confidence"),
            "tracking_provenance": cleaned.get("tracking_provenance", {}),
        }
    )
    meta = normalize_memory_metadata(
        merge_memory_metadata(
            {
                "quality": cleaned.get("quality"),
                "scoring": cleaned.get("scoring"),
                "memory_score_v1": cleaned.get("memory_score_v1"),
            },
            {
                "memory_kind": cleaned.get("memory_kind"),
                "category": cleaned.get("category"),
                "trace_id": cleaned.get("trace_id"),
                "source_event_id": cleaned.get("source_event_id"),
                "timestamp_ms": cleaned.get("timestamp_ms"),
                "confidence": cleaned.get("confidence"),
                "tracking_provenance": cleaned.get("tracking_provenance", {}),
                "persona_memory": cleaned.get("persona_memory", False),
                "retention": cleaned.get("retention"),
                "promotion_status": cleaned.get("promotion_status"),
                "writeback": cleaned.get("writeback", {}),
            },
        )
    )
    return {
        "text": str(cleaned.get("summary") or summarize_head_feedback_record(cleaned)),
        "title": _title_for(cleaned),
        "memory_type": str(cleaned.get("memory_type") or "head_execution_feedback"),
        "source": str(cleaned.get("source") or "eibrain.head_feedback"),
        "organ": str(cleaned.get("organ") or "head"),
        "modality": str(cleaned.get("modality") or "multimodal_action"),
        "confidence": cleaned.get("confidence"),
        "outcome": {
            "success": cleaned.get("success"),
            "latency_ms": cleaned.get("latency_ms"),
            "error": cleaned.get("error"),
            "trace_id": cleaned.get("trace_id"),
        },
        "content": content,
        "meta": meta,
        "tags": list(cleaned.get("tags", [])),
    }


def build_eitraining_trace(record: Mapping[str, object]) -> dict[str, object]:
    """Build a training/experience payload without writing it anywhere."""

    cleaned = dict(record)
    return {
        "signal_type": "head_execution_feedback",
        "trace_id": cleaned.get("trace_id"),
        "task_type": "head.execute",
        "category": cleaned.get("category"),
        "memory_kind": cleaned.get("memory_kind"),
        "what_attempted": cleaned.get("what_attempted"),
        "planned_by": cleaned.get("planned_by"),
        "executed_by": cleaned.get("executed_by"),
        "success": cleaned.get("success"),
        "latency_ms": cleaned.get("latency_ms"),
        "error": cleaned.get("error"),
        "user_feedback": cleaned.get("user_feedback"),
        "suggested_adjustment": cleaned.get("suggested_adjustment"),
        "confidence": cleaned.get("confidence"),
        "tracking_provenance": cleaned.get("tracking_provenance", {}),
        "timestamp_ms": cleaned.get("timestamp_ms"),
    }


def summarize_head_feedback_record(record: Mapping[str, object]) -> str:
    attempted = str(record.get("what_attempted") or "head action").strip()
    success = record.get("success")
    status = "succeeded" if success is True else "failed" if success is False else "unknown outcome"
    parts = [f"{attempted} {status}"]
    latency_ms = record.get("latency_ms")
    if latency_ms is not None:
        parts.append(f"latency={latency_ms}ms")
    error = str(record.get("error") or "").strip()
    if error:
        parts.append(f"error={error}")
    feedback = str(record.get("user_feedback") or "").strip()
    if feedback:
        parts.append(f"feedback={feedback}")
    adjustment = str(record.get("suggested_adjustment") or "").strip()
    if adjustment:
        parts.append(f"adjustment={adjustment}")
    trace_id = str(record.get("trace_id") or "").strip()
    if trace_id:
        parts.append(f"trace={trace_id}")
    return "; ".join(parts)


def _to_plain_dict(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return {str(key): _json_safe(item) for key, item in asdict(value).items()}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        mapped = to_dict()
        if isinstance(mapped, Mapping):
            return {str(key): _json_safe(item) for key, item in mapped.items()}
    return {}


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _first_text(mapping: Mapping[str, object], *keys: str, default: str = "") -> str:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default.strip()


def _first_text_from_maps(
    *mappings: Mapping[str, object],
    keys: tuple[str, ...],
    default: str = "",
) -> str:
    for mapping in mappings:
        value = _first_text(mapping, *keys)
        if value:
            return value
    return default


def _extract_success(outcome: Mapping[str, object], details: Mapping[str, object]) -> bool | None:
    for mapping in (outcome, details):
        if "success" in mapping and isinstance(mapping["success"], bool):
            return bool(mapping["success"])
    status = _first_text(outcome, "status") or _first_text(details, "status")
    normalized = status.lower()
    if normalized in {"ok", "success", "succeeded", "completed", "executed", "done"}:
        return True
    if normalized in {"error", "failed", "failure", "timeout", "aborted"}:
        return False
    return None


def _extract_int(*mappings_and_keys: object) -> int | None:
    mappings = [item for item in mappings_and_keys if isinstance(item, Mapping)]
    keys = [str(item) for item in mappings_and_keys if isinstance(item, str)]
    for mapping in mappings:
        for key in keys:
            value = mapping.get(str(key))
            if isinstance(value, bool) or value is None:
                continue
            try:
                return int(float(str(value)))
            except ValueError:
                continue
    return None


def _extract_float(*mappings_and_keys: object) -> float | None:
    mappings = [item for item in mappings_and_keys if isinstance(item, Mapping)]
    keys = [str(item) for item in mappings_and_keys if isinstance(item, str)]
    for mapping in mappings:
        for key in keys:
            value = mapping.get(str(key))
            if isinstance(value, bool) or value is None:
                continue
            try:
                return float(str(value))
            except ValueError:
                continue
    return None


def _extract_error(outcome: Mapping[str, object], details: Mapping[str, object]) -> str:
    return _first_text_from_maps(outcome, details, keys=("error", "error_message", "exception"), default="")


def _extract_feedback_text(raw_feedback: object, feedback: Mapping[str, object]) -> str:
    if isinstance(raw_feedback, str):
        return raw_feedback.strip()
    return _first_text(feedback, "user_feedback", "feedback", "text", "comment", "utterance")


def _extract_suggested_adjustment(*mappings: Mapping[str, object]) -> str:
    return _first_text_from_maps(
        *mappings,
        keys=("suggested_adjustment", "adjustment", "next_adjustment", "recommendation"),
        default="",
    )


def _infer_memory_kind(*, success: bool | None, user_feedback: str, suggested_adjustment: str) -> str:
    if suggested_adjustment:
        return "procedural"
    if success is None and not user_feedback:
        return "working"
    return "episodic"


def _normalize_memory_kind(memory_kind: str) -> str:
    normalized = str(memory_kind or "").strip().lower()
    if normalized not in ALLOWED_MEMORY_KINDS:
        raise ValueError(f"memory_kind must be one of {sorted(ALLOWED_MEMORY_KINDS)}")
    return normalized


def _infer_category(*, memory_kind: str, success: bool | None, error: str) -> str:
    if memory_kind == "procedural":
        return "head.procedural_adjustment_candidate"
    if memory_kind == "working":
        return "head.transient_execution_signal"
    if success is False or error:
        return "head.execution_incident"
    return "head.execution_outcome"


def _infer_organ(*mappings: Mapping[str, object]) -> str:
    organ = _first_text_from_maps(*mappings, keys=("organ", "target_organ", "target_id"), default="")
    return organ or "head"


def _infer_modality(*mappings: Mapping[str, object]) -> str:
    modality = _first_text_from_maps(*mappings, keys=("modality", "channel"), default="")
    return modality or "multimodal_action"


def _retention_for(memory_kind: str) -> str:
    return {
        "working": "short_lived",
        "episodic": "episode",
        "procedural": "adjustment_candidate",
    }[memory_kind]


def _writeback_for(memory_kind: str) -> dict[str, object]:
    if memory_kind == "procedural":
        return {
            "eligible": True,
            "durable": True,
            "reason": "procedural_adjustment",
            "target_memory_type": "head_execution_feedback",
        }
    if memory_kind == "episodic":
        return {
            "eligible": True,
            "durable": True,
            "reason": "execution_outcome",
            "target_memory_type": "head_execution_feedback",
        }
    return {
        "eligible": False,
        "durable": False,
        "reason": "transient_signal",
        "target_memory_type": "head_execution_feedback",
    }


def _title_for(record: Mapping[str, object]) -> str:
    category = str(record.get("category") or "head.execution_feedback")
    trace_id = str(record.get("trace_id") or "").strip()
    if trace_id:
        return f"{category}: {trace_id}"
    return category


def _tags_for(record: Mapping[str, object]) -> list[str]:
    tags = [
        "head_feedback",
        str(record.get("memory_kind") or ""),
        str(record.get("category") or "").replace(".", "_"),
    ]
    if record.get("success") is False:
        tags.append("incident")
    if record.get("suggested_adjustment"):
        tags.append("adjustment_candidate")
    return [tag for tag in tags if tag]


def _stringify_compact(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return "head action"
    mapped = _to_plain_dict(value)
    if mapped:
        return ", ".join(f"{key}={mapped[key]}" for key in sorted(mapped))
    return str(value).strip() or "head action"
