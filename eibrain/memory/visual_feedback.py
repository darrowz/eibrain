"""Visual feedback and training-record builders.

The functions here are pure adapters: they prepare memory/training payloads but
never perform network writes.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from time import time
from typing import Any, Mapping


SOURCE = "eibrain.vision_feedback"


def build_visual_feedback_record(
    *,
    feedback_type: str,
    subject: object | None = None,
    outcome: object | None = None,
    follow_score: object | None = None,
    action: object | None = None,
    user_feedback: object | None = None,
    label_correction: object | None = None,
    frame: object | None = None,
    privacy: object | None = None,
    round_id: str | None = None,
    session_id: str | None = None,
    timestamp_ms: int | None = None,
    source: str = SOURCE,
) -> dict[str, object]:
    """Build a JSON-ready visual feedback record."""

    subject_map = _to_plain_dict(subject)
    outcome_map = _to_plain_dict(outcome)
    follow_map = _to_plain_dict(follow_score)
    action_map = _to_plain_dict(action)
    correction_map = _to_plain_dict(label_correction)
    frame_map = _to_plain_dict(frame)
    privacy_map = _to_plain_dict(privacy)
    feedback_text = _feedback_text(user_feedback)
    resolved_type = _normalize_token(feedback_type) or "visual_feedback"
    resolved_outcome = _resolve_outcome(
        feedback_type=resolved_type,
        outcome_map=outcome_map,
        follow_map=follow_map,
        correction_map=correction_map,
    )
    before_metrics, after_metrics = _split_metrics(follow_map, outcome_map)
    error = _first_text(outcome_map, follow_map, keys=("error", "reason"), default="")
    privacy_result = _privacy_result(privacy_map)

    record: dict[str, object] = {
        "record_type": "visual_feedback",
        "feedback_type": resolved_type,
        "source": source,
        "subject": _redact_if_needed(subject_map, privacy_result),
        "outcome": resolved_outcome,
        "success": resolved_outcome == "success",
        "error": error,
        "user_feedback": feedback_text,
        "label_correction": correction_map,
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "action": _redact_if_needed(action_map, privacy_result),
        "frame": _redact_if_needed(frame_map, privacy_result),
        "round_id": round_id or "",
        "session_id": session_id or "",
        "timestamp_ms": timestamp_ms or int(time() * 1000),
        "importance": _importance(
            feedback_type=resolved_type,
            outcome=resolved_outcome,
            error=error,
            correction=correction_map,
        ),
        "privacy": privacy_result,
    }
    record["writeback"] = {
        "eligible": True,
        "durable": True,
        "reason": "visual_feedback_event",
        "target_memory_type": "visual_feedback",
    }
    record["summary"] = _summary(record)
    record["tags"] = _tags(record)
    return _json_safe(record)


def build_eimemory_visual_feedback_params(record: Mapping[str, object]) -> dict[str, object]:
    """Build eimemory upsert params from a visual feedback record."""

    cleaned = _json_safe(dict(record))
    return {
        "text": str(cleaned.get("summary") or ""),
        "title": _title(cleaned),
        "memory_type": "visual_feedback",
        "source": str(cleaned.get("source") or SOURCE),
        "organ": "eye",
        "modality": "vision",
        "content": {
            "feedback_type": cleaned.get("feedback_type"),
            "subject": cleaned.get("subject", {}),
            "outcome": cleaned.get("outcome"),
            "error": cleaned.get("error"),
            "user_feedback": cleaned.get("user_feedback"),
            "label_correction": cleaned.get("label_correction", {}),
            "before_metrics": cleaned.get("before_metrics", {}),
            "after_metrics": cleaned.get("after_metrics", {}),
            "round_id": cleaned.get("round_id", ""),
            "session_id": cleaned.get("session_id", ""),
        },
        "meta": {
            "importance": cleaned.get("importance"),
            "timestamp_ms": cleaned.get("timestamp_ms"),
            "privacy": cleaned.get("privacy", {}),
            "record_type": "visual_feedback",
            "writeback": cleaned.get("writeback", {}),
        },
        "tags": list(cleaned.get("tags", [])),
    }


def build_eitraining_visual_feedback_trace(record: Mapping[str, object]) -> dict[str, object]:
    """Build a training trace from a visual feedback record."""

    cleaned = _json_safe(dict(record))
    return {
        "signal_type": "visual_feedback",
        "feedback_type": cleaned.get("feedback_type"),
        "subject": cleaned.get("subject", {}),
        "outcome": cleaned.get("outcome"),
        "error": cleaned.get("error"),
        "label_correction": cleaned.get("label_correction", {}),
        "before_metrics": cleaned.get("before_metrics", {}),
        "after_metrics": cleaned.get("after_metrics", {}),
        "round_id": cleaned.get("round_id", ""),
        "session_id": cleaned.get("session_id", ""),
        "importance": cleaned.get("importance"),
        "timestamp_ms": cleaned.get("timestamp_ms"),
        "privacy": cleaned.get("privacy", {}),
    }


def _resolve_outcome(
    *,
    feedback_type: str,
    outcome_map: Mapping[str, object],
    follow_map: Mapping[str, object],
    correction_map: Mapping[str, object],
) -> str:
    if feedback_type in {"identity_correction", "label_correction"} and correction_map:
        return "corrected"
    if feedback_type in {"target_lost", "target_missing"}:
        return "lost"
    success = _coerce_bool(_first_value(outcome_map, keys=("success", "ok")))
    if success is None:
        success = _coerce_bool(_first_value(follow_map, keys=("success", "ok")))
    if success is True:
        return "success"
    if success is False:
        return "failure"
    raw = _normalize_token(_first_text(outcome_map, follow_map, keys=("status", "outcome", "result", "reason")))
    if raw in {"success", "succeeded", "ok", "stable", "completed"}:
        return "success"
    if raw in {"lost", "target_lost", "missing"}:
        return "lost"
    if raw in {"failure", "failed", "error", "timeout", "action_slow"}:
        return "failure"
    return "observed"


def _split_metrics(
    follow_map: Mapping[str, object],
    outcome_map: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    metrics = _to_plain_dict(_first_value(follow_map, keys=("metrics",)) or _first_value(outcome_map, keys=("metrics",)))
    before: dict[str, object] = {}
    after: dict[str, object] = {}
    for key, value in metrics.items():
        if str(key).startswith("before_"):
            before[str(key)] = value
        elif str(key).startswith("after_"):
            after[str(key)] = value
    before.update(_to_plain_dict(_first_value(follow_map, keys=("before_metrics",)) or _first_value(outcome_map, keys=("before_metrics",))))
    after.update(_to_plain_dict(_first_value(follow_map, keys=("after_metrics",)) or _first_value(outcome_map, keys=("after_metrics",))))
    return before, after


def _importance(
    *,
    feedback_type: str,
    outcome: str,
    error: str,
    correction: Mapping[str, object],
) -> float:
    if feedback_type in {"identity_correction", "label_correction"} or correction:
        return 1.0
    if outcome == "failure" or error:
        return 0.85
    if outcome == "lost":
        return 0.75
    if outcome == "success":
        return 0.7
    return 0.5


def _privacy_result(privacy: Mapping[str, object]) -> dict[str, object]:
    redact = _coerce_bool(privacy.get("redact_image_paths")) is True
    return {
        "redacted": redact,
        "redact_image_paths": redact,
    }


def _redact_if_needed(value: object, privacy: Mapping[str, object]) -> object:
    if not isinstance(value, Mapping):
        return value
    redacted = {}
    for key, item in value.items():
        text_key = str(key)
        if privacy.get("redact_image_paths") and text_key in {"image_path", "snapshot_path", "frame_path", "imageUrl", "image_url"}:
            redacted[text_key] = "<redacted>"
        elif isinstance(item, Mapping):
            redacted[text_key] = _redact_if_needed(item, privacy)
        elif isinstance(item, list):
            redacted[text_key] = [_redact_if_needed(entry, privacy) for entry in item]
        else:
            redacted[text_key] = item
    return redacted


def _summary(record: Mapping[str, object]) -> str:
    subject = _to_plain_dict(record.get("subject"))
    label = _first_text(subject, keys=("label", "track_id", "person_id"), default="visual target")
    feedback_type = str(record.get("feedback_type") or "visual_feedback")
    outcome = str(record.get("outcome") or "observed")
    parts = [f"{feedback_type} for {label}: {outcome}"]
    error = str(record.get("error") or "").strip()
    if error:
        parts.append(error)
    correction = _to_plain_dict(record.get("label_correction"))
    correction_to = _first_text(correction, keys=("to", "label", "person_id"), default="")
    if correction_to:
        parts.append(f"corrected_to={correction_to}")
    feedback = str(record.get("user_feedback") or "").strip()
    if feedback:
        parts.append(f"feedback={feedback}")
    return "; ".join(parts)


def _tags(record: Mapping[str, object]) -> list[str]:
    tags = ["vision", "visual_feedback", str(record.get("feedback_type") or "feedback")]
    outcome = str(record.get("outcome") or "")
    if outcome:
        tags.append(outcome)
    if record.get("privacy", {}).get("redacted") if isinstance(record.get("privacy"), Mapping) else False:
        tags.append("privacy_redacted")
    return sorted(set(tags))


def _title(record: Mapping[str, object]) -> str:
    return f"Visual feedback: {record.get('feedback_type') or 'feedback'}"


def _to_plain_dict(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return _to_plain_dict(asdict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        mapped = to_dict()
        if isinstance(mapped, Mapping):
            return _to_plain_dict(mapped)
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


def _first_value(*maps: Mapping[str, object], keys: tuple[str, ...] | None = None) -> object | None:
    key_list = keys or ()
    for mapping in maps:
        if not isinstance(mapping, Mapping):
            continue
        for key in key_list:
            if key in mapping:
                return mapping.get(key)
    return None


def _first_text(*maps: Mapping[str, object], keys: tuple[str, ...], default: str = "") -> str:
    value = _first_value(*maps, keys=keys)
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _feedback_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    mapped = _to_plain_dict(value)
    return _first_text(mapped, keys=("text", "feedback", "message", "user_feedback"), default="")


def _normalize_token(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(".", "_").replace(" ", "_")


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "ok", "success", "succeeded"}:
            return True
        if normalized in {"false", "0", "no", "failed", "failure", "error"}:
            return False
    return None
