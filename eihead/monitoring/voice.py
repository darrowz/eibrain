"""Voice diagnostics helpers for the eihead native monitor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

from eihead.protocol import serialize_message


VOICE_REALTIME_SCHEMA = "eihead.monitor.voice_realtime.v1"
VOICE_RUNTIME_ATTRS = (
    "voice_realtime",
    "voice_status",
    "latest_voice_realtime",
    "latest_voice_status",
)


def build_voice_diagnostics_from_app(app: Any, timestamp: float) -> dict[str, Any]:
    """Read voice diagnostics from runtime hooks or a body snapshot."""

    first_non_wired_payload: dict[str, Any] | None = None
    for attr_name in VOICE_RUNTIME_ATTRS:
        if not hasattr(app, attr_name):
            continue
        source = getattr(app, attr_name)
        raw_payload = _resolve_voice_candidate(source() if callable(source) else source)
        payload = _build_voice_payload(
            raw_payload,
            timestamp=timestamp,
            source=attr_name,
            wired=raw_payload is not None,
        )
        if payload.get("wired") is True:
            return payload
        if first_non_wired_payload is None:
            first_non_wired_payload = payload

    snapshot_payload = _voice_payload_from_snapshot(app)
    if snapshot_payload is not None:
        payload = _build_voice_payload(
            snapshot_payload,
            timestamp=timestamp,
            source="snapshot",
            wired=True,
        )
        if payload.get("wired") is True:
            return payload
        if first_non_wired_payload is None:
            first_non_wired_payload = payload

    if first_non_wired_payload is not None:
        return first_non_wired_payload

    return _build_voice_payload(None, timestamp=timestamp, source=None, wired=False)


def _build_voice_payload(
    observation: Any,
    *,
    timestamp: float,
    source: str | None,
    wired: bool | None,
) -> dict[str, Any]:
    resolved = _resolve_voice_candidate(observation)
    data = _mapping_payload(resolved) if resolved is not None else None
    ear = _normalize_ear(_mapping_from_keys(data, "ear"))
    mouth = _normalize_mouth(_mapping_from_keys(data, "mouth"))
    dialogue_source = _dialogue_mapping(data)
    dialogue = _normalize_dialogue(dialogue_source, root=data)
    latency = _latency_payload(data, dialogue_source)
    bottleneck = _bottleneck_payload(data, dialogue_source)
    last_turn = _last_turn_payload(data, dialogue_source)
    status, derived_wired, not_wired = _voice_overall_status(
        ear=ear,
        mouth=mouth,
        dialogue=dialogue,
        last_turn=last_turn,
        latency=latency,
    )
    is_wired = derived_wired if wired is None else bool(wired and derived_wired)
    readiness_message = _voice_readiness_message(data, ear=ear, mouth=mouth, dialogue=dialogue, status=status)
    payload: dict[str, Any] = {
        "schema": VOICE_REALTIME_SCHEMA,
        "runtime": "eihead",
        "status": status,
        "wired": is_wired,
        "source": source,
        "channel": "voice.realtime",
        "aliases": ["audio.realtime"],
        "captured_at_ts": float(timestamp),
        "observation": data,
        "ear": ear,
        "mouth": mouth,
        "dialogue": dialogue,
        "latency": latency,
        "bottleneck": bottleneck,
        "last_turn": last_turn,
        "not_wired": bool(not_wired),
        "readiness_message": readiness_message,
    }
    if not is_wired and status == "not_wired" and not readiness_message:
        payload["readiness_message"] = "runtime app does not expose voice diagnostics"
    return payload


def _voice_payload_from_snapshot(app: Any) -> dict[str, Any] | None:
    snapshot_fn = getattr(app, "snapshot", None)
    if not callable(snapshot_fn):
        return None
    snapshot = snapshot_fn()
    if not isinstance(snapshot, Mapping):
        return None

    payload: dict[str, Any] = {}
    voice_dialogue = _mapping_from_keys(snapshot, "voice_dialogue", "dialogue")
    if voice_dialogue is not None:
        payload["voice_dialogue"] = voice_dialogue
    organs = snapshot.get("organs")
    if isinstance(organs, Mapping):
        ear = _mapping_from_keys(organs, "ear")
        mouth = _mapping_from_keys(organs, "mouth")
        if ear is not None:
            payload["ear"] = ear
        if mouth is not None:
            payload["mouth"] = mouth
    return payload or None


def _normalize_ear(raw: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    capture = _subfunction(raw, "capture")
    asr = _subfunction(raw, "asr")
    provider = _first_text(
        raw.get("provider"),
        _details_value(asr, "provider"),
        _details_value(asr, "backend"),
    )
    status = _status_text(raw, fallback=capture or asr)
    readiness = _first_text(
        raw.get("readiness_message"),
        raw.get("message"),
        _details_value(asr, "reason"),
        _details_value(capture, "reason"),
        _details_value(asr, "status"),
        _details_value(capture, "status"),
    )
    return {
        "status": status,
        "health": _text_or_none(raw.get("health")),
        "state": _classify_component_state(raw, fallback_status=status, role="ear"),
        "provider": provider,
        "live_probe_skipped": _truthy(raw.get("live_probe_skipped")) or _details_truthy(asr, "live_probe_skipped"),
        "readiness_message": readiness,
        "capture": capture,
        "asr": asr,
    }


def _normalize_mouth(raw: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    playback = _subfunction(raw, "tts_playback")
    plan = _subfunction(raw, "tts_plan")
    backend = _first_text(
        raw.get("backend"),
        raw.get("provider"),
        _details_value(playback, "backend"),
        _details_value(playback, "provider"),
        _details_value(plan, "backend"),
        _details_value(plan, "provider"),
    )
    status = _status_text(raw, fallback=playback or plan)
    readiness = _first_text(
        raw.get("readiness_message"),
        raw.get("message"),
        _details_value(playback, "reason"),
        _details_value(plan, "reason"),
        _details_value(playback, "status"),
        _details_value(plan, "status"),
    )
    return {
        "status": status,
        "health": _text_or_none(raw.get("health")),
        "state": _classify_component_state(raw, fallback_status=status, role="mouth"),
        "backend": backend,
        "model": _first_text(raw.get("model"), _details_value(playback, "model"), _details_value(plan, "model")),
        "voice_id": _first_text(
            raw.get("voice_id"),
            _details_value(playback, "voice_id"),
            _details_value(plan, "voice_id"),
        ),
        "text_preview": _first_text(raw.get("text_preview"), _details_value(playback, "text_preview")),
        "readiness_message": readiness,
        "tts_playback": playback,
        "tts_plan": plan,
    }


def _normalize_dialogue(raw: Mapping[str, Any] | None, *, root: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if raw is None and root is None:
        return None
    mapping = raw or {}
    phase = _first_text(mapping.get("phase"), root.get("phase") if root else None)
    last_status = _first_text(mapping.get("last_status"), root.get("last_status") if root else None)
    status = last_status or phase or _status_text(mapping)
    readiness = _first_text(
        mapping.get("readiness_message"),
        mapping.get("message"),
        root.get("readiness_message") if root else None,
        root.get("message") if root else None,
    )
    return {
        "phase": phase,
        "last_status": last_status,
        "state": _classify_component_state(mapping, fallback_status=status, role="dialogue"),
        "enabled": _truthy(mapping.get("enabled")),
        "running": _truthy(mapping.get("running")),
        "last_transcript": _first_text(mapping.get("last_transcript"), root.get("last_transcript") if root else None),
        "last_reply": _first_text(mapping.get("last_reply"), root.get("last_reply") if root else None),
        "readiness_message": readiness,
    }


def _latency_payload(data: Mapping[str, Any] | None, dialogue: Mapping[str, Any] | None) -> dict[str, Any]:
    stage_latency = _mapping_from_keys(dialogue, "last_stage_latency_ms") or _mapping_from_keys(data, "last_stage_latency_ms")
    stage_latency_ms: dict[str, float] = {}
    if stage_latency is not None:
        for key, value in stage_latency.items():
            number = _float_or_none(value)
            if number is not None:
                stage_latency_ms[str(key)] = number
    latency_seconds = _mapping_from_keys(dialogue, "last_latency_s") or _mapping_from_keys(data, "last_latency_s")
    latency_s: dict[str, float] = {}
    if latency_seconds is not None:
        for key, value in latency_seconds.items():
            number = _float_or_none(value)
            if number is not None:
                latency_s[str(key)] = number
                stage_latency_ms.setdefault(str(key), round(number * 1000.0, 3))
    total_ms = _float_or_none(_first_value(data, "last_total_latency_ms", "total_latency_ms"))
    if total_ms is None and stage_latency_ms:
        total_ms = round(sum(stage_latency_ms.values()), 3)
    return {
        "total_ms": total_ms,
        "stage_latency_ms": stage_latency_ms,
        "stage_latency_s": latency_s,
    }


def _bottleneck_payload(data: Mapping[str, Any] | None, dialogue: Mapping[str, Any] | None) -> dict[str, Any] | None:
    stage = _first_text(
        _first_value(dialogue, "last_bottleneck_stage"),
        _first_value(data, "last_bottleneck_stage"),
    )
    latency_ms = _float_or_none(_first_value(dialogue, "last_bottleneck_ms"))
    if latency_ms is None:
        latency_ms = _float_or_none(_first_value(data, "last_bottleneck_ms"))
    if not stage and latency_ms is None:
        return None
    return {
        "stage": stage or None,
        "latency_ms": latency_ms,
    }


def _last_turn_payload(data: Mapping[str, Any] | None, dialogue: Mapping[str, Any] | None) -> dict[str, Any] | None:
    return (
        _mapping_from_keys(data, "last_turn")
        or _mapping_from_keys(dialogue, "last_completed_turn")
        or _mapping_from_keys(dialogue, "last_turn")
    )


def _voice_overall_status(
    *,
    ear: Mapping[str, Any] | None,
    mouth: Mapping[str, Any] | None,
    dialogue: Mapping[str, Any] | None,
    last_turn: Mapping[str, Any] | None,
    latency: Mapping[str, Any] | None,
) -> tuple[str, bool, bool]:
    states = [
        str(component.get("state", "") or "")
        for component in (ear, mouth, dialogue)
        if isinstance(component, Mapping)
    ]
    has_signal = bool(states or last_turn or (latency and latency.get("stage_latency_ms")))
    if not has_signal:
        return "not_wired", False, True
    if any(state == "wired" for state in states):
        if any(state in {"degraded", "not_wired"} for state in states):
            return "degraded", False, False
        return "wired", True, False
    if any(state == "degraded" for state in states):
        return "degraded", False, False
    if any(state == "not_wired" for state in states):
        return "not_wired", False, True
    return "unknown", False, False


def _voice_readiness_message(
    data: Mapping[str, Any] | None,
    *,
    ear: Mapping[str, Any] | None,
    mouth: Mapping[str, Any] | None,
    dialogue: Mapping[str, Any] | None,
    status: str,
) -> str:
    explicit = _first_text(
        _first_value(data, "readiness_message"),
        _first_value(data, "message"),
    )
    if explicit:
        return explicit
    messages = []
    for name, component in (("ear", ear), ("mouth", mouth), ("dialogue", dialogue)):
        if not isinstance(component, Mapping):
            continue
        text = _first_text(
            component.get("readiness_message"),
            component.get("message"),
            component.get("status"),
        )
        if text and text not in messages:
            messages.append(f"{name}: {text}")
    if messages:
        return "; ".join(messages)
    if status == "not_wired":
        return "voice diagnostics are not wired"
    if status == "degraded":
        return "voice diagnostics are degraded"
    if status == "unknown":
        return "voice diagnostics are present but incomplete"
    return ""


def _resolve_voice_candidate(payload: Any, *, seen: set[int] | None = None) -> Any:
    if payload is None:
        return None
    seen = seen or set()
    candidate_id = id(payload)
    if candidate_id in seen:
        return payload
    seen.add(candidate_id)

    latest_status = getattr(payload, "latest_status", None)
    if latest_status is not None:
        resolved = _resolve_voice_candidate(latest_status, seen=seen)
        if resolved is not None:
            return resolved

    for method_name in ("status", "poll"):
        method = getattr(payload, method_name, None)
        if not callable(method):
            continue
        try:
            resolved = _resolve_voice_candidate(method(), seen=seen)
        except TypeError:
            continue
        if resolved is not None:
            return resolved
    return payload


def _mapping_payload(payload: Any) -> dict[str, Any] | None:
    if payload is None:
        return None
    if isinstance(payload, Mapping):
        return _json_mapping(payload)
    if hasattr(payload, "to_dict") and callable(payload.to_dict):
        data = payload.to_dict()
        if isinstance(data, Mapping):
            return _json_mapping(data)
    if is_dataclass(payload):
        return _json_mapping(asdict(payload))
    try:
        serialized = serialize_message(payload)
    except TypeError:
        return None
    return _json_mapping(serialized) if isinstance(serialized, Mapping) else None


def _json_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_ready(value) for key, value in mapping.items()}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _json_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        payload = value.to_dict()
        if isinstance(payload, Mapping):
            return _json_mapping(payload)
    return str(value)


def _mapping_from_keys(mapping: Mapping[str, Any] | None, *keys: str) -> dict[str, Any] | None:
    if not isinstance(mapping, Mapping):
        return None
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, Mapping):
            return _json_mapping(value)
    return None


def _dialogue_mapping(mapping: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(mapping, Mapping):
        return None
    if isinstance(mapping.get("dialogue"), Mapping):
        return _json_mapping(mapping["dialogue"])
    if isinstance(mapping.get("voice_dialogue"), Mapping):
        return _json_mapping(mapping["voice_dialogue"])
    dialogue_keys = {
        "enabled",
        "running",
        "phase",
        "last_status",
        "last_transcript",
        "last_reply",
        "last_completed_turn",
        "last_stage_latency_ms",
        "last_latency_s",
        "last_bottleneck_stage",
        "last_bottleneck_ms",
    }
    if any(key in mapping for key in dialogue_keys):
        return _json_mapping({key: mapping[key] for key in dialogue_keys if key in mapping})
    return None


def _subfunction(raw: Mapping[str, Any], name: str) -> dict[str, Any] | None:
    subfunctions = raw.get("subfunctions")
    if isinstance(subfunctions, Mapping) and isinstance(subfunctions.get(name), Mapping):
        return _json_mapping(subfunctions[name])
    if isinstance(raw.get(name), Mapping):
        return _json_mapping(raw[name])
    return None


def _details_value(subfunction: Mapping[str, Any] | None, key: str) -> Any:
    if not isinstance(subfunction, Mapping):
        return None
    details = subfunction.get("details")
    if isinstance(details, Mapping):
        return details.get(key)
    return None


def _details_truthy(subfunction: Mapping[str, Any] | None, key: str) -> bool:
    value = _details_value(subfunction, key)
    return _truthy(value)


def _status_text(raw: Mapping[str, Any], *, fallback: Mapping[str, Any] | None = None) -> str:
    return _first_text(
        raw.get("status"),
        raw.get("health"),
        fallback.get("status") if fallback else None,
        fallback.get("health") if fallback else None,
    )


def _classify_component_state(raw: Mapping[str, Any], *, fallback_status: str, role: str) -> str:
    status = _normalized_text(fallback_status)
    if not status:
        status = _normalized_text(raw.get("status") or raw.get("health"))
    health = _normalized_text(raw.get("health"))
    data_status = _normalized_text(raw.get("data_status"))
    backend = _normalized_text(raw.get("backend") or raw.get("provider"))
    if _truthy(raw.get("not_wired")) or status in {"not_wired", "offline", "missing", "unavailable", "disabled"}:
        return "not_wired"
    if status == "noop" or backend == "noop":
        return "not_wired"
    if health in {"not_wired", "offline", "missing", "unavailable", "disabled"}:
        return "not_wired"
    if health in {"degraded", "error", "failed", "unhealthy"} or data_status in {"compat", "fallback"}:
        return "degraded"
    if _truthy(raw.get("live_probe_skipped")) or status in {
        "waiting",
        "waiting_for_data",
        "warming_up",
        "pending",
        "no_data",
    }:
        return "degraded"
    if status in {"degraded", "error", "failed", "unhealthy"}:
        return "degraded"
    if role == "dialogue" and status in {"idle", "waiting_for_voice", "sleeping", "dormant"}:
        return "unknown"
    if status in {"ok", "healthy", "ready", "running", "active", "listening", "thinking", "speaking", "completed"}:
        return "wired"
    return "unknown"


def _first_value(mapping: Mapping[str, Any] | None, *keys: str) -> Any:
    if not isinstance(mapping, Mapping):
        return None
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _first_text(*values: Any) -> str:
    for value in values:
        text = _text_or_none(value)
        if text:
            return text
    return ""


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalized_text(value: Any) -> str:
    text = _text_or_none(value)
    return text.lower() if text is not None else ""


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


__all__ = [
    "VOICE_REALTIME_SCHEMA",
    "build_voice_diagnostics_from_app",
]
