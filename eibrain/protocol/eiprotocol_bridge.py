"""Adapters from legacy eibrain protocol objects to eiprotocol envelopes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from eiprotocol import EventEnvelope, PolicyState, SourceRef, TargetRef

from .capabilities import (
    CapabilityManifest as LegacyCapabilityManifest,
    HeadBackend,
    HeadDevice,
    HeadHealth,
    HeadLimit,
)
from .head import (
    AudioTurn as LegacyAudioTurn,
    ExecutionOutcome as LegacyExecutionOutcome,
    HeadAction as LegacyHeadAction,
    VisionObservation as LegacyVisionObservation,
)


DEFAULT_EVENT_TIME = "1970-01-01T00:00:00.000Z"


def to_eiprotocol_event(
    message: object,
    *,
    event_id: str | None = None,
    request_id: str | None = None,
    sequence: int | None = None,
    time: str | None = None,
) -> EventEnvelope:
    """Convert a supported legacy protocol message into an eiprotocol event."""

    if isinstance(message, LegacyCapabilityManifest):
        return capability_manifest_to_eiprotocol_event(
            message,
            event_id=event_id,
            request_id=request_id,
            sequence=sequence,
            time=time,
        )
    if isinstance(message, LegacyAudioTurn):
        return audio_turn_to_eiprotocol_event(
            message,
            event_id=event_id,
            request_id=request_id,
            sequence=sequence,
            time=time,
        )
    if isinstance(message, LegacyVisionObservation):
        return vision_observation_to_eiprotocol_event(
            message,
            event_id=event_id,
            request_id=request_id,
            sequence=sequence,
            time=time,
        )
    if isinstance(message, LegacyHeadAction):
        return head_action_to_eiprotocol_event(
            message,
            event_id=event_id,
            request_id=request_id,
            sequence=sequence,
            time=time,
        )
    if isinstance(message, LegacyExecutionOutcome):
        return execution_outcome_to_eiprotocol_event(
            message,
            event_id=event_id,
            request_id=request_id,
            sequence=sequence,
            time=time,
        )

    raise TypeError(f"Unsupported eiprotocol bridge message: {type(message).__name__}")


def capability_manifest_to_eiprotocol_event(
    manifest: LegacyCapabilityManifest,
    *,
    event_id: str | None = None,
    request_id: str | None = None,
    sequence: int | None = None,
    time: str | None = None,
) -> EventEnvelope:
    """Wrap a legacy capability manifest in an eiprotocol manifest event."""

    content = {
        "manifestId": _first_text(manifest.node_id, manifest.trace_id, manifest.source, fallback="manifest"),
        "manifestVersion": manifest.protocol_version or "head.v1",
        "device": {
            "nodeId": manifest.node_id,
            "nodeRole": manifest.node_role,
            "source": manifest.source,
            "target": manifest.target,
            "timestampMs": manifest.timestamp_ms,
            "devices": [_device_to_capability_metadata(device) for device in manifest.devices],
        },
        "runtime": {},
        "transports": {},
        "modalities": {},
        "capabilities": [_device_to_capability(device) for device in manifest.devices],
        "backends": [_backend_to_capability(backend) for backend in manifest.backends],
        "health": _health_to_dict(manifest.health),
        "limits": {},
        "metadata": {
            **dict(manifest.metadata),
            "legacyCapabilities": list(manifest.capabilities),
        },
    }
    return _event(
        manifest,
        event_type="capability",
        name="ei.capability.manifest.report",
        content=content,
        priority="normal",
        event_id=_resolve_event_id(event_id, "capability_manifest", manifest.node_id, manifest.trace_id),
        request_id=request_id,
        sequence=sequence,
        time=time,
        source_device_id=manifest.node_id,
        round_scoped=False,
    )


def audio_turn_to_eiprotocol_event(
    turn: LegacyAudioTurn,
    *,
    event_id: str | None = None,
    request_id: str | None = None,
    sequence: int | None = None,
    time: str | None = None,
) -> EventEnvelope:
    """Wrap a legacy audio ASR turn in an eiprotocol dialogue event."""

    legacy_payload = dict(turn.payload)
    content = {
        "text": turn.text,
        "language": turn.language,
        "final": bool(turn.is_final),
        "confidence": turn.confidence,
        "startMs": turn.start_ms,
        "endMs": turn.end_ms,
        "audioLevel": turn.audio_level,
        "wakeWord": turn.wake_word,
        "asrBackend": _first_text(legacy_payload.get("asrBackend"), legacy_payload.get("asr_backend")),
        "timingsMs": _dict_from(legacy_payload.get("timingsMs") or legacy_payload.get("timings_ms")),
        "metadata": {
            "legacyPayload": legacy_payload,
            "observationType": turn.observation_type,
            "status": turn.status,
        },
    }
    return _event(
        turn,
        event_type="dialogue",
        name="ei.dialogue.asr.final" if turn.is_final else "ei.dialogue.asr.partial",
        content=content,
        priority="realtime",
        event_id=_resolve_event_id(event_id, "audio_turn", turn.trace_id, turn.text),
        request_id=request_id,
        sequence=sequence,
        time=time,
        source_device_id=turn.device_id,
        round_scoped=True,
    )


def vision_observation_to_eiprotocol_event(
    observation: LegacyVisionObservation,
    *,
    event_id: str | None = None,
    request_id: str | None = None,
    sequence: int | None = None,
    time: str | None = None,
) -> EventEnvelope:
    """Wrap a legacy vision observation in an eiprotocol vision-frame event."""

    legacy_payload = dict(observation.payload)
    content = {
        "frameId": observation.frame_id,
        "width": observation.width,
        "height": observation.height,
        "frameAgeMs": legacy_payload.get("frameAgeMs", legacy_payload.get("frame_age_ms")),
        "backend": _first_text(legacy_payload.get("backend"), legacy_payload.get("vision_backend")),
        "detections": [dict(item) for item in observation.detections],
        "latencyMs": _dict_from(legacy_payload.get("latencyMs") or legacy_payload.get("latency_ms")),
        "imageUrl": observation.image_url,
        "status": observation.status,
        "trackedTarget": dict(observation.tracked_target),
        "metadata": {
            "legacyPayload": legacy_payload,
            "observationType": observation.observation_type,
        },
    }
    return _event(
        observation,
        event_type="observation",
        name="ei.observation.vision.frame",
        content=content,
        priority="realtime",
        event_id=_resolve_event_id(event_id, "vision_observation", observation.frame_id, observation.trace_id),
        request_id=request_id,
        sequence=sequence,
        time=time,
        source_device_id=observation.device_id,
        round_scoped=False,
    )


def head_action_to_eiprotocol_event(
    action: LegacyHeadAction,
    *,
    event_id: str | None = None,
    request_id: str | None = None,
    sequence: int | None = None,
    time: str | None = None,
) -> EventEnvelope:
    """Wrap a legacy head action in an eiprotocol side-effecting action event."""

    legacy_payload = dict(action.payload)
    params = dict(action.params)
    resolved_event_id = _resolve_event_id(event_id, "head_action", action.action_id, action.trace_id)
    risk_level = _first_text(
        params.get("riskLevel"),
        params.get("risk_level"),
        legacy_payload.get("riskLevel"),
        legacy_payload.get("risk_level"),
        fallback="L1",
    )
    idempotency_key = _first_text(
        params.get("idempotencyKey"),
        params.get("idempotency_key"),
        legacy_payload.get("idempotencyKey"),
        legacy_payload.get("idempotency_key"),
        action.action_id,
        fallback=resolved_event_id,
    )
    content = {
        "actionId": action.action_id,
        "actionType": action.action_type,
        "target": _first_text(action.device_id, action.target),
        "params": params,
        "riskLevel": risk_level,
        "timeline": _list_of_dicts(legacy_payload.get("timeline")),
        "requiresPolicy": bool(
            params.get("requiresPolicy")
            or params.get("requires_policy")
            or legacy_payload.get("requiresPolicy")
            or legacy_payload.get("requires_policy")
            or False
        ),
        "metadata": {
            "legacyPayload": legacy_payload,
            "priority": action.priority,
        },
        "idempotencyKey": idempotency_key,
    }
    event = _event(
        action,
        event_type="action",
        name="ei.action.request",
        content=content,
        priority="high",
        event_id=resolved_event_id,
        request_id=request_id,
        sequence=sequence,
        time=time,
        source_device_id="",
        round_scoped=True,
    )
    event.policy = PolicyState(decision="not_required", risk_level=risk_level)
    return event


def execution_outcome_to_eiprotocol_event(
    outcome: LegacyExecutionOutcome,
    *,
    event_id: str | None = None,
    request_id: str | None = None,
    sequence: int | None = None,
    time: str | None = None,
) -> EventEnvelope:
    """Wrap a legacy execution outcome in an eiprotocol outcome event."""

    details = dict(outcome.details)
    content = {
        "outcomeId": _first_text(
            details.get("outcomeId"),
            details.get("outcome_id"),
            f"outcome-{outcome.action_id}" if outcome.action_id else "",
            outcome.trace_id,
            fallback="outcome",
        ),
        "actionId": outcome.action_id,
        "actionType": outcome.action_type,
        "success": bool(outcome.success),
        "status": outcome.status,
        "latencyMs": outcome.latency_ms,
        "didWhat": _list_of_text(details.get("didWhat") or details.get("did_what")),
        "errors": _list_of_dicts(details.get("errors")),
        "details": details,
        "deviceId": outcome.device_id,
    }
    return _event(
        outcome,
        event_type="outcome",
        name="ei.outcome.execution",
        content=content,
        priority="normal",
        event_id=_resolve_event_id(event_id, "execution_outcome", outcome.action_id, outcome.trace_id),
        request_id=request_id,
        sequence=sequence,
        time=time,
        source_device_id=outcome.device_id,
        round_scoped=True,
    )


def _event(
    message: object,
    *,
    event_type: str,
    name: str,
    content: dict[str, Any],
    priority: str,
    event_id: str,
    request_id: str | None,
    sequence: int | None,
    time: str | None,
    source_device_id: str,
    round_scoped: bool,
) -> EventEnvelope:
    resolved_request_id = _first_text(request_id, getattr(message, "trace_id", ""), event_id)
    session_id = _first_text(getattr(message, "session_id", ""))
    return EventEnvelope(
        event_id=event_id,
        event_type=event_type,
        name=name,
        time=_resolve_time(message, time),
        sequence=_resolve_sequence(message, sequence),
        request_id=resolved_request_id,
        session_id=session_id,
        round_id=_resolve_round_id(message, event_id) if round_scoped else "",
        trace_id=_first_text(getattr(message, "trace_id", "")),
        source=_source_ref(_first_text(getattr(message, "source", "")), source_device_id),
        target=_target_ref(_first_text(getattr(message, "target", ""))),
        priority=priority,
        content=content,
        policy=PolicyState(),
    )


def _resolve_event_id(explicit: str | None, prefix: str, *candidates: object) -> str:
    if explicit:
        return explicit
    token = _first_text(*candidates, fallback=prefix)
    return f"evt_{prefix}_{_stable_token(token)}"


def _resolve_sequence(message: object, explicit: int | None) -> int:
    if explicit is not None:
        return int(explicit)
    sequence = getattr(message, "sequence", None)
    return int(sequence) if sequence is not None else 0


def _resolve_time(message: object, explicit: str | None) -> str:
    if explicit:
        return explicit
    timestamp_ms = getattr(message, "timestamp_ms", None)
    if timestamp_ms is None:
        return DEFAULT_EVENT_TIME
    instant = datetime.fromtimestamp(float(timestamp_ms) / 1000.0, tz=UTC)
    return instant.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _resolve_round_id(message: object, event_id: str) -> str:
    return _first_text(
        getattr(message, "round_id", ""),
        getattr(message, "session_id", ""),
        getattr(message, "trace_id", ""),
        event_id,
    )


def _source_ref(source: str, device_id: str = "") -> SourceRef:
    domain, instance_id, source_device_id = _split_ref(source)
    return SourceRef(
        domain=domain or "unknown",
        instance_id=instance_id,
        device_id=_first_text(device_id, source_device_id),
    )


def _target_ref(target: str) -> TargetRef | None:
    if not target:
        return None
    domain, instance_id, _ = _split_ref(target)
    if not domain:
        return None
    return TargetRef(domain=domain, instance_id=instance_id)


def _split_ref(value: str) -> tuple[str, str, str]:
    parts = [part for part in str(value).split(".") if part]
    domain = parts[0] if parts else ""
    instance_id = parts[1] if len(parts) > 1 else ""
    device_id = ".".join(parts[2:]) if len(parts) > 2 else ""
    return domain, instance_id, device_id


def _device_to_capability(device: HeadDevice) -> dict[str, Any]:
    return {
        "capabilityId": _first_text(device.device_id, device.kind, fallback="device"),
        "kind": device.kind,
        "provider": "",
        "model": "",
        "version": "",
        "devicePath": device.path,
        "actions": list(device.capabilities),
        "status": device.health.status,
        "limits": _limits_by_name(device.limits),
        "metadata": {
            **dict(device.metadata),
            "name": device.name,
            "enabled": device.enabled,
            "health": _health_to_dict(device.health),
        },
    }


def _backend_to_capability(backend: HeadBackend) -> dict[str, Any]:
    return {
        "capabilityId": _first_text(backend.backend_id, backend.kind, fallback="backend"),
        "kind": backend.kind,
        "provider": backend.provider,
        "model": backend.model,
        "version": backend.version,
        "devicePath": "",
        "actions": list(backend.capabilities),
        "status": backend.health.status,
        "limits": _limits_by_name(backend.limits),
        "metadata": {
            **dict(backend.metadata),
            "enabled": backend.enabled,
            "health": _health_to_dict(backend.health),
        },
    }


def _device_to_capability_metadata(device: HeadDevice) -> dict[str, Any]:
    return {
        "deviceId": device.device_id,
        "kind": device.kind,
        "name": device.name,
        "path": device.path,
        "enabled": device.enabled,
        "capabilities": list(device.capabilities),
        "limits": [limit.to_dict() for limit in device.limits],
        "health": _health_to_dict(device.health),
        "metadata": dict(device.metadata),
    }


def _limits_by_name(limits: list[HeadLimit]) -> dict[str, Any]:
    indexed: dict[str, Any] = {}
    for index, limit in enumerate(limits):
        key = _first_text(limit.name, f"limit_{index}")
        indexed[key] = limit.to_dict()
    return indexed


def _health_to_dict(health: HeadHealth) -> dict[str, Any]:
    return {
        "status": health.status,
        "message": health.message,
        "checkedAtMs": health.checked_at_ms,
        "metrics": dict(health.metrics),
    }


def _dict_from(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _list_of_dicts(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _list_of_text(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _first_text(*values: object, fallback: str = "") -> str:
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    return fallback


def _stable_token(value: object) -> str:
    token = str(value).strip().replace(" ", "_")
    return token or "event"


__all__ = [
    "DEFAULT_EVENT_TIME",
    "audio_turn_to_eiprotocol_event",
    "capability_manifest_to_eiprotocol_event",
    "execution_outcome_to_eiprotocol_event",
    "head_action_to_eiprotocol_event",
    "to_eiprotocol_event",
    "vision_observation_to_eiprotocol_event",
]
