"""Factory helpers for constructing valid eiprotocol event envelopes."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from typing import Any
import uuid

from .models import (
    AudioTurn,
    Detection,
    EventEnvelope,
    ExecutionOutcome,
    PolicyState,
    RealtimeVisionObservation,
    SourceRef,
    TargetRef,
)
from .validation import ValidationIssue, validate_event_strict

try:
    from .catalog import get_event_definition
except ImportError:  # pragma: no cover - supports partial protocol checkouts.
    get_event_definition = None  # type: ignore[assignment]


Clock = Callable[[], datetime | str]
IdSuffixFactory = Callable[[], str]
SourceLike = SourceRef | Mapping[str, Any]
TargetLike = TargetRef | Mapping[str, Any] | None

_ROUND_SCOPED_EVENT_TYPES = {"dialogue", "action", "memory", "outcome", "training"}
_EVENT_DEFAULTS: dict[str, tuple[str, bool]] = {
    "ei.control.hello": ("control", False),
    "ei.control.ping": ("control", False),
    "ei.control.pong": ("control", False),
    "ei.control.resume": ("control", False),
    "ei.control.ack": ("control", False),
    "ei.control.error": ("control", False),
    "ei.capability.manifest.report": ("capability", False),
    "ei.observation.audio.chunk": ("observation", False),
    "ei.observation.vision.frame": ("observation", False),
    "ei.dialogue.asr.partial": ("dialogue", True),
    "ei.dialogue.asr.final": ("dialogue", True),
    "ei.dialogue.agent.delta": ("dialogue", True),
    "ei.dialogue.agent.final": ("dialogue", True),
    "ei.dialogue.tts.delta": ("dialogue", True),
    "ei.dialogue.tts.final": ("dialogue", True),
    "ei.dialogue.interrupt.requested": ("dialogue", True),
    "ei.action.request": ("action", True),
    "ei.action.dispatch": ("action", True),
    "ei.action.progress": ("action", True),
    "ei.action.complete": ("action", True),
    "ei.action.emergency.stop": ("action", True),
    "ei.policy.decision": ("policy", True),
    "ei.memory.recall.request": ("memory", True),
    "ei.memory.recall.result": ("memory", True),
    "ei.memory.write.proposed": ("memory", True),
    "ei.memory.write.committed": ("memory", True),
    "ei.outcome.execution": ("outcome", True),
    "ei.outcome.user.feedback": ("outcome", True),
    "ei.training.signal": ("training", True),
}


def _default_clock() -> datetime:
    return datetime.now(timezone.utc)


def _default_id_suffix() -> str:
    return uuid.uuid4().hex


def _event_defaults(name: str, definition: Any) -> tuple[str | None, bool | None]:
    if definition is not None:
        return str(definition.event_type), bool(definition.round_scoped)
    defaults = _EVENT_DEFAULTS.get(name)
    if defaults is None:
        return None, None
    return defaults


class EventIdFactory:
    """Generate protocol-prefixed IDs and timestamps for event builders."""

    def __init__(self, *, clock: Clock | None = None, id_factory: IdSuffixFactory | None = None) -> None:
        self._clock = clock or _default_clock
        self._id_factory = id_factory or _default_id_suffix

    def id(self, prefix: str) -> str:
        return f"{prefix}_{self._id_factory()}"

    def event_id(self) -> str:
        return self.id("evt")

    def request_id(self) -> str:
        return self.id("req")

    def round_id(self) -> str:
        return self.id("rnd")

    def trace_id(self) -> str:
        return self.id("trc")

    def evt(self) -> str:
        return self.event_id()

    def req(self) -> str:
        return self.request_id()

    def rnd(self) -> str:
        return self.round_id()

    def trc(self) -> str:
        return self.trace_id()

    def time(self) -> str:
        value = self._clock()
        if isinstance(value, str):
            return value
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat(timespec="milliseconds")


EventIds = EventIdFactory


def build_event(
    *,
    name: str,
    source: SourceLike,
    content: Mapping[str, Any] | None = None,
    event_type: str | None = None,
    ids: EventIdFactory | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    time: str | None = None,
    sequence: int = 1,
    session_id: str = "",
    round_id: str | None = None,
    correlation_id: str = "",
    causation_id: str = "",
    trace_id: str = "",
    target: TargetLike = None,
    priority: str = "normal",
    ttl_ms: int | None = None,
    mode: Mapping[str, Any] | None = None,
    policy: PolicyState | Mapping[str, Any] | None = None,
    extensions: Mapping[str, Any] | None = None,
    round_scoped: bool | None = None,
) -> EventEnvelope:
    """Build and validate an EventEnvelope with safe protocol defaults."""

    id_factory = ids or EventIdFactory()
    definition = get_event_definition(name) if get_event_definition is not None else None
    default_event_type, default_round_scoped = _event_defaults(name, definition)
    resolved_event_type = event_type or default_event_type
    if not resolved_event_type:
        raise ValueError("event_type is required for unknown eiprotocol event names")
    if int(sequence) < 1:
        raise ValueError("sequence must be >= 1")

    is_round_scoped = round_scoped
    if is_round_scoped is None:
        is_round_scoped = default_round_scoped if default_round_scoped is not None else resolved_event_type in _ROUND_SCOPED_EVENT_TYPES

    resolved_event_id = event_id or id_factory.event_id()
    resolved_request_id = request_id or id_factory.request_id()
    resolved_time = time or id_factory.time()
    resolved_round_id = round_id or ""
    if is_round_scoped and not resolved_round_id:
        resolved_round_id = id_factory.round_id()

    event = EventEnvelope(
        event_id=resolved_event_id,
        event_type=resolved_event_type,
        name=name,
        time=resolved_time,
        sequence=int(sequence),
        request_id=resolved_request_id,
        session_id=session_id,
        round_id=resolved_round_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=trace_id,
        source=_source_ref(source),
        target=_target_ref(target),
        priority=priority,
        ttl_ms=ttl_ms,
        mode=dict(mode or {}),
        content=dict(content or {}),
        policy=_policy_state(policy),
        extensions=dict(extensions or {}),
    )
    _raise_if_invalid(event)
    return event


def build_action_request_event(
    *,
    source: SourceLike,
    action_id: str,
    action_type: str,
    target: str,
    params: Mapping[str, Any] | None = None,
    risk_level: str = "L1",
    idempotency_key: str | None = None,
    ids: EventIdFactory | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    time: str | None = None,
    sequence: int = 1,
    session_id: str = "",
    round_id: str | None = None,
    correlation_id: str = "",
    causation_id: str = "",
    trace_id: str = "",
    target_ref: TargetLike = None,
    ttl_ms: int | None = None,
    mode: Mapping[str, Any] | None = None,
    policy: PolicyState | Mapping[str, Any] | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> EventEnvelope:
    content = {
        "actionId": action_id,
        "actionType": action_type,
        "target": target,
        "params": dict(params or {}),
        "riskLevel": risk_level,
        "idempotencyKey": idempotency_key or action_id,
    }
    action_policy = policy if policy is not None else PolicyState(decision="not_required", risk_level=risk_level)
    return build_event(
        ids=ids,
        name="ei.action.request",
        event_type="action",
        source=source,
        target=target_ref,
        content=content,
        event_id=event_id,
        request_id=request_id,
        time=time,
        sequence=sequence,
        session_id=session_id,
        round_id=round_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=trace_id,
        priority="high",
        ttl_ms=ttl_ms,
        mode=mode,
        policy=action_policy,
        extensions=extensions,
        round_scoped=True,
    )


def build_asr_event(
    *,
    source: SourceLike,
    text: str,
    final: bool,
    language: str = "und",
    confidence: float | None = None,
    start_ms: int | None = None,
    end_ms: int | None = None,
    audio_level: float | None = None,
    wake_word: str = "",
    asr_backend: str = "",
    timings_ms: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    ids: EventIdFactory | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    time: str | None = None,
    sequence: int = 1,
    session_id: str = "",
    round_id: str | None = None,
    correlation_id: str = "",
    causation_id: str = "",
    trace_id: str = "",
    target: TargetLike = None,
    ttl_ms: int | None = None,
    mode: Mapping[str, Any] | None = None,
    policy: PolicyState | Mapping[str, Any] | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> EventEnvelope:
    audio = AudioTurn(
        text=text,
        language=language,
        final=final,
        confidence=confidence,
        start_ms=start_ms,
        end_ms=end_ms,
        audio_level=audio_level,
        wake_word=wake_word,
        asr_backend=asr_backend,
        timings_ms=dict(timings_ms or {}),
        metadata=dict(metadata or {}),
    )
    return build_event(
        ids=ids,
        name="ei.dialogue.asr.final" if final else "ei.dialogue.asr.partial",
        event_type="dialogue",
        source=source,
        target=target,
        content=audio.to_content(),
        event_id=event_id,
        request_id=request_id,
        time=time,
        sequence=sequence,
        session_id=session_id,
        round_id=round_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=trace_id,
        priority="realtime",
        ttl_ms=ttl_ms,
        mode=mode,
        policy=policy,
        extensions=extensions,
        round_scoped=True,
    )


def build_vision_frame_event(
    *,
    source: SourceLike,
    frame_id: str,
    width: int | None = None,
    height: int | None = None,
    frame_age_ms: float | None = None,
    backend: str = "",
    detections: Iterable[Detection | Mapping[str, Any]] | None = None,
    latency_ms: Mapping[str, Any] | None = None,
    image_url: str = "",
    status: str = "ok",
    metadata: Mapping[str, Any] | None = None,
    ids: EventIdFactory | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    time: str | None = None,
    sequence: int = 1,
    session_id: str = "",
    round_id: str | None = None,
    correlation_id: str = "",
    causation_id: str = "",
    trace_id: str = "",
    target: TargetLike = None,
    ttl_ms: int | None = None,
    mode: Mapping[str, Any] | None = None,
    policy: PolicyState | Mapping[str, Any] | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> EventEnvelope:
    observation = RealtimeVisionObservation(
        frame_id=frame_id,
        width=width,
        height=height,
        frame_age_ms=frame_age_ms,
        backend=backend,
        detections=[_detection(item) for item in detections or ()],
        latency_ms=dict(latency_ms or {}),
        image_url=image_url,
        status=status,
        metadata=dict(metadata or {}),
    )
    return build_event(
        ids=ids,
        name="ei.observation.vision.frame",
        event_type="observation",
        source=source,
        target=target,
        content=observation.to_content(),
        event_id=event_id,
        request_id=request_id,
        time=time,
        sequence=sequence,
        session_id=session_id,
        round_id=round_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=trace_id,
        priority="realtime",
        ttl_ms=ttl_ms,
        mode=mode,
        policy=policy,
        extensions=extensions,
        round_scoped=False,
    )


def build_execution_outcome_event(
    *,
    source: SourceLike,
    outcome_id: str,
    action_id: str = "",
    action_type: str = "",
    success: bool = True,
    status: str = "completed",
    latency_ms: float | None = None,
    did_what: Iterable[str] | None = None,
    errors: Iterable[Mapping[str, Any]] | None = None,
    details: Mapping[str, Any] | None = None,
    ids: EventIdFactory | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    time: str | None = None,
    sequence: int = 1,
    session_id: str = "",
    round_id: str | None = None,
    correlation_id: str = "",
    causation_id: str = "",
    trace_id: str = "",
    target: TargetLike = None,
    ttl_ms: int | None = None,
    mode: Mapping[str, Any] | None = None,
    policy: PolicyState | Mapping[str, Any] | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> EventEnvelope:
    outcome = ExecutionOutcome(
        outcome_id=outcome_id,
        action_id=action_id,
        action_type=action_type,
        success=success,
        status=status,
        latency_ms=latency_ms,
        did_what=list(did_what or []),
        errors=[dict(item) for item in errors or ()],
        details=dict(details or {}),
    )
    return build_event(
        ids=ids,
        name="ei.outcome.execution",
        event_type="outcome",
        source=source,
        target=target,
        content=outcome.to_content(),
        event_id=event_id,
        request_id=request_id,
        time=time,
        sequence=sequence,
        session_id=session_id,
        round_id=round_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=trace_id,
        priority="normal",
        ttl_ms=ttl_ms,
        mode=mode,
        policy=policy,
        extensions=extensions,
        round_scoped=True,
    )


def _source_ref(source: SourceLike) -> SourceRef:
    if isinstance(source, SourceRef):
        return source
    if isinstance(source, Mapping):
        return SourceRef.from_dict(source)
    raise TypeError("source must be a SourceRef or mapping")


def _target_ref(target: TargetLike) -> TargetRef | None:
    if target is None:
        return None
    if isinstance(target, TargetRef):
        return target
    if isinstance(target, Mapping):
        return TargetRef.from_dict(target)
    raise TypeError("target must be a TargetRef, mapping, or None")


def _policy_state(policy: PolicyState | Mapping[str, Any] | None) -> PolicyState:
    if policy is None:
        return PolicyState()
    if isinstance(policy, PolicyState):
        return policy
    if isinstance(policy, Mapping):
        return PolicyState.from_dict(policy)
    raise TypeError("policy must be a PolicyState, mapping, or None")


def _detection(value: Detection | Mapping[str, Any]) -> Detection:
    if isinstance(value, Detection):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("detections must contain Detection objects or mappings")
    return Detection(
        label=str(value.get("label", "") or ""),
        score=float(value.get("score", 0.0) or 0.0),
        bbox=list(value.get("bbox", []) or []),
        track_id=str(value.get("trackId", value.get("track_id", "")) or ""),
        metadata=dict(value.get("metadata", {}) or {}),
    )


def _raise_if_invalid(event: EventEnvelope) -> None:
    errors = [_issue_to_error(issue) for issue in validate_event_strict(event, known_event_required=True)]
    if errors:
        raise ValueError("invalid eiprotocol event: " + "; ".join(errors))


def _issue_to_error(issue: ValidationIssue) -> str:
    if issue.code in {"required", "invalid_spec_version", "invalid_content", "missing_idempotency_key"}:
        return issue.message
    return f"{issue.code} at {issue.path}: {issue.message}"


__all__ = [
    "EventIds",
    "EventIdFactory",
    "build_action_request_event",
    "build_asr_event",
    "build_event",
    "build_execution_outcome_event",
    "build_vision_frame_event",
]
