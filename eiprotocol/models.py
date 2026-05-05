"""eiprotocol v0.1 MVP data contracts.

The package intentionally models only the shared wire shapes needed by the
eihead/eibrain split. It is transport-agnostic and keeps policy as metadata so
the first MVP does not force a safety-gate runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping


SPEC_VERSION = "eiprotocol/0.1"


def _dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


def _list(value: list[Any] | tuple[Any, ...] | None) -> list[Any]:
    return list(value or [])


@dataclass(slots=True)
class SourceRef:
    domain: str
    instance_id: str = ""
    device_id: str = ""
    bot_id: str = ""
    uid: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "instanceId": self.instance_id,
            "deviceId": self.device_id,
            "botId": self.bot_id,
            "uid": self.uid,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SourceRef":
        return cls(
            domain=str(data.get("domain", "")),
            instance_id=str(data.get("instanceId", data.get("instance_id", "")) or ""),
            device_id=str(data.get("deviceId", data.get("device_id", "")) or ""),
            bot_id=str(data.get("botId", data.get("bot_id", "")) or ""),
            uid=str(data.get("uid", "") or ""),
            metadata=_dict(data.get("metadata") if isinstance(data.get("metadata"), Mapping) else None),
        )


@dataclass(slots=True)
class TargetRef:
    domain: str
    instance_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "instanceId": self.instance_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TargetRef":
        return cls(
            domain=str(data.get("domain", "")),
            instance_id=str(data.get("instanceId", data.get("instance_id", "")) or ""),
            metadata=_dict(data.get("metadata") if isinstance(data.get("metadata"), Mapping) else None),
        )


@dataclass(slots=True)
class PolicyState:
    decision: str = "not_required"
    risk_level: str = "L0"
    decision_id: str = ""
    required_ack: bool = False
    reason: str = ""
    expires_at: str = ""
    extensions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "riskLevel": self.risk_level,
            "decisionId": self.decision_id,
            "requiredAck": self.required_ack,
            "reason": self.reason,
            "expiresAt": self.expires_at,
            "extensions": dict(self.extensions),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "PolicyState":
        payload = _dict(data)
        return cls(
            decision=str(payload.get("decision", "not_required") or "not_required"),
            risk_level=str(payload.get("riskLevel", payload.get("risk_level", "L0")) or "L0"),
            decision_id=str(payload.get("decisionId", payload.get("decision_id", "")) or ""),
            required_ack=bool(payload.get("requiredAck", payload.get("required_ack", False))),
            reason=str(payload.get("reason", "") or ""),
            expires_at=str(payload.get("expiresAt", payload.get("expires_at", "")) or ""),
            extensions=_dict(payload.get("extensions") if isinstance(payload.get("extensions"), Mapping) else None),
        )


@dataclass(slots=True)
class EventEnvelope:
    event_id: str
    event_type: str
    name: str
    time: str
    sequence: int
    request_id: str
    source: SourceRef
    content: dict[str, Any]
    session_id: str = ""
    round_id: str = ""
    correlation_id: str = ""
    causation_id: str = ""
    trace_id: str = ""
    target: TargetRef | None = None
    priority: str = "normal"
    ttl_ms: int | None = None
    mode: dict[str, Any] = field(default_factory=dict)
    policy: PolicyState = field(default_factory=PolicyState)
    extensions: dict[str, Any] = field(default_factory=dict)
    spec_version: str = SPEC_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "specVersion": self.spec_version,
            "id": self.event_id,
            "type": self.event_type,
            "name": self.name,
            "time": self.time,
            "sequence": int(self.sequence),
            "requestId": self.request_id,
            "sessionId": self.session_id,
            "roundId": self.round_id,
            "correlationId": self.correlation_id,
            "causationId": self.causation_id,
            "traceId": self.trace_id,
            "source": self.source.to_dict(),
            "priority": self.priority,
            "ttlMs": self.ttl_ms,
            "mode": dict(self.mode),
            "content": dict(self.content),
            "policy": self.policy.to_dict(),
            "extensions": dict(self.extensions),
        }
        if self.target is not None:
            payload["target"] = self.target.to_dict()
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "EventEnvelope":
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EventEnvelope":
        source = data.get("source")
        target = data.get("target")
        policy = data.get("policy")
        return cls(
            spec_version=str(data.get("specVersion", data.get("spec_version", SPEC_VERSION)) or SPEC_VERSION),
            event_id=str(data.get("id", data.get("event_id", "")) or ""),
            event_type=str(data.get("type", data.get("event_type", "")) or ""),
            name=str(data.get("name", "") or ""),
            time=str(data.get("time", "") or ""),
            sequence=int(data.get("sequence", 0) or 0),
            request_id=str(data.get("requestId", data.get("request_id", "")) or ""),
            session_id=str(data.get("sessionId", data.get("session_id", "")) or ""),
            round_id=str(data.get("roundId", data.get("round_id", "")) or ""),
            correlation_id=str(data.get("correlationId", data.get("correlation_id", "")) or ""),
            causation_id=str(data.get("causationId", data.get("causation_id", "")) or ""),
            trace_id=str(data.get("traceId", data.get("trace_id", "")) or ""),
            source=SourceRef.from_dict(source if isinstance(source, Mapping) else {}),
            target=TargetRef.from_dict(target) if isinstance(target, Mapping) else None,
            priority=str(data.get("priority", "normal") or "normal"),
            ttl_ms=int(data["ttlMs"]) if data.get("ttlMs") is not None else None,
            mode=_dict(data.get("mode") if isinstance(data.get("mode"), Mapping) else None),
            content=_dict(data.get("content") if isinstance(data.get("content"), Mapping) else None),
            policy=PolicyState.from_dict(policy if isinstance(policy, Mapping) else None),
            extensions=_dict(data.get("extensions") if isinstance(data.get("extensions"), Mapping) else None),
        )


@dataclass(slots=True)
class DeviceStatus:
    status: str = "unknown"
    message: str = ""
    checked_at_ms: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "checkedAtMs": self.checked_at_ms,
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "DeviceStatus":
        payload = _dict(data)
        return cls(
            status=str(payload.get("status", "unknown") or "unknown"),
            message=str(payload.get("message", "") or ""),
            checked_at_ms=int(payload["checkedAtMs"]) if payload.get("checkedAtMs") is not None else None,
            metrics=_dict(payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else None),
        )


@dataclass(slots=True)
class Capability:
    capability_id: str
    kind: str
    provider: str = ""
    model: str = ""
    version: str = ""
    device_path: str = ""
    actions: list[str] = field(default_factory=list)
    status: str = "unknown"
    limits: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capabilityId": self.capability_id,
            "kind": self.kind,
            "provider": self.provider,
            "model": self.model,
            "version": self.version,
            "devicePath": self.device_path,
            "actions": list(self.actions),
            "status": self.status,
            "limits": dict(self.limits),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Capability":
        return cls(
            capability_id=str(data.get("capabilityId", data.get("capability_id", "")) or ""),
            kind=str(data.get("kind", "") or ""),
            provider=str(data.get("provider", "") or ""),
            model=str(data.get("model", "") or ""),
            version=str(data.get("version", "") or ""),
            device_path=str(data.get("devicePath", data.get("device_path", "")) or ""),
            actions=[str(item) for item in _list(data.get("actions") if isinstance(data.get("actions"), list) else None)],
            status=str(data.get("status", "unknown") or "unknown"),
            limits=_dict(data.get("limits") if isinstance(data.get("limits"), Mapping) else None),
            metadata=_dict(data.get("metadata") if isinstance(data.get("metadata"), Mapping) else None),
        )


@dataclass(slots=True)
class CapabilityManifest:
    manifest_id: str
    manifest_version: str = "0.1.0"
    device: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    transports: dict[str, Any] = field(default_factory=dict)
    modalities: dict[str, Any] = field(default_factory=dict)
    capabilities: list[Capability] = field(default_factory=list)
    backends: list[Capability] = field(default_factory=list)
    health: DeviceStatus = field(default_factory=DeviceStatus)
    limits: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        return {
            "manifestId": self.manifest_id,
            "manifestVersion": self.manifest_version,
            "device": dict(self.device),
            "runtime": dict(self.runtime),
            "transports": dict(self.transports),
            "modalities": dict(self.modalities),
            "capabilities": [item.to_dict() for item in self.capabilities],
            "backends": [item.to_dict() for item in self.backends],
            "health": self.health.to_dict(),
            "limits": dict(self.limits),
            "metadata": dict(self.metadata),
        }

    def to_event(
        self,
        *,
        event_id: str,
        request_id: str,
        sequence: int,
        source: SourceRef,
        time: str,
        target: TargetRef | None = None,
    ) -> EventEnvelope:
        return EventEnvelope(
            event_id=event_id,
            event_type="capability",
            name="ei.capability.manifest.report",
            time=time,
            sequence=sequence,
            request_id=request_id,
            source=source,
            target=target,
            content=self.to_content(),
            priority="normal",
        )


@dataclass(slots=True)
class AudioTurn:
    text: str
    language: str = "und"
    final: bool = True
    confidence: float | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    audio_level: float | None = None
    wake_word: str = ""
    asr_backend: str = ""
    timings_ms: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "final": self.final,
            "confidence": self.confidence,
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "audioLevel": self.audio_level,
            "wakeWord": self.wake_word,
            "asrBackend": self.asr_backend,
            "timingsMs": dict(self.timings_ms),
            "metadata": dict(self.metadata),
        }

    def to_event(self, **kwargs: Any) -> EventEnvelope:
        return _round_event(
            event_type="dialogue",
            name="ei.dialogue.asr.final" if self.final else "ei.dialogue.asr.partial",
            content=self.to_content(),
            priority="realtime",
            **kwargs,
        )


@dataclass(slots=True)
class HeadStatusReport:
    status: str
    components: dict[str, Any] = field(default_factory=dict)
    reported_at: str = ""
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "components": dict(self.components),
            "reportedAt": self.reported_at,
            "summary": self.summary,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_content(cls, data: Mapping[str, Any]) -> "HeadStatusReport":
        components = data.get("components")
        metadata = data.get("metadata")
        return cls(
            status=str(data.get("status", "") or ""),
            components=_dict(components if isinstance(components, Mapping) else None),
            reported_at=str(data.get("reportedAt", data.get("reported_at", "")) or ""),
            summary=str(data.get("summary", "") or ""),
            metadata=_dict(metadata if isinstance(metadata, Mapping) else None),
        )


@dataclass(slots=True)
class DialogueFastHypothesis:
    hypothesis_id: str
    text: str
    confidence: float
    basis_event_id: str = ""
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        return {
            "hypothesisId": self.hypothesis_id,
            "text": self.text,
            "confidence": float(self.confidence),
            "basisEventId": self.basis_event_id,
            "latencyMs": self.latency_ms,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_content(cls, data: Mapping[str, Any]) -> "DialogueFastHypothesis":
        metadata = data.get("metadata")
        return cls(
            hypothesis_id=str(data.get("hypothesisId", data.get("hypothesis_id", "")) or ""),
            text=str(data.get("text", "") or ""),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            basis_event_id=str(data.get("basisEventId", data.get("basis_event_id", "")) or ""),
            latency_ms=_optional_float(data.get("latencyMs", data.get("latency_ms"))),
            metadata=_dict(metadata if isinstance(metadata, Mapping) else None),
        )

    def to_event(self, **kwargs: Any) -> EventEnvelope:
        return _round_event(
            event_type="dialogue",
            name="ei.dialogue.fast_hypothesis",
            content=self.to_content(),
            priority="realtime",
            **kwargs,
        )


@dataclass(slots=True)
class DialogueStableDecision:
    decision_id: str
    decision: str
    confidence: float
    text: str = ""
    actions: list[dict[str, Any]] = field(default_factory=list)
    stable_since_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        return {
            "decisionId": self.decision_id,
            "decision": self.decision,
            "confidence": float(self.confidence),
            "text": self.text,
            "actions": [dict(item) for item in self.actions],
            "stableSinceMs": self.stable_since_ms,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_content(cls, data: Mapping[str, Any]) -> "DialogueStableDecision":
        actions = data.get("actions")
        metadata = data.get("metadata")
        return cls(
            decision_id=str(data.get("decisionId", data.get("decision_id", "")) or ""),
            decision=str(data.get("decision", "") or ""),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            text=str(data.get("text", "") or ""),
            actions=[dict(item) for item in actions] if isinstance(actions, list) else [],
            stable_since_ms=_optional_float(data.get("stableSinceMs", data.get("stable_since_ms"))),
            metadata=_dict(metadata if isinstance(metadata, Mapping) else None),
        )

    def to_event(self, **kwargs: Any) -> EventEnvelope:
        return _round_event(
            event_type="dialogue",
            name="ei.dialogue.decision.stable",
            content=self.to_content(),
            priority="high",
            **kwargs,
        )


@dataclass(slots=True)
class EmotionContext:
    context_id: str
    mood: str
    confidence: float
    signals: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        return {
            "contextId": self.context_id,
            "mood": self.mood,
            "confidence": float(self.confidence),
            "signals": dict(self.signals),
            "environment": dict(self.environment),
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_content(cls, data: Mapping[str, Any]) -> "EmotionContext":
        signals = data.get("signals")
        if not isinstance(signals, Mapping) and isinstance(data.get("prosody"), Mapping):
            signals = {"prosody": dict(data.get("prosody") or {})}
        environment = data.get("environment")
        metadata = data.get("metadata")
        return cls(
            context_id=str(data.get("contextId", data.get("context_id", "")) or ""),
            mood=str(data.get("mood", data.get("state", "")) or ""),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            signals=_dict(signals if isinstance(signals, Mapping) else None),
            environment=_dict(environment if isinstance(environment, Mapping) else None),
            source=str(data.get("source", "") or ""),
            metadata=_dict(metadata if isinstance(metadata, Mapping) else None),
        )

    def to_event(self, **kwargs: Any) -> EventEnvelope:
        return _round_event(
            event_type="observation",
            name="ei.observation.emotion.context",
            content=self.to_content(),
            priority="realtime",
            **kwargs,
        )


@dataclass(slots=True)
class MemoryPrefetchRequest:
    prefetch_id: str
    query: str
    reason: str
    candidates: list[dict[str, Any]] = field(default_factory=list)
    scope: list[str] = field(default_factory=list)
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        return {
            "prefetchId": self.prefetch_id,
            "query": self.query,
            "reason": self.reason,
            "candidates": [dict(item) for item in self.candidates],
            "scope": list(self.scope),
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_content(cls, data: Mapping[str, Any]) -> "MemoryPrefetchRequest":
        candidates = data.get("candidates")
        scope = data.get("scope")
        metadata = data.get("metadata")
        return cls(
            prefetch_id=str(data.get("prefetchId", data.get("prefetch_id", "")) or ""),
            query=str(data.get("query", "") or ""),
            reason=str(data.get("reason", "") or ""),
            candidates=_dict_items(candidates),
            scope=[str(item) for item in _list(scope if isinstance(scope, (list, tuple)) else None)],
            source=str(data.get("source", "") or ""),
            metadata=_dict(metadata if isinstance(metadata, Mapping) else None),
        )

    def to_event(self, **kwargs: Any) -> EventEnvelope:
        return _round_event(
            event_type="memory",
            name="ei.memory.prefetch.requested",
            content=self.to_content(),
            priority="realtime",
            **kwargs,
        )


@dataclass(slots=True)
class SpeechActionPlan:
    plan_id: str
    stable: bool
    speech_segments: list[dict[str, Any]] = field(default_factory=list)
    action_segments: list[dict[str, Any]] = field(default_factory=list)
    language: str = "zh-CN"
    fallback_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        return {
            "planId": self.plan_id,
            "stable": bool(self.stable),
            "speechSegments": [dict(item) for item in self.speech_segments],
            "actionSegments": [dict(item) for item in self.action_segments],
            "language": self.language,
            "fallbackText": self.fallback_text,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_content(cls, data: Mapping[str, Any]) -> "SpeechActionPlan":
        speech_segments = data.get("speechSegments", data.get("speech_segments", data.get("speech")))
        action_segments = data.get("actionSegments", data.get("action_segments", data.get("actions")))
        metadata = data.get("metadata")
        return cls(
            plan_id=str(data.get("planId", data.get("plan_id", "")) or ""),
            stable=bool(data.get("stable", False)),
            speech_segments=_dict_items(speech_segments),
            action_segments=_dict_items(action_segments),
            language=str(data.get("language", "zh-CN") or "zh-CN"),
            fallback_text=str(data.get("fallbackText", data.get("fallback_text", "")) or ""),
            metadata=_dict(metadata if isinstance(metadata, Mapping) else None),
        )

    def to_event(self, **kwargs: Any) -> EventEnvelope:
        return _round_event(
            event_type="dialogue",
            name="ei.dialogue.speech_action.plan",
            content=self.to_content(),
            priority="high",
            **kwargs,
        )


@dataclass(slots=True)
class ProactiveActivityProposal:
    proposal_id: str
    channel: str
    reason: str
    should_emit: bool
    urgency: float | None = None
    disturbance: str = "low"
    requires_user_attention: bool = False
    text: str = ""
    memory_refs: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        return {
            "proposalId": self.proposal_id,
            "channel": self.channel,
            "reason": self.reason,
            "shouldEmit": bool(self.should_emit),
            "urgency": self.urgency,
            "disturbance": self.disturbance,
            "requiresUserAttention": self.requires_user_attention,
            "text": self.text,
            "memoryRefs": [dict(item) for item in self.memory_refs],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_content(cls, data: Mapping[str, Any]) -> "ProactiveActivityProposal":
        memory_refs = data.get("memoryRefs", data.get("memory_refs"))
        metadata = data.get("metadata")
        return cls(
            proposal_id=str(data.get("proposalId", data.get("proposal_id", "")) or ""),
            channel=str(data.get("channel", "") or ""),
            reason=str(data.get("reason", "") or ""),
            should_emit=bool(data.get("shouldEmit", data.get("should_emit", False))),
            urgency=_optional_float(data.get("urgency")),
            disturbance=str(data.get("disturbance", "low") or "low"),
            requires_user_attention=bool(
                data.get("requiresUserAttention", data.get("requires_user_attention", False))
            ),
            text=str(data.get("text", "") or ""),
            memory_refs=_dict_items(memory_refs),
            metadata=_dict(metadata if isinstance(metadata, Mapping) else None),
        )

    def to_event(self, **kwargs: Any) -> EventEnvelope:
        return _round_event(
            event_type="dialogue",
            name="ei.activity.proactive.proposed",
            content=self.to_content(),
            priority="realtime",
            **kwargs,
        )


@dataclass(slots=True)
class DialogueCancellationApplied:
    cancellation_id: str
    cancelled_round_id: str
    cancellation_token: str
    reason: str
    applied_to: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        return {
            "cancellationId": self.cancellation_id,
            "cancelledRoundId": self.cancelled_round_id,
            "cancellationToken": self.cancellation_token,
            "reason": self.reason,
            "appliedTo": list(self.applied_to),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_content(cls, data: Mapping[str, Any]) -> "DialogueCancellationApplied":
        applied_to = data.get("appliedTo", data.get("applied_to"))
        metadata = data.get("metadata")
        return cls(
            cancellation_id=str(data.get("cancellationId", data.get("cancellation_id", "")) or ""),
            cancelled_round_id=str(data.get("cancelledRoundId", data.get("cancelled_round_id", "")) or ""),
            cancellation_token=str(data.get("cancellationToken", data.get("cancellation_token", "")) or ""),
            reason=str(data.get("reason", "") or ""),
            applied_to=[str(item) for item in _list(applied_to if isinstance(applied_to, (list, tuple)) else None)],
            metadata=_dict(metadata if isinstance(metadata, Mapping) else None),
        )

    def to_event(self, **kwargs: Any) -> EventEnvelope:
        return _round_event(
            event_type="dialogue",
            name="ei.dialogue.cancellation.applied",
            content=self.to_content(),
            priority="high",
            **kwargs,
        )


@dataclass(slots=True)
class Detection:
    label: str
    score: float
    bbox: list[float]
    track_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "score": float(self.score),
            "bbox": list(self.bbox),
            "trackId": self.track_id,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class RealtimeVisionObservation:
    frame_id: str
    width: int | None = None
    height: int | None = None
    frame_age_ms: float | None = None
    backend: str = ""
    detections: list[Detection] = field(default_factory=list)
    latency_ms: dict[str, Any] = field(default_factory=dict)
    image_url: str = ""
    status: str = "ok"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        return {
            "frameId": self.frame_id,
            "width": self.width,
            "height": self.height,
            "frameAgeMs": self.frame_age_ms,
            "backend": self.backend,
            "detections": [item.to_dict() for item in self.detections],
            "latencyMs": dict(self.latency_ms),
            "imageUrl": self.image_url,
            "status": self.status,
            "metadata": dict(self.metadata),
        }

    def to_event(self, **kwargs: Any) -> EventEnvelope:
        return _round_event(
            event_type="observation",
            name="ei.observation.vision.frame",
            content=self.to_content(),
            priority="realtime",
            **kwargs,
        )


@dataclass(slots=True)
class HeadAction:
    action_id: str
    action_type: str
    target: str
    params: dict[str, Any] = field(default_factory=dict)
    risk_level: str = "L1"
    idempotency_key: str = ""
    timeline: list[dict[str, Any]] = field(default_factory=list)
    requires_policy: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        content = {
            "actionId": self.action_id,
            "actionType": self.action_type,
            "target": self.target,
            "params": dict(self.params),
            "riskLevel": self.risk_level,
            "timeline": [dict(item) for item in self.timeline],
            "requiresPolicy": self.requires_policy,
            "metadata": dict(self.metadata),
        }
        if self.idempotency_key:
            content["idempotencyKey"] = self.idempotency_key
        return content

    def to_event(self, **kwargs: Any) -> EventEnvelope:
        event = _round_event(
            event_type="action",
            name="ei.action.request",
            content=self.to_content(),
            priority="high",
            **kwargs,
        )
        event.policy = PolicyState(decision="not_required", risk_level=self.risk_level)
        return event


@dataclass(slots=True)
class ExecutionOutcome:
    outcome_id: str
    action_id: str = ""
    action_type: str = ""
    success: bool = True
    status: str = "completed"
    latency_ms: float | None = None
    did_what: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        return {
            "outcomeId": self.outcome_id,
            "actionId": self.action_id,
            "actionType": self.action_type,
            "success": self.success,
            "status": self.status,
            "latencyMs": self.latency_ms,
            "didWhat": list(self.did_what),
            "errors": [dict(item) for item in self.errors],
            "details": dict(self.details),
        }

    def to_event(self, **kwargs: Any) -> EventEnvelope:
        return _round_event(
            event_type="outcome",
            name="ei.outcome.execution",
            content=self.to_content(),
            priority="normal",
            **kwargs,
        )


@dataclass(slots=True)
class UserFeedback:
    feedback_id: str
    satisfied: bool | None = None
    rating: int | None = None
    text: str = ""
    next_time_change: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        return {
            "feedbackId": self.feedback_id,
            "satisfied": self.satisfied,
            "rating": self.rating,
            "text": self.text,
            "nextTimeChange": self.next_time_change,
            "metadata": dict(self.metadata),
        }

    def to_event(self, **kwargs: Any) -> EventEnvelope:
        return _round_event(
            event_type="outcome",
            name="ei.outcome.user.feedback",
            content=self.to_content(),
            priority="normal",
            **kwargs,
        )


def _round_event(
    *,
    event_type: str,
    name: str,
    content: dict[str, Any],
    priority: str,
    event_id: str,
    request_id: str,
    session_id: str,
    round_id: str,
    sequence: int,
    source: SourceRef,
    time: str,
    target: TargetRef | None = None,
    correlation_id: str = "",
    causation_id: str = "",
    trace_id: str = "",
    ttl_ms: int | None = None,
    mode: dict[str, Any] | None = None,
    extensions: dict[str, Any] | None = None,
) -> EventEnvelope:
    return EventEnvelope(
        event_id=event_id,
        event_type=event_type,
        name=name,
        time=time,
        sequence=sequence,
        request_id=request_id,
        session_id=session_id,
        round_id=round_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=trace_id,
        source=source,
        target=target,
        priority=priority,
        ttl_ms=ttl_ms,
        mode=dict(mode or {}),
        content=content,
        policy=PolicyState(),
        extensions=dict(extensions or {}),
    )


def _dict_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _optional_float(value: Any) -> float | None:
    return float(value) if value not in (None, "") else None


def validate_event(event: EventEnvelope | Mapping[str, Any]) -> list[str]:
    payload = event.to_dict() if isinstance(event, EventEnvelope) else dict(event)
    errors: list[str] = []
    required = ("specVersion", "id", "type", "name", "time", "sequence", "requestId", "source", "priority", "content", "policy")
    for key in required:
        if key not in payload or payload.get(key) in (None, ""):
            errors.append(f"{key} is required")
    if payload.get("specVersion") != SPEC_VERSION:
        errors.append("specVersion must be eiprotocol/0.1")

    name = str(payload.get("name", ""))
    event_type = str(payload.get("type", ""))
    if event_type in {"dialogue", "action", "memory", "outcome", "training"} and not payload.get("roundId"):
        errors.append("roundId is required for turn-scoped events")

    content = payload.get("content")
    if not isinstance(content, Mapping):
        errors.append("content must be an object")
        return errors
    if name in {"ei.action.request", "ei.action.dispatch", "ei.action.emergency.stop"} and not content.get("idempotencyKey"):
        errors.append("content.idempotencyKey is required for side-effecting action events")
    return errors
