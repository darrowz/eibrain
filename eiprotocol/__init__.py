"""Shared eiprotocol v0.1 MVP contracts for EI projects."""

from .models import (
    SPEC_VERSION,
    AudioTurn,
    Capability,
    CapabilityManifest,
    Detection,
    DeviceStatus,
    EventEnvelope,
    ExecutionOutcome,
    HeadAction,
    PolicyState,
    RealtimeVisionObservation,
    SourceRef,
    TargetRef,
    UserFeedback,
    validate_event,
)

__all__ = [
    "SPEC_VERSION",
    "AudioTurn",
    "Capability",
    "CapabilityManifest",
    "Detection",
    "DeviceStatus",
    "EventEnvelope",
    "ExecutionOutcome",
    "HeadAction",
    "PolicyState",
    "RealtimeVisionObservation",
    "SourceRef",
    "TargetRef",
    "UserFeedback",
    "validate_event",
]
