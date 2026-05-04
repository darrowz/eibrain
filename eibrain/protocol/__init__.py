"""Protocol contracts for observations, intents, actions, outcomes, and envelopes."""

from .actions import PlaySpeechAction
from .capabilities import CapabilityManifest, HeadBackend, HeadDevice, HeadHealth, HeadLimit
from .envelopes import Envelope
from .events import CognitiveDecision, Moment, ObservationEvent, SalienceDecision
from .head import (
    AudioTurn,
    DeviceStatus,
    ExecutionOutcome,
    HeadAction,
    HeadObservation,
    UserFeedback,
    VisionObservation,
)
from .intents import SpeakIntent
from .observations import AudioTranscriptFinal
from .outcomes import SpeechPlaybackCompleted

__all__ = [
    "AudioTranscriptFinal",
    "AudioTurn",
    "CapabilityManifest",
    "CognitiveDecision",
    "DeviceStatus",
    "ExecutionOutcome",
    "SpeakIntent",
    "HeadAction",
    "HeadBackend",
    "HeadDevice",
    "HeadHealth",
    "HeadLimit",
    "HeadObservation",
    "Moment",
    "ObservationEvent",
    "PlaySpeechAction",
    "SalienceDecision",
    "SpeechPlaybackCompleted",
    "UserFeedback",
    "VisionObservation",
    "Envelope",
]
