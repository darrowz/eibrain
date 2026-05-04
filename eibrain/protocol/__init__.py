"""Protocol contracts for observations, intents, actions, outcomes, and envelopes."""

from .actions import PlaySpeechAction
from .capabilities import CapabilityManifest, HeadBackend, HeadDevice, HeadHealth, HeadLimit
from .envelopes import Envelope
from .eiprotocol_bridge import (
    audio_turn_to_eiprotocol_event,
    capability_manifest_to_eiprotocol_event,
    execution_outcome_to_eiprotocol_event,
    head_action_to_eiprotocol_event,
    to_eiprotocol_event,
    vision_observation_to_eiprotocol_event,
)
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
    "audio_turn_to_eiprotocol_event",
    "CapabilityManifest",
    "capability_manifest_to_eiprotocol_event",
    "CognitiveDecision",
    "DeviceStatus",
    "ExecutionOutcome",
    "execution_outcome_to_eiprotocol_event",
    "SpeakIntent",
    "HeadAction",
    "head_action_to_eiprotocol_event",
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
    "to_eiprotocol_event",
    "UserFeedback",
    "VisionObservation",
    "vision_observation_to_eiprotocol_event",
    "Envelope",
]
