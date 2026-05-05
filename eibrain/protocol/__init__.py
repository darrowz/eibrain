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
from .joyinside_voice import (
    JoyInsideVoiceEvent,
    audio_chunk,
    audio_finish,
    input_text_to_speech,
    interrupt,
    normalize_voice_event,
    parse_downstream_event,
    ping,
    text_message,
    to_eiprotocol_name,
    voice_chat_update,
)
from .observations import AudioTranscriptFinal
from .outcomes import SpeechPlaybackCompleted

__all__ = [
    "AudioTranscriptFinal",
    "AudioTurn",
    "JoyInsideVoiceEvent",
    "audio_chunk",
    "audio_finish",
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
    "input_text_to_speech",
    "interrupt",
    "Moment",
    "normalize_voice_event",
    "ObservationEvent",
    "parse_downstream_event",
    "ping",
    "PlaySpeechAction",
    "SalienceDecision",
    "SpeechPlaybackCompleted",
    "to_eiprotocol_event",
    "text_message",
    "to_eiprotocol_name",
    "UserFeedback",
    "VisionObservation",
    "vision_observation_to_eiprotocol_event",
    "voice_chat_update",
    "Envelope",
]
