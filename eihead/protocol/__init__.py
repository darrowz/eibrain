"""Minimal local protocol models for the eihead runtime."""

from __future__ import annotations

from .actions import Action, MoveHeadAction, PlaySpeechAction, StopSpeechAction
from .base import ProtocolMessage, message_payload, serialize_message
from .observations import AudioTranscriptFinal, HeadObservation, VisionObservation
from .outcomes import ActionExecuted, Outcome, SpeechPlaybackCompleted

__all__ = [
    "ActionExecuted",
    "Action",
    "AudioTranscriptFinal",
    "HeadObservation",
    "MoveHeadAction",
    "Outcome",
    "PlaySpeechAction",
    "ProtocolMessage",
    "SpeechPlaybackCompleted",
    "StopSpeechAction",
    "VisionObservation",
    "message_payload",
    "serialize_message",
]
