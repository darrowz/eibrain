"""Minimal local protocol models for the eihead runtime."""

from __future__ import annotations

from .actions import MoveHeadAction, PlaySpeechAction, StopSpeechAction
from .outcomes import ActionExecuted, SpeechPlaybackCompleted

__all__ = [
    "ActionExecuted",
    "MoveHeadAction",
    "PlaySpeechAction",
    "SpeechPlaybackCompleted",
    "StopSpeechAction",
]
