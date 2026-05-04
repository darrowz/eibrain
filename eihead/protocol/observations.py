"""Local eihead observation contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import ProtocolMessage


@dataclass(slots=True)
class AudioTranscriptFinal(ProtocolMessage):
    kind: str = field(init=False, default="audio_transcript_final")
    text: str = ""
    language: str = "zh"


@dataclass(slots=True)
class HeadObservation(ProtocolMessage):
    target: str = ""
    timestamp_ms: int | None = None
    trace_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    kind: str = field(init=False, default="head_observation")

    @property
    def observation_type(self) -> str:
        return self.kind

    @property
    def modality(self) -> str:
        return "head"


@dataclass(slots=True)
class VisionObservation(HeadObservation):
    frame_id: str = ""
    width: int | None = None
    height: int | None = None
    detections: list[dict[str, Any]] = field(default_factory=list)
    tracked_target: dict[str, Any] = field(default_factory=dict)
    kind: str = field(init=False, default="vision_observation")

    @property
    def modality(self) -> str:
        return "vision"


__all__ = ["AudioTranscriptFinal", "HeadObservation", "VisionObservation"]
