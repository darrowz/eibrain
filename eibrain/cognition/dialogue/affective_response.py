"""Structured response contract for embodied, affective dialogue."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


EMOTIONS = {"neutral", "warm", "curious", "focused", "playful", "concerned", "confirming"}
SPEAKING_STYLES = {"plain", "gentle", "brief", "energetic", "reassuring"}
GAZE_INTENTS = {"hold", "search", "track_speaker", "look_away", "idle"}


@dataclass(frozen=True, slots=True)
class AffectiveResponse:
    text: str
    emotion: str = "warm"
    speaking_style: str = "gentle"
    gaze_intent: str = "track_speaker"
    memory_writeback: bool = True
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "emotion": self.emotion,
            "speaking_style": self.speaking_style,
            "gaze_intent": self.gaze_intent,
            "memory_writeback": self.memory_writeback,
            "tags": list(self.tags),
        }

    @classmethod
    def from_text(cls, text: str) -> "AffectiveResponse":
        clean = " ".join(text.strip().split())
        return cls(text=clean)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "AffectiveResponse":
        text = str(payload.get("text", "") or "").strip()
        emotion = _safe_choice(payload.get("emotion"), EMOTIONS, "warm")
        speaking_style = _safe_choice(payload.get("speaking_style"), SPEAKING_STYLES, "gentle")
        gaze_intent = _safe_choice(payload.get("gaze_intent"), GAZE_INTENTS, "track_speaker")
        tags_raw = payload.get("tags", ())
        if isinstance(tags_raw, str):
            tags = (tags_raw,)
        elif isinstance(tags_raw, (list, tuple)):
            tags = tuple(str(tag) for tag in tags_raw if str(tag).strip())
        else:
            tags = ()
        return cls(
            text=text,
            emotion=emotion,
            speaking_style=speaking_style,
            gaze_intent=gaze_intent,
            memory_writeback=bool(payload.get("memory_writeback", True)),
            tags=tags,
        )


def _safe_choice(value: object, allowed: set[str], default: str) -> str:
    candidate = str(value or "").strip()
    return candidate if candidate in allowed else default
