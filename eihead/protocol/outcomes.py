"""Local eihead outcome contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class _OutcomeMessage:
    ts: float
    source: str
    session_id: str | None = None
    actor_id: str | None = None
    target_id: str | None = None
    status: str = "ok"
    kind: str = field(init=False, default="outcome")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SpeechPlaybackCompleted(_OutcomeMessage):
    kind: str = field(init=False, default="speech_playback_completed")


@dataclass(slots=True)
class ActionExecuted(_OutcomeMessage):
    action_kind: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    kind: str = field(init=False, default="action_executed")
