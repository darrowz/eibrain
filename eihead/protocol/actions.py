"""Local eihead action contracts.

These models intentionally mirror the small action shape that eihead needs
without importing the eibrain protocol package.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class _ActionMessage:
    ts: float
    source: str
    session_id: str | None = None
    actor_id: str | None = None
    target_id: str | None = None
    kind: str = field(init=False, default="action")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlaySpeechAction(_ActionMessage):
    text: str = ""
    kind: str = field(init=False, default="play_speech_action")


@dataclass(slots=True)
class StopSpeechAction(_ActionMessage):
    kind: str = field(init=False, default="stop_speech_action")


@dataclass(slots=True)
class MoveHeadAction(_ActionMessage):
    target_name: str = ""
    target_x: float | None = None
    target_angle: int | None = None
    kind: str = field(init=False, default="move_head_action")
