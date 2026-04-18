"""World state models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class WorldState:
    current_speaker_id: str | None = None
    last_transcript: str = ""
    last_visual_summary: str = ""
    focus_target_name: str = ""
    focus_target_x: float | None = None
