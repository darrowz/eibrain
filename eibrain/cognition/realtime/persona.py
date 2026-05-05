"""Runtime persona constraints for realtime cognition."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .events import to_json_ready


def _default_speaking_style() -> dict[str, Any]:
    return {
        "tone": "gentle",
        "pace": "unhurried",
        "brevity": "concise",
        "language": "zh-CN",
        "avoid": ["overclaiming", "harsh_correction", "needless_complexity"],
    }


def _default_emotion_policy() -> dict[str, Any]:
    return {
        "default_emotion": "warm",
        "de_escalate_on_stress": True,
        "mirror_user_intensity": "bounded",
        "max_intensity": 0.65,
    }


def _default_action_style() -> dict[str, Any]:
    return {
        "interruptibility": "high",
        "motion": "soft",
        "confirmation": "before_irreversible_actions",
    }


def _default_memory_policy() -> dict[str, Any]:
    return {
        "recall": "relevant_recent_and_user_aligned",
        "writeback": "salient_or_user_requested",
        "sensitive_inference": "avoid_without_user_signal",
    }


@dataclass
class PersonaRuntime:
    """Runtime profile used by fast and slow realtime lanes."""

    persona_id: str = "gentle_companion"
    speaking_style: dict[str, Any] = field(default_factory=_default_speaking_style)
    voice_code: str = "gentle_companion_zh_cn"
    emotion_policy: dict[str, Any] = field(default_factory=_default_emotion_policy)
    action_style: dict[str, Any] = field(default_factory=_default_action_style)
    memory_policy: dict[str, Any] = field(default_factory=_default_memory_policy)

    def constraints(self) -> dict[str, Any]:
        return deepcopy(
            to_json_ready(
                {
                    "speaking_style": self.speaking_style,
                    "voice_code": self.voice_code,
                    "emotion_policy": self.emotion_policy,
                    "action_style": self.action_style,
                    "memory_policy": self.memory_policy,
                }
            )
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {"persona_id": self.persona_id}
        payload.update(self.constraints())
        return payload

    def snapshot(self) -> dict[str, Any]:
        return self.to_dict()


__all__ = ["PersonaRuntime"]
