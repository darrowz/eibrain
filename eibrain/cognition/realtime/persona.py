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


def _default_response_policy() -> dict[str, Any]:
    return {
        "max_chars": 96,
        "sentence_limit": 2,
        "repair_on_uncertainty": "ask_brief_clarifying_question",
    }


def _default_proactive_policy() -> dict[str, Any]:
    return {
        "mode": "low_disturbance",
        "quiet_check_in_after_seconds": 120,
        "suppress_speech_when": ["night", "high_noise", "recent_user_interrupt"],
    }


_PERSONA_PROFILES: dict[str, dict[str, Any]] = {
    "gentle_companion": {
        "persona_id": "gentle_companion",
        "speaking_style": _default_speaking_style(),
        "voice_code": "gentle_companion_zh_cn",
        "emotion_policy": _default_emotion_policy(),
        "action_style": _default_action_style(),
        "memory_policy": _default_memory_policy(),
        "response_policy": _default_response_policy(),
        "proactive_policy": _default_proactive_policy(),
    },
    "joyinside_companion": {
        "persona_id": "joyinside_companion",
        "speaking_style": {
            "tone": "warm_playful",
            "pace": "light",
            "brevity": "brief",
            "language": "zh-CN",
            "avoid": ["overclaiming", "lecturing", "high_pressure_prompting"],
        },
        "voice_code": "joyinside_warm_zh_cn",
        "emotion_policy": {
            "default_emotion": "warm",
            "de_escalate_on_stress": True,
            "mirror_user_intensity": "gentle_cap",
            "max_intensity": 0.58,
        },
        "action_style": {
            "interruptibility": "high",
            "motion": "small_expressive",
            "confirmation": "before_irreversible_actions",
        },
        "memory_policy": _default_memory_policy(),
        "response_policy": {
            "max_chars": 48,
            "sentence_limit": 1,
            "repair_on_uncertainty": "soft_micro_question",
        },
        "proactive_policy": {
            "mode": "low_disturbance_check_in",
            "quiet_check_in_after_seconds": 120,
            "suppress_speech_when": ["night", "high_noise", "recent_user_interrupt"],
        },
    },
}

_PERSONA_ALIASES = {
    "joyinside": "joyinside_companion",
    "joy_inside": "joyinside_companion",
    "joy_inside_companion": "joyinside_companion",
}


def _canonical_persona_code(persona_code: str | None) -> str:
    raw = str(persona_code or "gentle_companion").strip() or "gentle_companion"
    normalized = raw.replace("-", "_").lower()
    return _PERSONA_ALIASES.get(normalized, normalized)


def _profile(persona_code: str | None) -> dict[str, Any]:
    code = _canonical_persona_code(persona_code)
    return deepcopy(_PERSONA_PROFILES.get(code, _PERSONA_PROFILES["gentle_companion"]))


def _limit_text(text: str, *, max_chars: int) -> str:
    cleaned = " ".join(str(text or "").strip().split())
    if max_chars <= 0 or len(cleaned) <= max_chars:
        return cleaned
    if max_chars <= 3:
        return cleaned[:max_chars]
    return cleaned[: max_chars - 3].rstrip() + "..."


@dataclass
class PersonaRuntime:
    """Runtime profile used by fast and slow realtime lanes."""

    persona_id: str = "gentle_companion"
    persona_code: str | None = field(default=None, kw_only=True)
    speaking_style: dict[str, Any] = field(default_factory=_default_speaking_style)
    voice_code: str = "gentle_companion_zh_cn"
    emotion_policy: dict[str, Any] = field(default_factory=_default_emotion_policy)
    action_style: dict[str, Any] = field(default_factory=_default_action_style)
    memory_policy: dict[str, Any] = field(default_factory=_default_memory_policy)
    response_policy: dict[str, Any] = field(default_factory=_default_response_policy)
    proactive_policy: dict[str, Any] = field(default_factory=_default_proactive_policy)

    @classmethod
    def from_persona_code(cls, persona_code: str | None) -> "PersonaRuntime":
        code = _canonical_persona_code(persona_code)
        profile = _profile(code)
        return cls(
            persona_id=str(profile["persona_id"]),
            persona_code=code,
            speaking_style=dict(profile["speaking_style"]),
            voice_code=str(profile["voice_code"]),
            emotion_policy=dict(profile["emotion_policy"]),
            action_style=dict(profile["action_style"]),
            memory_policy=dict(profile["memory_policy"]),
            response_policy=dict(profile["response_policy"]),
            proactive_policy=dict(profile["proactive_policy"]),
        )

    def constraints(self) -> dict[str, Any]:
        return deepcopy(
            to_json_ready(
                {
                    "personaCode": self.persona_code or self.persona_id,
                    "speaking_style": self.speaking_style,
                    "voice_code": self.voice_code,
                    "emotion_policy": self.emotion_policy,
                    "action_style": self.action_style,
                    "memory_policy": self.memory_policy,
                    "response_policy": self.response_policy,
                    "proactive_policy": self.proactive_policy,
                }
            )
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {"persona_id": self.persona_id}
        payload.update(self.constraints())
        return payload

    def snapshot(self) -> dict[str, Any]:
        return self.to_dict()

    def shape_reply(
        self,
        text: str,
        *,
        emotion_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        emotion_context = emotion_context or {}
        max_chars = int(self.response_policy.get("max_chars") or 0)
        tone = str(self.speaking_style.get("tone") or "gentle")
        mood = str(emotion_context.get("mood") or emotion_context.get("state") or "").lower()
        if mood in {"sad", "anxious", "stressed", "lonely", "tired"}:
            tone = "gentle"
        return to_json_ready(
            {
                "text": _limit_text(text, max_chars=max_chars),
                "tone": tone,
                "voice_code": self.voice_code,
                "action_style": deepcopy(self.action_style),
                "response_policy": deepcopy(self.response_policy),
                "proactive_policy": deepcopy(self.proactive_policy),
            }
        )


__all__ = ["PersonaRuntime"]
