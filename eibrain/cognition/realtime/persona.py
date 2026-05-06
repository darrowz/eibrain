"""Runtime persona constraints for realtime cognition."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

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


_VOICE_STYLE_PRESETS: dict[str, dict[str, Any]] = {
    "warm": {
        "voice_token": "warm",
        "emotion": "warm",
        "speed": 1.0,
        "volume": 0.76,
    },
    "tired": {
        "voice_token": "soft",
        "emotion": "tired",
        "speed": 0.88,
        "volume": 0.64,
    },
    "noisy": {
        "voice_token": "clear",
        "emotion": "focused",
        "speed": 0.96,
        "volume": 0.92,
    },
    "night": {
        "voice_token": "night",
        "emotion": "soft",
        "speed": 0.86,
        "volume": 0.5,
    },
}


def resolve_voice_style_policy(
    persona_state: Mapping[str, Any] | None = None,
    emotion_state: Mapping[str, Any] | None = None,
    *,
    fallback_voice_code: str = "gentle_companion_zh_cn",
    fallback_emotion: str = "warm",
) -> dict[str, Any]:
    """Map realtime emotion/environment hints to a concrete TTS voice style."""

    persona_state = persona_state or {}
    state = _emotion_state_from_context(emotion_state)
    environment = _mapping(state.get("environment"))
    voice_code = str(persona_state.get("voice_code") or fallback_voice_code or "gentle_companion_zh_cn")
    style = _voice_style_name(state, environment)
    preset = dict(_VOICE_STYLE_PRESETS[style])
    emotion = str(preset.get("emotion") or fallback_emotion or "warm")
    if style == "warm" and fallback_emotion and fallback_emotion != "warm":
        emotion = str(fallback_emotion)
    return to_json_ready(
        {
            "voice_style": style,
            "voice_code": _voice_code_for_style(voice_code, style),
            "emotion": emotion,
            "speed": float(preset["speed"]),
            "volume": float(preset["volume"]),
            "base_voice_code": voice_code,
            "policy": "persona_emotion_voice_policy.v1",
        }
    )


def _emotion_state_from_context(value: Mapping[str, Any] | None) -> dict[str, Any]:
    context = _mapping(value)
    nested = _mapping(context.get("emotion_state"))
    return nested if nested else context


def _voice_style_name(state: Mapping[str, Any], environment: Mapping[str, Any]) -> str:
    mood = _lower_text(state.get("mood"), state.get("state"))
    energy = _lower_text(state.get("energy"))
    noise = _lower_text(
        environment.get("noise"),
        environment.get("noise_level"),
        environment.get("noiseLevel"),
    )
    time_of_day = _lower_text(
        environment.get("time"),
        environment.get("time_of_day"),
        environment.get("timeOfDay"),
    )
    if time_of_day == "night":
        return "night"
    if noise in {"high", "noisy", "loud"}:
        return "noisy"
    if mood in {"tired", "fatigued", "sleepy"} or energy == "low":
        return "tired"
    return "warm"


def _voice_code_for_style(base_voice_code: str, style: str) -> str:
    token = str(_VOICE_STYLE_PRESETS[style]["voice_token"])
    if style == "warm":
        return base_voice_code
    parts = base_voice_code.split("_")
    for index, part in enumerate(parts):
        if part in {"warm", "soft", "clear", "night", "gentle", "calm"}:
            parts[index] = token
            return "_".join(parts)
    if base_voice_code.startswith("joyinside_") and base_voice_code.endswith("_zh_cn"):
        return f"joyinside_{token}_zh_cn"
    if base_voice_code.endswith("_zh_cn"):
        return f"{base_voice_code[:-6]}_{token}_zh_cn"
    return f"{base_voice_code}_{token}"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _lower_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip().lower()
        if text:
            return text
    return ""


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

    def voice_style_for_emotion(
        self,
        emotion_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return resolve_voice_style_policy(
            {
                "voice_code": self.voice_code,
                "emotion_policy": self.emotion_policy,
                "speaking_style": self.speaking_style,
            },
            emotion_state,
            fallback_voice_code=self.voice_code,
            fallback_emotion=str(self.emotion_policy.get("default_emotion") or "warm"),
        )

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
        voice_style = self.voice_style_for_emotion(emotion_context)
        return to_json_ready(
            {
                "text": _limit_text(text, max_chars=max_chars),
                "tone": tone,
                "voice_code": voice_style["voice_code"],
                "voice_style": voice_style["voice_style"],
                "emotion": voice_style["emotion"],
                "speed": voice_style["speed"],
                "volume": voice_style["volume"],
                "action_style": deepcopy(self.action_style),
                "response_policy": deepcopy(self.response_policy),
                "proactive_policy": deepcopy(self.proactive_policy),
            }
        )


__all__ = ["PersonaRuntime", "resolve_voice_style_policy"]
