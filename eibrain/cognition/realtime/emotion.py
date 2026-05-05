"""Emotion and environment context normalization for realtime lanes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .events import to_json_ready


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _merge(target: dict[str, Any], source: Mapping[str, Any] | None) -> None:
    if source:
        target.update(dict(source))


@dataclass
class EmotionContextBuilder:
    """Merge prosody, environment, and vision hints into response guidance."""

    high_noise_db: float = 70.0
    medium_noise_db: float = 60.0

    def build(
        self,
        *,
        observations: Iterable[Mapping[str, Any]] | None = None,
        prosody: Mapping[str, Any] | None = None,
        environment: Mapping[str, Any] | None = None,
        vision: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        prosody_hints: dict[str, Any] = {}
        environment_hints: dict[str, Any] = {}
        vision_hints: dict[str, Any] = {}

        for item in observations or ():
            kind = item.get("kind")
            payload = item.get("payload", {})
            if kind == "prosody" and isinstance(payload, Mapping):
                _merge(prosody_hints, payload)
            elif kind == "environment" and isinstance(payload, Mapping):
                _merge(environment_hints, payload)
            elif kind == "vision" and isinstance(payload, Mapping):
                _merge(vision_hints, payload)

        _merge(prosody_hints, prosody)
        _merge(environment_hints, environment)
        _merge(vision_hints, vision)

        emotion_hint = self._emotion_hint(prosody_hints, vision_hints)
        noise_policy = self._noise_policy(environment_hints)
        response_style = self._response_style(emotion_hint=emotion_hint, noise_policy=noise_policy)

        return to_json_ready(
            {
                "emotion_hint": emotion_hint,
                "noise_policy": noise_policy,
                "response_style": response_style,
                "inputs": {
                    "prosody": prosody_hints,
                    "environment": environment_hints,
                    "vision": vision_hints,
                },
            }
        )

    def _emotion_hint(
        self,
        prosody_hints: Mapping[str, Any],
        vision_hints: Mapping[str, Any],
    ) -> dict[str, Any]:
        arousal = _as_float(prosody_hints.get("arousal"))
        valence = _as_float(prosody_hints.get("valence"))
        stress = _as_float(prosody_hints.get("stress"))
        expression = str(vision_hints.get("face_expression", "")).lower()
        attention = str(vision_hints.get("attention", "")).lower()
        sources: list[str] = []

        if stress >= 0.7 or (arousal >= 0.75 and valence < 0.0):
            label = "stressed"
            confidence = max(stress, arousal)
            sources.append("prosody")
        elif valence >= 0.25 or attention == "present":
            label = "engaged"
            confidence = max(0.55, valence)
            if prosody_hints:
                sources.append("prosody")
        else:
            label = "calm"
            confidence = 0.5
            if prosody_hints:
                sources.append("prosody")

        if expression in {"tired", "sad", "concerned", "angry", "stressed"}:
            if label == "calm":
                label = "concerned"
                confidence = max(confidence, 0.6)
            if "vision" not in sources:
                sources.append("vision")
        elif attention and "vision" not in sources and label != "stressed":
            sources.append("vision")

        return {
            "label": label,
            "confidence": round(min(max(confidence, 0.0), 1.0), 2),
            "sources": sources,
        }

    def _noise_policy(self, environment_hints: Mapping[str, Any]) -> dict[str, Any]:
        noise_db = _as_float(environment_hints.get("noise_db"), default=-1.0)
        noise_level = str(environment_hints.get("noise_level", "")).lower()

        if noise_db >= self.high_noise_db or noise_level in {"high", "noisy", "loud"}:
            return {
                "mode": "reduce_verbal_density",
                "reason": "high_environment_noise",
                "noise_db": noise_db if noise_db >= 0 else None,
            }
        if noise_db >= self.medium_noise_db or noise_level == "medium":
            return {
                "mode": "confirm_hearing",
                "reason": "moderate_environment_noise",
                "noise_db": noise_db if noise_db >= 0 else None,
            }
        return {
            "mode": "normal",
            "reason": "clear_environment",
            "noise_db": noise_db if noise_db >= 0 else None,
        }

    def _response_style(
        self,
        *,
        emotion_hint: Mapping[str, Any],
        noise_policy: Mapping[str, Any],
    ) -> dict[str, Any]:
        stressed = emotion_hint.get("label") in {"stressed", "concerned"}
        noisy = noise_policy.get("mode") in {"reduce_verbal_density", "confirm_hearing"}
        return {
            "tone": "gentle" if stressed else "warm",
            "pace": "slow" if stressed else "normal",
            "brevity": "concise" if noisy else "normal",
            "micro_ack": stressed or noisy,
        }


__all__ = ["EmotionContextBuilder"]
