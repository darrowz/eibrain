"""Native eihead mouth playback status contracts.

This module is intentionally standard-library only. It provides serializable
config, action summary, stop summary, and playback status helpers without
pulling in legacy body runtime implementations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any


PRIMARY_TTS_PROVIDER = "minimax"
FALLBACK_TTS_PROVIDERS = {"noop", "espeak"}
BUSY_STATUSES = {"synthesizing", "playing"}


@dataclass(frozen=True, slots=True)
class MouthTtsConfig:
    provider: str = ""
    model: str = ""
    voice_id: str = ""
    output_device: str = ""
    api_base_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SpeakActionSummary:
    action_type: str = "speak"
    text: str = ""
    text_preview: str = ""
    text_char_count: int = 0
    voice_id: str = ""
    session_id: str = ""
    provider: str = ""
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StopSpeechSummary:
    status: str = "stopped"
    success: bool = False
    busy: bool = False
    reason: str = ""
    last_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SpeechPlaybackStatus:
    status: str = "not_wired"
    provider: str = ""
    model: str = ""
    voice_id: str = ""
    text_preview: str = ""
    text_char_count: int = 0
    synthesis_elapsed_ms: int | None = None
    playback_elapsed_ms: int | None = None
    total_elapsed_ms: int | None = None
    busy: bool = False
    last_error: str = ""
    not_wired: bool = True
    readiness_message: str = "mouth TTS/playback is not wired"
    health: str = "not_wired"
    data_status: str = "not_wired"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_mouth_status(
    *,
    config: MouthTtsConfig | Mapping[str, Any] | None = None,
    status: str = "idle",
    action: Mapping[str, Any] | Any | None = None,
    details: Mapping[str, Any] | Any | None = None,
    outcome: Mapping[str, Any] | Any | None = None,
    busy: bool | None = None,
    last_error: str | None = None,
) -> SpeechPlaybackStatus:
    config_payload = _mapping(config)
    detail_payload = _merge_details(details, outcome)
    action_summary = summarize_speak_action(action) if action is not None else SpeakActionSummary()

    provider = _string_or_default(detail_payload.get("provider"), _string_or_default(config_payload.get("provider"), ""))
    model = _string_or_default(detail_payload.get("model"), _string_or_default(config_payload.get("model"), ""))
    voice_id = _string_or_default(
        detail_payload.get("voice_id"),
        _string_or_default(action_summary.voice_id, _string_or_default(config_payload.get("voice_id"), "")),
    )

    effective_status, health, data_status, not_wired, readiness_message = _provider_state(
        provider=provider,
        requested_status=status,
        config_payload=config_payload,
    )
    error_text = _string_or_default(last_error, _string_or_default(detail_payload.get("last_error"), ""))
    if error_text:
        effective_status = "error"
    effective_busy = busy if busy is not None else effective_status in BUSY_STATUSES

    text_value = _string_or_default(
        detail_payload.get("text"),
        action_summary.text,
    )
    preview = _string_or_default(
        detail_payload.get("text_preview"),
        action_summary.text_preview or _preview_text(text_value),
    )
    char_count = _safe_int(
        detail_payload.get("text_char_count"),
        default=action_summary.text_char_count or len(text_value),
    )

    return SpeechPlaybackStatus(
        status=effective_status,
        provider=provider,
        model=model,
        voice_id=voice_id,
        text_preview=preview,
        text_char_count=char_count,
        synthesis_elapsed_ms=_safe_int(detail_payload.get("synthesis_elapsed_ms")),
        playback_elapsed_ms=_safe_int(detail_payload.get("playback_elapsed_ms")),
        total_elapsed_ms=_safe_int(detail_payload.get("total_elapsed_ms")),
        busy=bool(effective_busy),
        last_error=error_text,
        not_wired=bool(not_wired),
        readiness_message=readiness_message,
        health=health,
        data_status=data_status,
    )


def summarize_speak_action(action: Mapping[str, Any] | Any | None) -> SpeakActionSummary:
    payload = _mapping(action)
    text = _string_or_default(_action_value(payload, "text"), "")
    voice_id = _string_or_default(_action_value(payload, "voice_id"), "")
    session_id = _string_or_default(_action_value(payload, "session_id"), "")
    provider = _string_or_default(_action_value(payload, "provider"), "")
    model = _string_or_default(_action_value(payload, "model"), "")
    return SpeakActionSummary(
        action_type=_string_or_default(payload.get("type") or payload.get("kind"), "speak"),
        text=text,
        text_preview=_preview_text(text),
        text_char_count=len(text),
        voice_id=voice_id,
        session_id=session_id,
        provider=provider,
        model=model,
    )


def summarize_stop_speech_result(result: Mapping[str, Any] | Any | None) -> StopSpeechSummary:
    payload = _mapping(result)
    detail_payload = _mapping(payload.get("details"))
    success = bool(payload.get("success", payload.get("ok", False)))
    raw_status = _string_or_default(payload.get("status"), "")
    last_error = _string_or_default(
        payload.get("last_error"),
        _string_or_default(detail_payload.get("error"), ""),
    )
    status = "stopped" if success and raw_status not in {"error", "failed"} else "error"
    return StopSpeechSummary(
        status=status,
        success=success,
        busy=False if success else True,
        reason=_string_or_default(detail_payload.get("reason"), raw_status),
        last_error=last_error,
    )


def _provider_state(
    *,
    provider: str,
    requested_status: str,
    config_payload: Mapping[str, Any],
) -> tuple[str, str, str, bool, str]:
    normalized = provider.strip().lower()
    if not normalized:
        return (
            "not_wired",
            "not_wired",
            "not_wired",
            True,
            "mouth TTS provider is not configured",
        )
    if normalized == PRIMARY_TTS_PROVIDER:
        return (
            requested_status,
            "online",
            "live",
            False,
            f"{PRIMARY_TTS_PROVIDER} TTS is configured as the primary provider",
        )
    if normalized in FALLBACK_TTS_PROVIDERS:
        return (
            requested_status,
            "degraded",
            "compat",
            False,
            f"{normalized} is a fallback/compat mouth provider, not the primary online TTS path",
        )
    return (
        requested_status,
        "degraded",
        "unknown",
        False,
        f"{normalized} is not recognized as the primary mouth TTS provider",
    )


def _merge_details(details: Mapping[str, Any] | Any | None, outcome: Mapping[str, Any] | Any | None) -> dict[str, Any]:
    merged = dict(_mapping(details))
    outcome_payload = _mapping(outcome)
    outcome_details = _mapping(outcome_payload.get("details"))
    for key, value in outcome_details.items():
        merged.setdefault(str(key), value)
    return merged


def _action_value(action: Mapping[str, Any], key: str) -> Any:
    if key in action:
        return action[key]
    params = action.get("params")
    if isinstance(params, Mapping) and key in params:
        return params[key]
    return None


def _mapping(value: Mapping[str, Any] | Any | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        payload = value.to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    if is_dataclass(value):
        return asdict(value)
    return {}


def _preview_text(text: str, *, limit: int = 48) -> str:
    trimmed = text.strip()
    if len(trimmed) <= limit:
        return trimmed
    return f"{trimmed[:limit].rstrip()}..."


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _string_or_default(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value)
    return text if text else default


__all__ = [
    "MouthTtsConfig",
    "PRIMARY_TTS_PROVIDER",
    "SpeechPlaybackStatus",
    "SpeakActionSummary",
    "StopSpeechSummary",
    "build_mouth_status",
    "summarize_speak_action",
    "summarize_stop_speech_result",
]
