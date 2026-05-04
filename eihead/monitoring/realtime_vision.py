"""Standard realtime eye/vision monitor payload helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

from eihead.protocol import (
    EYE_REALTIME_CHANNEL,
    VISION_REALTIME_ALIAS,
    VISION_REALTIME_MODE,
    VISION_STATIC_COMPAT_MODE,
)
from eihead.protocol.base import serialize_message


REALTIME_VISION_SCHEMA = "eihead.monitor.vision_realtime.v1"
REALTIME_VISION_ATTRS = (
    "eye_realtime",
    "vision_realtime",
    "realtime_vision",
    "latest_eye_realtime",
    "latest_vision_realtime",
    "latest_realtime_vision",
)


def realtime_vision_payload_from_app(app: Any, *, timestamp: float) -> dict[str, Any]:
    """Read the first realtime eye/vision hook from an app and standardize it."""

    for attr_name in REALTIME_VISION_ATTRS:
        if not hasattr(app, attr_name):
            continue
        source = getattr(app, attr_name)
        raw_payload = source() if callable(source) else source
        return build_realtime_vision_payload(
            raw_payload,
            timestamp=timestamp,
            source=attr_name,
            wired=raw_payload is not None,
        )

    return build_realtime_vision_payload(
        None,
        timestamp=timestamp,
        source=None,
        wired=False,
    )


def build_realtime_vision_payload(
    observation: Any = None,
    *,
    timestamp: float,
    source: str | None = None,
    wired: bool | None = None,
) -> dict[str, Any]:
    """Return the monitor JSON envelope for primary realtime eye state."""

    serialized_observation = _serialize_observation(observation) if observation is not None else None
    derived_status, derived_wired, derived_message = _derive_realtime_status(serialized_observation)
    is_wired = derived_wired if wired is None else bool(wired and derived_wired)
    payload: dict[str, Any] = {
        "schema": REALTIME_VISION_SCHEMA,
        "runtime": "eihead",
        "status": derived_status if serialized_observation is not None else "not_wired",
        "wired": is_wired,
        "source": source,
        "channel": EYE_REALTIME_CHANNEL,
        "aliases": [VISION_REALTIME_ALIAS],
        "primary_mode": VISION_REALTIME_MODE,
        "compat_static": {"mode": VISION_STATIC_COMPAT_MODE, "primary": False},
        "captured_at_ts": timestamp,
        "observation": serialized_observation,
    }
    if not is_wired:
        if derived_message:
            payload["message"] = derived_message
        elif source:
            payload["message"] = f"runtime app.{source} did not return an eye.realtime payload"
        else:
            payload["message"] = "runtime app does not expose eye.realtime or vision.realtime payload"
    return payload


def _derive_realtime_status(observation: Mapping[str, Any] | None) -> tuple[str, bool, str]:
    if observation is None:
        return "not_wired", False, ""
    status = str(observation.get("status", "") or "").strip().lower()
    kind = str(observation.get("kind", "") or "").strip().lower()
    mode = str(observation.get("mode", "") or "").strip().lower()
    primary_mode = observation.get("primary_mode")
    placeholder = bool(observation.get("placeholder", False))
    not_wired = bool(observation.get("not_wired", False))
    compatibility_mode = bool(observation.get("compatibility_mode", False))

    if not_wired or placeholder or status in {"not_wired", "offline", "missing", "unavailable"}:
        return "not_wired", False, "eye.realtime payload is present but not ready"
    if kind == "vision_observation" or mode == VISION_STATIC_COMPAT_MODE or primary_mode is False or compatibility_mode:
        return "compat_static", False, "compat/static vision payload is not accepted as realtime eye data"
    if kind == "realtime_vision_observation" or mode in {VISION_REALTIME_MODE, "realtime_stream"}:
        return "wired", True, ""
    return "unknown", False, "payload is not a recognized realtime eye observation"


def _serialize_observation(observation: Any) -> dict[str, Any]:
    if isinstance(observation, Mapping):
        return {str(k): _json_ready(v) for k, v in observation.items()}
    try:
        payload = serialize_message(observation)
    except TypeError:
        return {
            "value": _json_ready(observation),
            "payload_type": type(observation).__name__,
        }
    return {str(k): _json_ready(v) for k, v in payload.items()}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if is_dataclass(value):
        return _json_ready(asdict(value))
    return str(value)


__all__ = [
    "REALTIME_VISION_ATTRS",
    "REALTIME_VISION_SCHEMA",
    "build_realtime_vision_payload",
    "realtime_vision_payload_from_app",
]
