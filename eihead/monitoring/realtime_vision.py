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

    first_non_live_payload: dict[str, Any] | None = None
    for attr_name in REALTIME_VISION_ATTRS:
        if not hasattr(app, attr_name):
            continue
        source = getattr(app, attr_name)
        raw_payload = source() if callable(source) else source
        payload = build_realtime_vision_payload(
            raw_payload,
            timestamp=timestamp,
            source=attr_name,
            wired=raw_payload is not None,
        )
        if payload.get("wired") is True:
            return payload
        if first_non_live_payload is None:
            first_non_live_payload = payload

    if first_non_live_payload is not None:
        return first_non_live_payload

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
    diagnostic = _build_realtime_diagnostic(
        serialized_observation,
        status=derived_status if serialized_observation is not None else "not_wired",
        wired=is_wired,
        timestamp=timestamp,
    )
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
        "diagnostic": diagnostic,
        "backend": diagnostic["backend"],
        "frame_id": diagnostic["frame_id"],
        "fps": diagnostic["fps"],
        "last_frame_age": diagnostic["last_frame_age"],
        "last_frame_age_s": diagnostic["last_frame_age_s"],
        "detections": diagnostic["detections"],
        "boxes": diagnostic["boxes"],
        "scores": diagnostic["scores"],
        "top_detection": diagnostic["top_detection"],
        "not_wired": diagnostic["not_wired"],
        "placeholder": diagnostic["placeholder"],
        "stale": diagnostic["stale"],
        "compat_static_active": diagnostic["compat_static"],
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
    placeholder = _truthy(observation.get("placeholder", False))
    not_wired = _truthy(observation.get("not_wired", False))
    compatibility_mode = _truthy(observation.get("compatibility_mode", False))
    stale = _is_stale_observation(observation, status=status)

    if not_wired or placeholder or status in {"not_wired", "offline", "missing", "unavailable"}:
        return "not_wired", False, "eye.realtime payload is present but not ready"
    if kind == "vision_observation" or mode == VISION_STATIC_COMPAT_MODE or primary_mode is False or compatibility_mode:
        return "compat_static", False, "compat/static vision payload is not accepted as realtime eye data"
    if stale and (kind == "realtime_vision_observation" or mode in {VISION_REALTIME_MODE, "realtime_stream"}):
        return "stale", True, ""
    if kind == "realtime_vision_observation" or mode in {VISION_REALTIME_MODE, "realtime_stream"}:
        return "wired", True, ""
    return "unknown", False, "payload is not a recognized realtime eye observation"


def _build_realtime_diagnostic(
    observation: Mapping[str, Any] | None,
    *,
    status: str,
    wired: bool,
    timestamp: float,
) -> dict[str, Any]:
    detections = _normalized_detections(_first_present(observation, "detections") if observation else None)
    boxes = [box for box in (_normalize_box(item.get("bbox")) for item in detections) if box is not None]
    scores = [score for score in (_detection_score(item) for item in detections) if score is not None]
    top_detection = _normalized_detection(_first_present(observation, "top_detection") if observation else None)
    if top_detection is None:
        top_detection = _top_detection(detections)

    pipeline_status = _string_or_none(_first_present(observation, "status") if observation else None)
    last_frame_age = _last_frame_age(observation, timestamp=timestamp)
    placeholder = _truthy(_first_present(observation, "placeholder")) if observation else False
    not_wired = status == "not_wired" or bool(observation and _truthy(_first_present(observation, "not_wired")))
    compat_static = status == "compat_static" or _is_compat_static_observation(observation)
    stale = status == "stale" or _is_stale_observation(observation, status=pipeline_status or "")

    return {
        "status": status,
        "pipeline_status": pipeline_status,
        "wired": bool(wired),
        "not_wired": bool(not_wired),
        "placeholder": bool(placeholder),
        "compat_static": bool(compat_static),
        "stale": bool(stale),
        "backend": _backend(observation),
        "frame_id": _frame_id(observation),
        "fps": _number_or_none(_first_nested_present(observation, "fps")),
        "last_frame_age": last_frame_age,
        "last_frame_age_s": last_frame_age,
        "detection_count": len(detections),
        "boxes": boxes,
        "scores": scores,
        "top_detection": top_detection,
        "detections": detections,
    }


def _is_compat_static_observation(observation: Mapping[str, Any] | None) -> bool:
    if observation is None:
        return False
    kind = str(observation.get("kind", "") or "").strip().lower()
    mode = str(observation.get("mode", "") or "").strip().lower()
    primary_mode = observation.get("primary_mode")
    compatibility_mode = _truthy(observation.get("compatibility_mode", False))
    return (
        kind == "vision_observation"
        or mode in {VISION_STATIC_COMPAT_MODE, "compat_static_frame"}
        or primary_mode is False
        or compatibility_mode
    )


def _is_stale_observation(observation: Mapping[str, Any] | None, *, status: str = "") -> bool:
    if observation is None:
        return False
    if str(status or "").strip().lower() == "stale":
        return True
    if _truthy(observation.get("stale")):
        return True
    for key in ("health", "stream", "payload"):
        nested = observation.get(key)
        if isinstance(nested, Mapping) and _truthy(nested.get("stale")):
            return True
    return False


def _normalized_detections(raw_detections: Any) -> list[dict[str, Any]]:
    if raw_detections is None:
        return []
    if isinstance(raw_detections, Mapping) or isinstance(raw_detections, (str, bytes)):
        raw_items = [raw_detections]
    else:
        try:
            raw_items = list(raw_detections)
        except TypeError:
            raw_items = [raw_detections]
    return [
        detection
        for detection in (_normalized_detection(item) for item in raw_items)
        if detection is not None
    ]


def _normalized_detection(raw_detection: Any) -> dict[str, Any] | None:
    if raw_detection is None:
        return None
    if isinstance(raw_detection, Mapping):
        detection = {str(k): _json_ready(v) for k, v in raw_detection.items()}
    else:
        detection = _json_ready(raw_detection)
        if not isinstance(detection, Mapping):
            return {"value": detection, "payload_type": type(raw_detection).__name__}
        detection = {str(k): _json_ready(v) for k, v in detection.items()}
    if "bbox" in detection:
        normalized_box = _normalize_box(detection.get("bbox"))
        detection["bbox"] = normalized_box if normalized_box is not None else detection.get("bbox")
    return detection


def _normalize_box(raw_box: Any) -> dict[str, Any] | None:
    if raw_box is None:
        return None
    if isinstance(raw_box, Mapping):
        return {str(k): _json_ready(v) for k, v in raw_box.items()}
    if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
        x_min, y_min, x_max, y_max = raw_box
        return {
            "x_min": _json_ready(x_min),
            "y_min": _json_ready(y_min),
            "x_max": _json_ready(x_max),
            "y_max": _json_ready(y_max),
        }
    return None


def _detection_score(detection: Mapping[str, Any]) -> float | None:
    score = _first_present(detection, "score")
    if score is None:
        score = _first_present(detection, "confidence")
    numeric_score = _number_or_none(score)
    return round(numeric_score, 6) if numeric_score is not None else None


def _top_detection(detections: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not detections:
        return None
    return max(detections, key=lambda item: _detection_score(item) if _detection_score(item) is not None else -1.0)


def _backend(observation: Mapping[str, Any] | None) -> str | None:
    value = _first_nested_present(observation, "backend")
    if value is None:
        stream = observation.get("stream") if observation else None
        if isinstance(stream, Mapping):
            value = stream.get("transport")
    return _string_or_none(value)


def _frame_id(observation: Mapping[str, Any] | None) -> str | None:
    return _string_or_none(_first_nested_present(observation, "frame_id", "last_frame_id"))


def _last_frame_age(observation: Mapping[str, Any] | None, *, timestamp: float) -> float | None:
    age = _number_or_none(_first_nested_present(observation, "last_frame_age", "last_frame_age_s"))
    if age is not None:
        return age
    captured_at_ts = _number_or_none(
        _first_nested_present(observation, "last_frame_captured_at_ts", "captured_at_ts")
    )
    if captured_at_ts is None or captured_at_ts > timestamp:
        return None
    return round(max(0.0, timestamp - captured_at_ts), 4)


def _first_nested_present(observation: Mapping[str, Any] | None, *keys: str) -> Any:
    value = _first_present(observation, *keys)
    if value is not None:
        return value
    if observation is None:
        return None
    for nested_key in ("health", "stream", "payload"):
        nested = observation.get(nested_key)
        if isinstance(nested, Mapping):
            value = _first_present(nested, *keys)
            if value is not None:
                return value
    return None


def _first_present(mapping: Mapping[str, Any] | None, *keys: str) -> Any:
    if mapping is None:
        return None
    for key in keys:
        if key in mapping and mapping[key] not in ("", None):
            return mapping[key]
    return None


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "stale"}
    return bool(value)


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
