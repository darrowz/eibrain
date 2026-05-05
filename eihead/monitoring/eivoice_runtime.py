"""EIVoice runtime monitoring normalization."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping


QUEUE_NAMES = (
    "opus_encode_queue",
    "ws_send_queue",
    "opus_decode_queue",
    "audio_playback_queue",
)


def build_eivoice_runtime_panel(status: dict[str, Any]) -> dict[str, Any]:
    """Build a Web-friendly EIVoice runtime diagnostics panel."""

    runtime = _mapping(status)
    state = _text(runtime.get("state") or runtime.get("runtime_state") or runtime.get("status"))
    conversation_state = _text(
        runtime.get("conversationState")
        or runtime.get("conversation_state")
        or runtime.get("dialogue_state")
        or runtime.get("mode")
        or state,
        default="unknown",
    )
    queues = {name: _normalize_queue(name, runtime) for name in QUEUE_NAMES}
    dropped_total = sum(queue["droppedOldest"] + queue["droppedNewest"] for queue in queues.values())
    has_audio_frontend = any(key in runtime for key in ("audio_frontend", "acousticFrontend"))
    audio_frontend = _normalize_audio_frontend(runtime)
    wakeword = dict(_mapping(runtime.get("wakeword") or runtime.get("wake_word") or runtime.get("wakeword_buffer")))

    warnings: list[str] = []
    if not state:
        warnings.append("runtime state is missing")
    if dropped_total > 0:
        warnings.append(f"queue drops detected: {dropped_total}")
    if state and not has_audio_frontend:
        warnings.append("audio frontend readiness is missing")
    warnings.extend(str(item) for item in audio_frontend.get("warnings", []) if item)
    if _component_unavailable(audio_frontend.get("aec")):
        warnings.append("AEC unavailable")
    if _component_unavailable(audio_frontend.get("ns")):
        warnings.append("NS unavailable")
    if _component_unavailable(audio_frontend.get("vad")):
        warnings.append("VAD unavailable")
    if _component_unavailable(audio_frontend.get("loopback")):
        warnings.append("loopback unavailable")
    warnings = list(dict.fromkeys(warnings))

    health = "healthy"
    if (
        dropped_total > 0
        or (state and not has_audio_frontend)
        or _component_unavailable(audio_frontend.get("aec"))
        or _component_unavailable(audio_frontend.get("ns"))
        or _component_unavailable(audio_frontend.get("vad"))
        or _component_unavailable(audio_frontend.get("loopback"))
    ):
        health = "degraded"
    elif not state:
        health = "waiting"

    return {
        "state": state or "waiting",
        "conversationState": conversation_state,
        "queueSummary": _queue_summary(queues),
        "queues": queues,
        "droppedTotal": dropped_total,
        "audioFrontend": audio_frontend,
        "wakeword": wakeword,
        "health": health,
        "warnings": warnings,
    }


def _normalize_queue(name: str, runtime: Mapping[str, Any]) -> dict[str, Any]:
    queue = _queue_source(name, runtime)
    depth = _number(
        queue.get("depth")
        or queue.get("size")
        or queue.get("qsize")
        or queue.get("current_depth"),
        default=0,
    )
    capacity = _number(
        queue.get("capacity")
        or queue.get("maxsize")
        or queue.get("max_size")
        or queue.get("limit"),
        default=0,
    )
    dropped_oldest = _number(
        queue.get("droppedOldest")
        or queue.get("dropped_oldest")
        or queue.get("drop_oldest")
        or _mapping(queue.get("dropped")).get("oldest"),
        default=0,
    )
    dropped_newest = _number(
        queue.get("droppedNewest")
        or queue.get("dropped_newest")
        or queue.get("drop_newest")
        or _mapping(queue.get("dropped")).get("newest"),
        default=0,
    )
    return {
        "depth": depth,
        "capacity": capacity,
        "fillRatio": _fill_ratio(depth, capacity),
        "policy": _text(
            queue.get("policy")
            or queue.get("full_policy")
            or queue.get("drop_policy")
            or queue.get("overflow_policy"),
            default="unknown",
        ),
        "droppedOldest": dropped_oldest,
        "droppedNewest": dropped_newest,
    }


def _queue_source(name: str, runtime: Mapping[str, Any]) -> Mapping[str, Any]:
    queues = _mapping(runtime.get("queues") or runtime.get("queue_status") or runtime.get("queueStatus"))
    return _mapping(queues.get(name) or runtime.get(name))


def _queue_summary(queues: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    total_depth = sum(_number(queue.get("depth"), default=0) for queue in queues.values())
    total_capacity = sum(_number(queue.get("capacity"), default=0) for queue in queues.values())
    max_fill = max((_float(queue.get("fillRatio"), default=0.0) for queue in queues.values()), default=0.0)
    return {
        "count": len(queues),
        "totalDepth": total_depth,
        "totalCapacity": total_capacity,
        "maxFillRatio": max_fill,
    }


def _normalize_audio_frontend(runtime: Mapping[str, Any]) -> dict[str, Any]:
    frontend = _mapping(runtime.get("audio_frontend") or runtime.get("acousticFrontend"))
    return {
        "aec": _normalize_component(frontend.get("aec")),
        "ns": _normalize_component(frontend.get("ns") or frontend.get("noise_suppression")),
        "vad": _normalize_component(frontend.get("vad")),
        "loopback": _normalize_component(frontend.get("loopback")),
        "warnings": _list(frontend.get("warnings")),
    }


def _normalize_component(value: Any) -> dict[str, Any]:
    component = dict(_mapping(value))
    if not component and value is not None:
        component["enabled"] = bool(value)
    return component


def _component_unavailable(value: Any) -> bool:
    component = _mapping(value)
    if component.get("enabled") is False:
        return True
    available = component.get("available")
    if available is False:
        return True
    state = _text(component.get("state") or component.get("status"))
    return state in {"unavailable", "missing", "disabled_by_platform"}


def _list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return []


def _fill_ratio(depth: int, capacity: int) -> float:
    if capacity <= 0:
        return 0.0
    return depth / capacity


def _mapping(value: Any) -> Mapping[str, Any]:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        return value
    return {}


def _text(value: Any, *, default: str = "") -> str:
    if value in (None, ""):
        return default
    return str(value)


def _number(value: Any, *, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, *, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
