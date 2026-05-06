"""Vision pipeline soak-test summaries.

This module is intentionally hardware-agnostic: callers can feed samples from
the live Hailo service, web monitor snapshots, or fixture data.
"""

from __future__ import annotations

from math import ceil
from typing import Any, Iterable


def summarize_vision_soak(
    samples: Iterable[dict[str, Any]],
    *,
    target_fps: float | None = None,
    min_fps_ratio: float = 0.8,
    max_p95_frame_age_ms: float = 500.0,
    max_drop_rate: float = 0.1,
    min_service_ok_ratio: float = 0.95,
) -> dict[str, Any]:
    """Summarize health from vision telemetry samples.

    ``drop_rate`` is intentionally reported as dropped frames per sample because
    current runtime telemetry exposes cumulative-ish sample counters rather than
    total produced frames.
    """

    rows = [dict(sample) for sample in samples]
    if not rows:
        return {
            "pass": False,
            "sample_count": 0,
            "fps": _stats([]),
            "frame_age_ms": _stats([]),
            "loop_elapsed_ms": _stats([]),
            "fps_ratio": 0.0,
            "drop_rate": 0.0,
            "stale_ratio": 0.0,
            "service_ok_ratio": 0.0,
            "bottleneck_reason": "no_samples",
        }

    fps_values = [_to_float(row.get("fps")) for row in rows if row.get("fps") is not None]
    frame_age_values = [
        _to_float(row.get("frame_age_ms"))
        for row in rows
        if row.get("frame_age_ms") is not None
    ]
    loop_values = [
        _to_float(row.get("loop_elapsed_ms"))
        for row in rows
        if row.get("loop_elapsed_ms") is not None
    ]
    effective_target_fps = _resolve_target_fps(rows, target_fps)
    fps_avg = _average(fps_values)
    fps_ratio = fps_avg / effective_target_fps if effective_target_fps > 0 else 0.0
    total_drops = sum(max(0.0, _to_float(row.get("dropped_frames", 0.0))) for row in rows)
    drop_rate = total_drops / len(rows)
    stale_count = sum(1 for value in frame_age_values if value > max_p95_frame_age_ms)
    stale_ratio = stale_count / len(rows)
    service_ok_count = sum(1 for row in rows if _service_is_ok(row.get("service_state")))
    service_ok_ratio = service_ok_count / len(rows)

    fps_summary = _stats(fps_values)
    frame_age_summary = _stats(frame_age_values)
    service_unstable = service_ok_ratio < min_service_ok_ratio
    low_fps = fps_ratio < min_fps_ratio
    stale_frames = frame_age_summary["p95"] > max_p95_frame_age_ms
    frame_drops = drop_rate > max_drop_rate
    reason = _bottleneck_reason(
        service_unstable=service_unstable,
        low_fps=low_fps,
        stale_frames=stale_frames,
        frame_drops=frame_drops,
    )

    return {
        "pass": reason == "healthy",
        "sample_count": len(rows),
        "fps": fps_summary,
        "frame_age_ms": frame_age_summary,
        "loop_elapsed_ms": _stats(loop_values),
        "target_fps": _round(effective_target_fps),
        "fps_ratio": fps_ratio,
        "drop_rate": drop_rate,
        "stale_ratio": stale_ratio,
        "service_ok_ratio": service_ok_ratio,
        "bottleneck_reason": reason,
    }


def _resolve_target_fps(samples: list[dict[str, Any]], explicit: float | None) -> float:
    if explicit is not None:
        return max(0.0, _to_float(explicit))
    for row in samples:
        value = row.get("target_fps")
        if value is not None:
            return max(0.0, _to_float(value))
    return 10.0


def _service_is_ok(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"ok", "active", "running", "healthy", "ready"}


def _bottleneck_reason(
    *,
    service_unstable: bool,
    low_fps: bool,
    stale_frames: bool,
    frame_drops: bool,
) -> str:
    if service_unstable:
        return "service_unstable"
    if low_fps:
        return "low_fps"
    if stale_frames:
        return "stale_frames"
    if frame_drops:
        return "frame_drops"
    return "healthy"


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    sorted_values = sorted(values)
    return {
        "avg": _round(_average(sorted_values)),
        "p50": _round(_percentile(sorted_values, 0.50)),
        "p95": _round(_percentile(sorted_values, 0.95)),
        "max": _round(sorted_values[-1]),
    }


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    index = max(0, min(len(sorted_values) - 1, ceil(percentile * len(sorted_values)) - 1))
    return sorted_values[index]


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _round(value: float) -> float:
    return round(float(value), 3)
