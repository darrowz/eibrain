from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import ceil, isfinite
from numbers import Real
from typing import Any


DEFAULT_THRESHOLDS = {
    "asrFinalMs": 800.0,
    "firstTokenMs": 700.0,
    "firstAudioMs": 2000.0,
    "interruptStopMs": 300.0,
}

_METRIC_LABELS = {
    "wakeToListenMs": "wake_to_listen",
    "asrFinalMs": "asr_final",
    "firstTokenMs": "first_token",
    "firstAudioMs": "first_audio",
    "interruptStopMs": "interrupt_stop",
}


def summarize_voice_chain(turns: Iterable[Mapping[str, Any]], *, thresholds: Mapping[str, Any] | None = None) -> dict[str, Any]:
    turn_list = [turn for turn in turns if isinstance(turn, Mapping)]
    threshold_values = _coerce_thresholds(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    fields = _metric_fields(turn_list, threshold_values)

    metrics = {}
    for field in fields:
        values = [_as_float(turn[field]) for turn in turn_list if field in turn]
        numeric_values = [value for value in values if value is not None]
        if not numeric_values:
            continue
        threshold = threshold_values.get(field)
        p95 = _nearest_rank_p95(numeric_values)
        metrics[field] = {
            "count": len(numeric_values),
            "avg": sum(numeric_values) / len(numeric_values),
            "p95": p95,
            "threshold": threshold,
            "pass": p95 <= threshold if threshold is not None else None,
        }

    round_leak_count = sum(1 for turn in turn_list if turn.get("roundLeak") is True)
    turn_count = len(turn_list)
    return {
        "turnCount": turn_count,
        "roundLeakCount": round_leak_count,
        "roundLeakRate": round_leak_count / turn_count if turn_count else 0.0,
        "metrics": metrics,
        "bottleneck": _bottleneck(metrics),
    }


def _coerce_thresholds(thresholds: Mapping[str, Any]) -> dict[str, float]:
    coerced = {}
    for field, value in thresholds.items():
        threshold = _as_float(value)
        if threshold is not None:
            coerced[str(field)] = threshold
    return coerced


def _metric_fields(turns: list[Mapping[str, Any]], thresholds: Mapping[str, float]) -> list[str]:
    fields = list(_METRIC_LABELS)
    extra_fields = set(thresholds) - set(fields)
    for turn in turns:
        extra_fields.update(str(field) for field in turn if str(field).endswith("Ms") and field not in _METRIC_LABELS)
    fields.extend(sorted(extra_fields))
    return fields


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if isfinite(number) else None


def _nearest_rank_p95(values: list[float]) -> float:
    ordered = sorted(values)
    rank = max(1, ceil(0.95 * len(ordered)))
    return ordered[rank - 1]


def _bottleneck(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    selected = None
    selected_ratio = None
    for field, metric in metrics.items():
        threshold = metric.get("threshold")
        p95 = metric.get("p95")
        if threshold is None or threshold <= 0 or p95 is None:
            continue
        ratio = p95 / threshold
        if selected_ratio is None or ratio > selected_ratio:
            selected = (field, metric, ratio)
            selected_ratio = ratio

    if selected is None:
        return {"field": None, "label": None, "p95": None, "threshold": None, "ratio": None}

    field, metric, ratio = selected
    return {
        "field": field,
        "label": _METRIC_LABELS.get(field, field),
        "p95": metric["p95"],
        "threshold": metric["threshold"],
        "ratio": ratio,
    }
