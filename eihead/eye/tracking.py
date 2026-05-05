"""Vision target selection helpers for stable pan-only following."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_TRACKING_LABELS = ("person", "face")


@dataclass(frozen=True)
class TrackingTarget:
    bbox: tuple[float, float, float, float]
    center_x: float
    center_y: float
    horizontal_error: float
    score: float
    label: str
    track_id: Any | None = None
    frame_id: Any | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "center_x": self.center_x,
            "center_y": self.center_y,
            "horizontal_error": self.horizontal_error,
            "score": self.score,
            "label": self.label,
            "track_id": self.track_id,
            "frame_id": self.frame_id,
        }


def select_tracking_target(
    detections: Iterable[Mapping[str, Any]],
    *,
    frame_width: float,
    frame_height: float,
    frame_id: Any | None = None,
    preferred_labels: Sequence[str] = DEFAULT_TRACKING_LABELS,
) -> TrackingTarget | None:
    """Pick one detector target for visual following.

    The selector prefers human/face-like labels, then confidence, area, and
    center bias. That keeps the neck planner from hopping to irrelevant high
    confidence objects at frame edges.
    """

    frame_width = float(frame_width)
    frame_height = float(frame_height)
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame_width and frame_height must be positive")

    preferred = {label.lower() for label in preferred_labels}
    candidates: list[tuple[tuple[float, float, float, float], Mapping[str, Any], float, float, float, float]] = []
    has_preferred = False

    for detection in detections:
        bbox = _coerce_bbox(detection, frame_width=frame_width, frame_height=frame_height)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            continue

        label = _detection_label(detection)
        score = _as_float(detection.get("score", detection.get("confidence", 0.0)), default=0.0)
        center_x = x1 + width / 2.0
        horizontal_error = (center_x - frame_width / 2.0) / (frame_width / 2.0)
        area_ratio = min(1.0, (width * height) / (frame_width * frame_height))
        center_bias = 1.0 - min(1.0, abs(horizontal_error))
        has_preferred = has_preferred or label in preferred
        candidates.append((bbox, detection, score, area_ratio, center_bias, horizontal_error))

    if has_preferred:
        candidates = [candidate for candidate in candidates if _detection_label(candidate[1]) in preferred]
    if not candidates:
        return None

    bbox, detection, score, _area_ratio, _center_bias, horizontal_error = max(
        candidates,
        key=lambda candidate: (
            candidate[2],
            candidate[3],
            candidate[4],
            -abs(candidate[5]),
        ),
    )
    x1, y1, x2, y2 = bbox
    return TrackingTarget(
        bbox=bbox,
        center_x=x1 + (x2 - x1) / 2.0,
        center_y=y1 + (y2 - y1) / 2.0,
        horizontal_error=horizontal_error,
        score=score,
        label=_detection_label(detection),
        track_id=detection.get("trackId", detection.get("track_id", detection.get("id"))),
        frame_id=detection.get("frameId", detection.get("frame_id", frame_id)),
    )


def _coerce_bbox(
    detection: Mapping[str, Any],
    *,
    frame_width: float,
    frame_height: float,
) -> tuple[float, float, float, float] | None:
    raw_bbox = detection.get("bbox", detection.get("box", detection.get("xyxy")))
    if raw_bbox is None:
        return None
    if isinstance(raw_bbox, Mapping):
        bbox = _coerce_mapping_bbox(raw_bbox)
    elif isinstance(raw_bbox, Sequence) and not isinstance(raw_bbox, (str, bytes)) and len(raw_bbox) >= 4:
        bbox = tuple(float(value) for value in raw_bbox[:4])
    else:
        return None
    return _scale_normalized_bbox(bbox, frame_width=frame_width, frame_height=frame_height)


def _coerce_mapping_bbox(raw_bbox: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    for keys in (
        ("x1", "y1", "x2", "y2"),
        ("x_min", "y_min", "x_max", "y_max"),
        ("xmin", "ymin", "xmax", "ymax"),
        ("left", "top", "right", "bottom"),
    ):
        if all(key in raw_bbox for key in keys):
            return tuple(float(raw_bbox[key]) for key in keys)  # type: ignore[return-value]
    if all(key in raw_bbox for key in ("x", "y", "w", "h")):
        x = float(raw_bbox["x"])
        y = float(raw_bbox["y"])
        return (x, y, x + float(raw_bbox["w"]), y + float(raw_bbox["h"]))
    if all(key in raw_bbox for key in ("x", "y", "width", "height")):
        x = float(raw_bbox["x"])
        y = float(raw_bbox["y"])
        return (x, y, x + float(raw_bbox["width"]), y + float(raw_bbox["height"]))
    return None


def _scale_normalized_bbox(
    bbox: tuple[float, float, float, float],
    *,
    frame_width: float,
    frame_height: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
        return (x1 * frame_width, y1 * frame_height, x2 * frame_width, y2 * frame_height)
    return bbox


def _detection_label(detection: Mapping[str, Any]) -> str:
    return str(detection.get("label", detection.get("name", detection.get("class", "")))).lower()


def _as_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
