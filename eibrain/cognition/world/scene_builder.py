"""Build a compact 2D scene graph from visual detections."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def build_scene_graph(
    visual_state: dict[str, object],
    *,
    previous_state: dict[str, object] | None = None,
    previous_scene: dict[str, object] | None = None,
) -> dict[str, object]:
    """Convert a vision state payload into stable objects, relations, and events."""

    objects = _build_objects(visual_state.get("objects") or visual_state.get("detections") or [])
    previous = previous_scene
    if previous is None and previous_state is not None:
        previous = build_scene_graph(previous_state)

    relations = _build_relations(objects)
    events = _build_events(objects, previous)
    change_score = _change_score(objects, previous, events)
    if previous is not None and change_score >= 0.6:
        events.append({"type": "significant_scene_change", "change_score": round(change_score, 3)})

    labels = [str(item["label"]) for item in objects]
    summary = _summary(labels, events)
    fingerprint_payload = {
        "objects": [
            {
                "label": item["label"],
                "stable_id": item["stable_id"],
                "region": item["region"],
                "bbox": item["bbox"],
            }
            for item in objects
        ],
        "relations": relations,
    }
    fingerprint = _digest(fingerprint_payload, length=16)

    return {
        "scene_id": f"scene:{fingerprint}",
        "objects": objects,
        "relations": relations,
        "events": events,
        "summary": summary,
        "fingerprint": fingerprint,
        "dedupe_key": f"world_scene:{fingerprint}",
        "change_score": round(change_score, 3),
    }


def should_write_world_observation(
    current: dict[str, object],
    previous: dict[str, object] | None,
    *,
    min_change_score: float = 0.35,
) -> bool:
    """Return True only when the scene changed enough to merit memory writeback."""

    if previous is None:
        return bool(current.get("objects") or current.get("events"))
    if current.get("fingerprint") == previous.get("fingerprint"):
        return False
    if any(str(event.get("type")) == "significant_scene_change" for event in _events(current)):
        return True
    return float(current.get("change_score", 0.0) or 0.0) >= min_change_score


def _build_objects(raw_objects: object) -> list[dict[str, object]]:
    if not isinstance(raw_objects, list):
        return []
    objects: list[dict[str, object]] = []
    for raw in raw_objects:
        if not isinstance(raw, dict):
            continue
        label = str(raw.get("label") or raw.get("name") or raw.get("class") or "").strip()
        bbox = _normalize_bbox(raw.get("bbox"))
        if not label or bbox is None:
            continue
        confidence = _coerce_float(raw.get("confidence", raw.get("score", 0.0)))
        center = _center(bbox)
        item: dict[str, object] = {
            "label": label,
            "confidence": confidence,
            "bbox": bbox,
            "model_id": str(raw.get("model_id") or raw.get("source") or ""),
            "stable_id": str(raw.get("stable_id") or _stable_id(label, center)),
            "region": str(raw.get("region") or _region(center)),
        }
        objects.append(item)
    return sorted(objects, key=lambda item: (str(item["label"]), str(item["stable_id"])))


def _build_relations(objects: list[dict[str, object]]) -> list[dict[str, object]]:
    relations: list[dict[str, object]] = []
    for subject in objects:
        for obj in objects:
            if subject["stable_id"] == obj["stable_id"]:
                continue
            subject_center = _center(subject["bbox"])
            object_center = _center(obj["bbox"])
            dx = object_center[0] - subject_center[0]
            dy = object_center[1] - subject_center[1]
            for relation in _relation_types(dx, dy):
                relations.append(
                    {
                        "subject_id": subject["stable_id"],
                        "subject_label": subject["label"],
                        "relation": relation,
                        "object_id": obj["stable_id"],
                        "object_label": obj["label"],
                    }
                )
    return sorted(
        relations,
        key=lambda item: (
            str(item["subject_id"]),
            str(item["relation"]),
            str(item["object_id"]),
        ),
    )


def _relation_types(dx: float, dy: float) -> list[str]:
    relations: list[str] = []
    if dx > 0.18:
        relations.append("left_of")
    if dx < -0.18:
        relations.append("right_of")
    if dy > 0.18:
        relations.append("above")
    if dy < -0.18:
        relations.append("below")
    if math.hypot(dx, dy) <= 0.6:
        relations.append("near")
    return relations


def _build_events(objects: list[dict[str, object]], previous_scene: dict[str, object] | None) -> list[dict[str, object]]:
    if previous_scene is None:
        return [
            {"type": "appeared", "object_id": str(item["stable_id"]), "label": str(item["label"])}
            for item in objects
        ]

    previous_objects = [item for item in previous_scene.get("objects", []) if isinstance(item, dict)]
    matches = _match_objects(objects, previous_objects)
    events: list[dict[str, object]] = []

    matched_current = {current_index for current_index, _ in matches}
    matched_previous = {previous_index for _, previous_index in matches}

    for current_index, previous_index in matches:
        current = objects[current_index]
        previous = previous_objects[previous_index]
        distance = _distance(_center(current["bbox"]), _center(previous["bbox"]))
        if distance >= 0.18 or current.get("region") != previous.get("region"):
            events.append(
                {
                    "type": "moved",
                    "object_id": str(current["stable_id"]),
                    "label": str(current["label"]),
                    "from_region": str(previous.get("region", "")),
                    "to_region": str(current.get("region", "")),
                    "distance": round(distance, 3),
                }
            )

    for index, item in enumerate(objects):
        if index not in matched_current:
            events.append({"type": "appeared", "object_id": str(item["stable_id"]), "label": str(item["label"])})

    for index, item in enumerate(previous_objects):
        if index not in matched_previous:
            events.append({"type": "disappeared", "object_id": str(item["stable_id"]), "label": str(item["label"])})

    return events


def _match_objects(
    current_objects: list[dict[str, object]],
    previous_objects: list[dict[str, object]],
) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for current_index, current in enumerate(current_objects):
        for previous_index, previous in enumerate(previous_objects):
            if current.get("label") != previous.get("label"):
                continue
            candidates.append(
                (
                    _distance(_center(current["bbox"]), _center(previous["bbox"])),
                    current_index,
                    previous_index,
                )
            )
    matches: list[tuple[int, int]] = []
    used_current: set[int] = set()
    used_previous: set[int] = set()
    for distance, current_index, previous_index in sorted(candidates):
        if distance > 0.55 or current_index in used_current or previous_index in used_previous:
            continue
        matches.append((current_index, previous_index))
        used_current.add(current_index)
        used_previous.add(previous_index)
    return matches


def _change_score(
    objects: list[dict[str, object]],
    previous_scene: dict[str, object] | None,
    events: list[dict[str, object]],
) -> float:
    if previous_scene is None:
        return 1.0 if objects else 0.0
    previous_count = len([item for item in previous_scene.get("objects", []) if isinstance(item, dict)])
    denominator = max(1, len(objects), previous_count)
    score = 0.0
    for event in events:
        event_type = str(event.get("type"))
        if event_type in {"appeared", "disappeared"}:
            score += 0.5
        elif event_type == "moved":
            score += min(0.5, float(event.get("distance", 0.0) or 0.0))
    return min(1.0, score / denominator)


def _summary(labels: list[str], events: list[dict[str, object]]) -> str:
    if labels:
        counts: dict[str, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        observed = ", ".join(
            f"{count} {label}" if count > 1 else label for label, count in sorted(counts.items())
        )
        base = f"Observed {observed}"
    else:
        base = "Observed empty visual scene"
    event_types = sorted({str(event.get("type")) for event in events if event.get("type")})
    if event_types:
        return f"{base}; events: {', '.join(event_types)}"
    return base


def _normalize_bbox(raw_bbox: object) -> dict[str, float] | None:
    if not isinstance(raw_bbox, dict):
        return None
    try:
        x_min = _clip01(float(raw_bbox.get("x_min", raw_bbox.get("xmin", raw_bbox.get("left", 0.0)))))
        y_min = _clip01(float(raw_bbox.get("y_min", raw_bbox.get("ymin", raw_bbox.get("top", 0.0)))))
        x_max = _clip01(float(raw_bbox.get("x_max", raw_bbox.get("xmax", raw_bbox.get("right", 0.0)))))
        y_max = _clip01(float(raw_bbox.get("y_max", raw_bbox.get("ymax", raw_bbox.get("bottom", 0.0)))))
    except (TypeError, ValueError):
        return None
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if y_max < y_min:
        y_min, y_max = y_max, y_min
    return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}


def _stable_id(label: str, center: tuple[float, float]) -> str:
    x_bucket = int(center[0] * 10)
    y_bucket = int(center[1] * 10)
    return f"{label}:{x_bucket:02d}:{y_bucket:02d}"


def _region(center: tuple[float, float]) -> str:
    x_name = "left" if center[0] < 0.33 else "center" if center[0] < 0.66 else "right"
    y_name = "top" if center[1] < 0.25 else "middle" if center[1] < 0.75 else "bottom"
    return f"{x_name}_{y_name}"


def _center(bbox: object) -> tuple[float, float]:
    if not isinstance(bbox, dict):
        return (0.0, 0.0)
    return (
        (_coerce_float(bbox.get("x_min")) + _coerce_float(bbox.get("x_max"))) / 2.0,
        (_coerce_float(bbox.get("y_min")) + _coerce_float(bbox.get("y_max"))) / 2.0,
    )


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _events(scene: dict[str, object]) -> list[dict[str, object]]:
    events = scene.get("events", [])
    return [item for item in events if isinstance(item, dict)] if isinstance(events, list) else []


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _digest(payload: object, *, length: int) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]
