"""Build a compact 2D scene graph from visual detections."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


_DEVICE_LABELS = {"phone", "mobile", "smartphone", "tablet", "laptop", "monitor", "screen", "device"}
_HAND_KEYPOINTS = {"left_wrist", "right_wrist", "left_hand", "right_hand", "hand", "wrist"}


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
            "model_id": str(raw.get("model_id") or raw.get("modelId") or raw.get("model") or ""),
            "stable_id": str(raw.get("stable_id") or _stable_id(label, center)),
            "region": str(raw.get("region") or _region(center)),
            "source": str(raw.get("source") or raw.get("provider") or ""),
            "provenance": _provenance(raw),
        }
        depth_m = _structured_depth_m(raw) or _optional_float(raw.get("depth_m", raw.get("distance_m", raw.get("z_m"))))
        if depth_m is not None:
            item["depth_m"] = round(depth_m, 3)
        item["distance_band"] = str(raw.get("distance_band") or raw.get("depth_band") or _distance_band(depth_m=depth_m, bbox=bbox))
        pose = _normalize_pose(raw.get("pose"))
        if pose:
            item["pose"] = pose
            item["keypoints"] = list(pose["keypoints"])
        clip_labels = _normalize_label_annotations(raw.get("clip_labels", raw.get("clipLabels")))
        if clip_labels:
            item["clip_labels"] = clip_labels
        semantic_labels = _normalize_semantic_labels(raw.get("semantic_labels", raw.get("semanticLabels")))
        if semantic_labels:
            item["semantic_labels"] = semantic_labels
        tracking_diagnostics = raw.get("tracking_diagnostics", raw.get("trackingDiagnostics"))
        if isinstance(tracking_diagnostics, dict):
            item["tracking_diagnostics"] = dict(tracking_diagnostics)
        looking_at_device = _optional_bool(raw.get("looking_at_device", raw.get("lookingAtDevice")))
        if looking_at_device is not None:
            item["looking_at_device"] = looking_at_device
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
            if _hand_near_object(subject, obj):
                relations.append(
                    {
                        "subject_id": subject["stable_id"],
                        "subject_label": subject["label"],
                        "relation": "hand_near_object",
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
        ] + _lightweight_events(objects, _build_relations(objects))

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

    return events + _lightweight_events(objects, _build_relations(objects))


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


def _lightweight_events(
    objects: list[dict[str, object]],
    relations: list[dict[str, object]],
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    objects_by_id = {str(item.get("stable_id")): item for item in objects}
    for item in objects:
        if item.get("looking_at_device") is True:
            target = _nearest_device(item, objects)
            event: dict[str, object] = {
                "type": "looking_at_device",
                "object_id": str(item["stable_id"]),
                "label": str(item["label"]),
                "confidence": item.get("confidence", 0.0),
                "source": item.get("source", ""),
                "model_id": item.get("model_id", ""),
                "provenance": dict(item.get("provenance")) if isinstance(item.get("provenance"), dict) else {},
            }
            if target is not None:
                event["target_id"] = str(target.get("stable_id", ""))
                event["target_label"] = str(target.get("label", ""))
            events.append(event)
    for relation in relations:
        if relation.get("relation") != "hand_near_object":
            continue
        subject = objects_by_id.get(str(relation.get("subject_id")))
        obj = objects_by_id.get(str(relation.get("object_id")))
        if subject is None or obj is None:
            continue
        events.append(
            {
                "type": "hand_near_object",
                "object_id": str(subject["stable_id"]),
                "label": str(subject["label"]),
                "target_id": str(obj["stable_id"]),
                "target_label": str(obj["label"]),
                "confidence": min(_coerce_float(subject.get("confidence")), _coerce_float(obj.get("confidence"))),
                "source": subject.get("source", ""),
                "model_id": subject.get("model_id", ""),
                "provenance": dict(subject.get("provenance")) if isinstance(subject.get("provenance"), dict) else {},
            }
        )
    return sorted(events, key=lambda item: (str(item.get("type")), str(item.get("object_id")), str(item.get("target_id", ""))))


def _normalize_pose(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict) or not isinstance(raw.get("keypoints"), list):
        return {}
    keypoints: list[dict[str, object]] = []
    for item in raw["keypoints"]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("label") or item.get("part") or "").strip()
        x = _optional_float(item.get("x"))
        y = _optional_float(item.get("y"))
        if not name or x is None or y is None:
            continue
        point: dict[str, object] = {"name": name, "x": round(_clip01(x), 4), "y": round(_clip01(y), 4)}
        confidence = _optional_float(item.get("confidence", item.get("score")))
        if confidence is not None:
            point["confidence"] = round(confidence, 3)
        keypoints.append(point)
    return {"keypoints": keypoints} if keypoints else {}


def _normalize_label_annotations(raw: object) -> list[dict[str, object]]:
    if not isinstance(raw, list):
        return []
    labels: list[dict[str, object]] = []
    for item in raw:
        if isinstance(item, dict):
            label = str(item.get("label") or item.get("name") or item.get("text") or "").strip()
            if not label:
                continue
            normalized: dict[str, object] = {"label": label}
            confidence = _optional_float(item.get("confidence", item.get("score")))
            if confidence is not None:
                normalized["confidence"] = round(confidence, 3)
            source = str(item.get("source") or "").strip()
            if source:
                normalized["source"] = source
            labels.append(normalized)
        else:
            label = str(item).strip()
            if label:
                labels.append({"label": label})
    return labels


def _normalize_semantic_labels(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    labels: list[str] = []
    for item in raw:
        label = str(item.get("label") if isinstance(item, dict) else item).strip()
        if label and label not in labels:
            labels.append(label)
    return labels


def _structured_depth_m(raw: dict[str, object]) -> float | None:
    depth = raw.get("depth")
    if isinstance(depth, dict):
        for key in ("median", "subjectMedian", "subject_median", "meters", "m", "value"):
            value = _optional_float(depth.get(key))
            if value is not None:
                return value
    distance = raw.get("distance")
    if isinstance(distance, dict):
        for key in ("fromCameraM", "trackedTargetM", "nearestObjectM", "meters", "m", "value"):
            value = _optional_float(distance.get(key))
            if value is not None:
                return value
    return None


def _provenance(raw: dict[str, object]) -> dict[str, object]:
    provenance = dict(raw.get("provenance")) if isinstance(raw.get("provenance"), dict) else {}
    source = str(raw.get("source") or raw.get("provider") or "").strip()
    model_id = str(raw.get("model_id") or raw.get("modelId") or raw.get("model") or "").strip()
    if source:
        provenance.setdefault("source", source)
    if model_id:
        provenance.setdefault("model_id", model_id)
    return provenance


def _distance_band(*, depth_m: float | None, bbox: dict[str, float]) -> str:
    if depth_m is not None:
        return "near" if depth_m <= 1.0 else "far"
    area = max(0.0, bbox["x_max"] - bbox["x_min"]) * max(0.0, bbox["y_max"] - bbox["y_min"])
    return "near" if area >= 0.10 else "far"


def _hand_near_object(subject: dict[str, object], obj: dict[str, object]) -> bool:
    if subject.get("stable_id") == obj.get("stable_id") or not _hand_points(subject):
        return False
    bbox = obj.get("bbox")
    if not isinstance(bbox, dict):
        return False
    return any(_point_bbox_gap(point, bbox) <= 0.12 for point in _hand_points(subject))


def _hand_points(subject: dict[str, object]) -> list[tuple[float, float]]:
    pose = subject.get("pose")
    keypoints = pose.get("keypoints") if isinstance(pose, dict) else subject.get("keypoints")
    if not isinstance(keypoints, list):
        return []
    points: list[tuple[float, float]] = []
    for item in keypoints:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("label") or item.get("part") or "").strip().lower()
        if name in _HAND_KEYPOINTS:
            points.append((_coerce_float(item.get("x")), _coerce_float(item.get("y"))))
    return points


def _point_bbox_gap(point: tuple[float, float], bbox: dict[str, object]) -> float:
    x, y = point
    horizontal_gap = max(_coerce_float(bbox.get("x_min")) - x, x - _coerce_float(bbox.get("x_max")), 0.0)
    vertical_gap = max(_coerce_float(bbox.get("y_min")) - y, y - _coerce_float(bbox.get("y_max")), 0.0)
    return math.hypot(horizontal_gap, vertical_gap)


def _nearest_device(subject: dict[str, object], objects: list[dict[str, object]]) -> dict[str, object] | None:
    devices = [item for item in objects if item.get("stable_id") != subject.get("stable_id") and _is_device(item)]
    if not devices:
        return None
    subject_center = _center(subject["bbox"])
    return min(devices, key=lambda item: _distance(subject_center, _center(item["bbox"])))


def _is_device(item: dict[str, object]) -> bool:
    labels = {str(item.get("label") or "").lower()}
    labels.update(label.lower() for label in _normalize_semantic_labels(item.get("semantic_labels")))
    for clip_label in _normalize_label_annotations(item.get("clip_labels")):
        labels.add(str(clip_label.get("label") or "").lower())
    return bool(labels & _DEVICE_LABELS)


def _normalize_bbox(raw_bbox: object) -> dict[str, float] | None:
    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        try:
            x_min = _clip01(float(raw_bbox[0]))
            y_min = _clip01(float(raw_bbox[1]))
            x_max = _clip01(x_min + float(raw_bbox[2]))
            y_max = _clip01(y_min + float(raw_bbox[3]))
        except (TypeError, ValueError):
            return None
        return {
            "x_min": round(x_min, 4),
            "y_min": round(y_min, 4),
            "x_max": round(x_max, 4),
            "y_max": round(y_max, 4),
        }
    if not isinstance(raw_bbox, dict):
        return None
    try:
        if "x" in raw_bbox and "y" in raw_bbox and ("w" in raw_bbox or "width" in raw_bbox) and (
            "h" in raw_bbox or "height" in raw_bbox
        ):
            x_min = _clip01(float(raw_bbox.get("x", 0.0)))
            y_min = _clip01(float(raw_bbox.get("y", 0.0)))
            x_max = _clip01(x_min + float(raw_bbox.get("w", raw_bbox.get("width", 0.0))))
            y_max = _clip01(y_min + float(raw_bbox.get("h", raw_bbox.get("height", 0.0))))
            return {
                "x_min": round(x_min, 4),
                "y_min": round(y_min, 4),
                "x_max": round(x_max, 4),
                "y_max": round(y_max, 4),
            }
        if "x1" in raw_bbox and "y1" in raw_bbox and "x2" in raw_bbox and "y2" in raw_bbox:
            x_min = _clip01(float(raw_bbox.get("x1", 0.0)))
            y_min = _clip01(float(raw_bbox.get("y1", 0.0)))
            x_max = _clip01(float(raw_bbox.get("x2", 0.0)))
            y_max = _clip01(float(raw_bbox.get("y2", 0.0)))
            if x_max < x_min:
                x_min, x_max = x_max, x_min
            if y_max < y_min:
                y_min, y_max = y_max, y_min
            return {
                "x_min": round(x_min, 4),
                "y_min": round(y_min, 4),
                "x_max": round(x_max, 4),
                "y_max": round(y_max, 4),
            }
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


def _optional_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _digest(payload: object, *, length: int) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]
