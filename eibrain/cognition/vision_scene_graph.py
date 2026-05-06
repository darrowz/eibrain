"""Build a lightweight spatial scene graph from visual detections and tracks.

The module is intentionally model-free: it accepts detector/tracker-shaped
payloads, normalizes their bounding boxes, and derives spatial semantics with
deterministic heuristics.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping


_PERSON_LABELS = {"person", "human", "face", "body", "head"}
_EMPTY_REGIONS: dict[str, list[dict[str, Any]]] = {
    "left": [],
    "center": [],
    "right": [],
    "near": [],
    "far": [],
}
_FRAME_KEYS = {
    "detections",
    "objects",
    "tracks",
}


def build_vision_scene_graph(
    detections: list[Mapping[str, Any]] | None = None,
    *,
    tracks: list[Mapping[str, Any]] | None = None,
    frame_metadata: Mapping[str, Any] | None = None,
    near_area_threshold: float = 0.10,
    near_relation_gap: float = 0.12,
) -> dict[str, Any]:
    """Build people, objects, relations, regions, and summary from one frame."""

    frame = _frame_metadata(frame_metadata)
    objects = _build_objects(
        detections=detections,
        tracks=tracks,
        frame=frame,
        near_area_threshold=near_area_threshold,
    )
    people = [dict(item) for item in objects if item["kind"] == "person"]
    relations = _build_relations(objects, near_relation_gap=near_relation_gap)
    regions = _build_regions(objects)
    dominant_target = _dominant_target(objects)
    safety = {"pathClear": None, "nearObstacle": None}
    summary = _summary(objects, relations, dominant_target, safety)
    scene_id = _scene_id(frame, objects, relations)

    return {
        "sceneId": scene_id,
        "observedAt": frame["observedAt"],
        "frameId": frame["frameId"],
        "people": people,
        "objects": objects,
        "relations": relations,
        "relationships": relations,
        "regions": regions,
        "dominant_target": dominant_target,
        "summary": summary,
        "safety": safety,
        "metadata": _public_metadata(frame),
    }


def build_scene_graph(
    payload: Mapping[str, Any] | None = None,
    *,
    detections: list[Mapping[str, Any]] | None = None,
    tracks: list[Mapping[str, Any]] | None = None,
    frame_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Payload-style wrapper for callers that already hold a vision frame dict."""

    if payload is None:
        return build_vision_scene_graph(
            detections=detections,
            tracks=tracks,
            frame_metadata=frame_metadata,
        )

    metadata = {key: value for key, value in payload.items() if key not in _FRAME_KEYS}
    if frame_metadata is not None:
        metadata.update(dict(frame_metadata))
    return build_vision_scene_graph(
        detections=detections if detections is not None else _mapping_list(payload.get("detections", payload.get("objects"))),
        tracks=tracks if tracks is not None else _mapping_list(payload.get("tracks")),
        frame_metadata=metadata,
    )


def to_eiprotocol_scene_content(scene: Mapping[str, Any]) -> dict[str, Any]:
    """Map a scene graph to generic VisionSceneObservation content."""

    metadata = dict(scene.get("metadata") if isinstance(scene.get("metadata"), Mapping) else {})
    return {
        "sceneId": str(scene.get("sceneId", "")),
        "observedAt": str(scene.get("observedAt", "")),
        "summary": str(scene.get("summary", "")),
        "objects": [dict(item) for item in _mapping_list(scene.get("objects"))],
        "relationships": [dict(item) for item in _mapping_list(scene.get("relations", scene.get("relationships")))],
        "environment": {"source": "vision_scene_graph"},
        "imageUrl": str(metadata.get("imageUrl", "")),
        "metadata": metadata,
    }


def _build_objects(
    *,
    detections: list[Mapping[str, Any]] | None,
    tracks: list[Mapping[str, Any]] | None,
    frame: Mapping[str, Any],
    near_area_threshold: float,
) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for source, raw_items in (("track", tracks), ("detection", detections)):
        for index, raw in enumerate(raw_items or []):
            item = _normalize_object(
                raw,
                source=source,
                index=index,
                frame=frame,
                near_area_threshold=near_area_threshold,
                seen_ids=seen_ids,
            )
            if item is not None:
                objects.append(item)
    return sorted(objects, key=lambda item: (str(item["label"]), str(item["id"])))


def _normalize_object(
    raw: Mapping[str, Any],
    *,
    source: str,
    index: int,
    frame: Mapping[str, Any],
    near_area_threshold: float,
    seen_ids: set[str],
) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    label = _first_text(raw.get("label"), raw.get("name"), raw.get("class"))
    bbox = _normalize_bbox(raw.get("bbox"), width=frame.get("width"), height=frame.get("height"))
    if not label or bbox is None:
        return None

    confidence = round(_coerce_float(raw.get("confidence", raw.get("score", 0.0))), 3)
    center = _center(bbox)
    area = _area(bbox)
    horizontal = _horizontal_region(center[0])
    vertical = _vertical_region(center[1])
    kind = "person" if _is_person(label) else "object"
    object_id = _object_id(raw, label=label, center=center, source=source, index=index, seen_ids=seen_ids)
    item = {
        "id": object_id,
        "trackId": object_id,
        "label": label,
        "kind": kind,
        "confidence": confidence,
        "bbox": bbox,
        "center": {"x": round(center[0], 3), "y": round(center[1], 3)},
        "region": f"{horizontal}_{vertical}",
        "horizontal_region": horizontal,
        "vertical_region": vertical,
        "depth": "near" if area >= near_area_threshold else "far",
        "area": round(area, 4),
        "source": source,
    }
    if kind == "person":
        item["looking_at_device"] = None
    return item


def _build_relations(objects: list[dict[str, Any]], *, near_relation_gap: float) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    for subject in objects:
        for obj in objects:
            if subject["id"] == obj["id"]:
                continue
            subject_center = _object_center(subject)
            object_center = _object_center(obj)
            dx = object_center[0] - subject_center[0]
            dy = object_center[1] - subject_center[1]
            for relation in _relation_types(subject["bbox"], obj["bbox"], dx, dy, near_relation_gap=near_relation_gap):
                relations.append(
                    {
                        "subjectId": subject["id"],
                        "subjectLabel": subject["label"],
                        "relation": relation,
                        "objectId": obj["id"],
                        "objectLabel": obj["label"],
                    }
                )
    return sorted(
        relations,
        key=lambda item: (str(item["subjectId"]), str(item["relation"]), str(item["objectId"])),
    )


def _relation_types(
    subject_bbox: Mapping[str, float],
    object_bbox: Mapping[str, float],
    dx: float,
    dy: float,
    *,
    near_relation_gap: float,
) -> list[str]:
    relations: list[str] = []
    if dx > 0.12:
        relations.append("left_of")
    if dx < -0.12:
        relations.append("right_of")
    if dy > 0.18:
        relations.append("above")
    if dy < -0.18:
        relations.append("below")
    if _overlap_area(subject_bbox, object_bbox) > 0.0:
        relations.append("overlaps")
    if _bbox_gap_distance(subject_bbox, object_bbox) <= near_relation_gap:
        relations.append("near")
    return relations


def _build_regions(objects: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    regions = {key: [dict(item) for item in value] for key, value in _EMPTY_REGIONS.items()}
    for item in objects:
        summary = {
            "id": item["id"],
            "label": item["label"],
            "region": item["region"],
            "depth": item["depth"],
        }
        regions[str(item["horizontal_region"])].append(summary)
        regions[str(item["depth"])].append(summary)
    for key, values in regions.items():
        regions[key] = sorted(values, key=lambda item: (str(item["label"]), str(item["id"])))
    return regions


def _dominant_target(objects: list[dict[str, Any]]) -> dict[str, Any]:
    if not objects:
        return {}
    target = max(
        objects,
        key=lambda item: (
            float(item.get("area", 0.0)) * max(float(item.get("confidence", 0.0)), 0.0),
            1 if item.get("kind") == "person" else 0,
            str(item.get("id", "")),
        ),
    )
    salience = round(float(target["area"]) * max(float(target["confidence"]), 0.0), 4)
    return {
        "id": target["id"],
        "trackId": target["trackId"],
        "label": target["label"],
        "kind": target["kind"],
        "confidence": target["confidence"],
        "region": target["region"],
        "depth": target["depth"],
        "salience": salience,
    }


def _summary(
    objects: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    dominant_target: Mapping[str, Any],
    safety: Mapping[str, Any],
) -> str:
    path_clear = "unknown" if safety.get("pathClear") is None else str(safety.get("pathClear")).lower()
    near_obstacle = "unknown" if safety.get("nearObstacle") is None else str(safety.get("nearObstacle")).lower()
    if not objects:
        return f"Observed empty visual scene; pathClear {path_clear}; nearObstacle {near_obstacle}"

    counts: dict[str, int] = {}
    for item in objects:
        label = str(item["label"])
        counts[label] = counts.get(label, 0) + 1
    observed = ", ".join(
        f"{count} {label}" if count > 1 else label
        for label, count in sorted(counts.items())
    )
    relation_names = sorted({str(item["relation"]) for item in relations})
    relation_summary = ", ".join(relation_names) if relation_names else "none"
    target_label = str(dominant_target.get("label", "none"))
    target_region = str(dominant_target.get("region", ""))
    return (
        f"Observed {observed}; dominant target {target_label} in {target_region}; "
        f"relations: {relation_summary}; pathClear {path_clear}; nearObstacle {near_obstacle}"
    )


def _frame_metadata(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = raw if isinstance(raw, Mapping) else {}
    frame_id = _first_text(raw.get("frameId"), raw.get("frame_id"), raw.get("frame"), raw.get("id"))
    observed_at = _first_text(raw.get("observedAt"), raw.get("observed_at"), raw.get("timestamp"), raw.get("time"))
    metadata: dict[str, Any] = {
        "frameId": frame_id,
        "observedAt": observed_at,
        "source": "vision_scene_graph",
    }
    width = _optional_int(raw.get("width"))
    height = _optional_int(raw.get("height"))
    if width is not None:
        metadata["width"] = width
    if height is not None:
        metadata["height"] = height
    image_url = _first_text(raw.get("imageUrl"), raw.get("image_url"))
    if image_url:
        metadata["imageUrl"] = image_url
    for key, value in raw.items():
        if key in {"frameId", "frame_id", "frame", "id", "observedAt", "observed_at", "timestamp", "time"}:
            continue
        if key in {"width", "height", "imageUrl", "image_url"}:
            continue
        metadata[str(key)] = value
    return metadata


def _public_metadata(frame: Mapping[str, Any]) -> dict[str, Any]:
    return dict(frame)


def _normalize_bbox(raw: Any, *, width: Any = None, height: Any = None) -> dict[str, float] | None:
    values = _bbox_values(raw)
    if values is None:
        return None
    x_min, y_min, x_max, y_max = values
    frame_width = _optional_int(width)
    frame_height = _optional_int(height)
    if frame_width and max(abs(x_min), abs(x_max)) > 1.0:
        x_min /= frame_width
        x_max /= frame_width
    if frame_height and max(abs(y_min), abs(y_max)) > 1.0:
        y_min /= frame_height
        y_max /= frame_height
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if y_max < y_min:
        y_min, y_max = y_max, y_min
    return {
        "x_min": round(_clip01(x_min), 4),
        "y_min": round(_clip01(y_min), 4),
        "x_max": round(_clip01(x_max), 4),
        "y_max": round(_clip01(y_max), 4),
    }


def _bbox_values(raw: Any) -> tuple[float, float, float, float] | None:
    try:
        if isinstance(raw, Mapping):
            return (
                float(raw.get("x_min", raw.get("xmin", raw.get("left", 0.0)))),
                float(raw.get("y_min", raw.get("ymin", raw.get("top", 0.0)))),
                float(raw.get("x_max", raw.get("xmax", raw.get("right", 0.0)))),
                float(raw.get("y_max", raw.get("ymax", raw.get("bottom", 0.0)))),
            )
        if isinstance(raw, (list, tuple)) and len(raw) == 4:
            return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
    except (TypeError, ValueError):
        return None
    return None


def _object_id(
    raw: Mapping[str, Any],
    *,
    label: str,
    center: tuple[float, float],
    source: str,
    index: int,
    seen_ids: set[str],
) -> str:
    object_id = _first_text(
        raw.get("track_id"),
        raw.get("trackId"),
        raw.get("object_id"),
        raw.get("objectId"),
        raw.get("stable_id"),
        raw.get("stableId"),
        raw.get("id"),
    )
    if not object_id:
        object_id = f"{_slug(label)}:{int(center[0] * 10):02d}:{int(center[1] * 10):02d}"
    candidate = object_id
    suffix = 2
    while candidate in seen_ids:
        candidate = f"{object_id}:{source}:{index:02d}:{suffix}"
        suffix += 1
    seen_ids.add(candidate)
    return candidate


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _optional_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _is_person(label: str) -> bool:
    normalized = label.strip().lower()
    return normalized in _PERSON_LABELS or normalized.startswith("person")


def _slug(label: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in label).strip("_") or "object"


def _horizontal_region(x: float) -> str:
    if x < 0.33:
        return "left"
    if x < 0.66:
        return "center"
    return "right"


def _vertical_region(y: float) -> str:
    if y < 0.25:
        return "top"
    if y < 0.75:
        return "middle"
    return "bottom"


def _center(bbox: Mapping[str, float]) -> tuple[float, float]:
    return (
        (float(bbox["x_min"]) + float(bbox["x_max"])) / 2.0,
        (float(bbox["y_min"]) + float(bbox["y_max"])) / 2.0,
    )


def _object_center(obj: Mapping[str, Any]) -> tuple[float, float]:
    center = obj.get("center")
    if isinstance(center, Mapping):
        return (_coerce_float(center.get("x")), _coerce_float(center.get("y")))
    bbox = obj.get("bbox")
    return _center(bbox) if isinstance(bbox, Mapping) else (0.0, 0.0)


def _area(bbox: Mapping[str, float]) -> float:
    return max(0.0, float(bbox["x_max"]) - float(bbox["x_min"])) * max(0.0, float(bbox["y_max"]) - float(bbox["y_min"]))


def _overlap_area(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    width = min(float(a["x_max"]), float(b["x_max"])) - max(float(a["x_min"]), float(b["x_min"]))
    height = min(float(a["y_max"]), float(b["y_max"])) - max(float(a["y_min"]), float(b["y_min"]))
    return max(0.0, width) * max(0.0, height)


def _bbox_gap_distance(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    horizontal_gap = max(float(a["x_min"]) - float(b["x_max"]), float(b["x_min"]) - float(a["x_max"]), 0.0)
    vertical_gap = max(float(a["y_min"]) - float(b["y_max"]), float(b["y_min"]) - float(a["y_max"]), 0.0)
    return math.hypot(horizontal_gap, vertical_gap)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _scene_id(frame: Mapping[str, Any], objects: list[dict[str, Any]], relations: list[dict[str, Any]]) -> str:
    payload = {
        "frameId": frame.get("frameId", ""),
        "observedAt": frame.get("observedAt", ""),
        "objects": [
            {
                "id": item["id"],
                "label": item["label"],
                "region": item["region"],
                "depth": item["depth"],
            }
            for item in objects
        ],
        "relations": relations,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")).hexdigest()[:12]
    return f"scene_vg_{digest}"


__all__ = [
    "build_scene_graph",
    "build_vision_scene_graph",
    "to_eiprotocol_scene_content",
]
