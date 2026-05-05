"""Software-only realtime vision event simulator.

This module intentionally has no honjia device dependency. It turns sequential
frame detections into stable tracks, lifecycle events, attention, and
eiprotocol-friendly observation content.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping


@dataclass(slots=True)
class _Track:
    track_id: str
    label: str
    bbox: dict[str, float]
    confidence: float
    first_seen_frame: str
    last_seen_frame: str
    last_observed_at: str
    missing_frames: int = 0


class RealtimeVisionSimulator:
    """Track detections across frames and emit realtime scene events."""

    def __init__(
        self,
        *,
        match_distance: float = 0.35,
        move_threshold: float = 0.12,
        max_missing_frames: int = 1,
    ) -> None:
        self.match_distance = float(match_distance)
        self.move_threshold = float(move_threshold)
        self.max_missing_frames = int(max_missing_frames)
        self._tracks: dict[str, _Track] = {}
        self._next_ids: dict[str, int] = {}

    def update(
        self,
        *,
        frame_id: str,
        observed_at: str,
        detections: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        normalized = [_normalize_detection(item) for item in detections]
        normalized = [item for item in normalized if item is not None]
        events: list[dict[str, Any]] = []
        active_before = {track_id: track for track_id, track in self._tracks.items() if track.missing_frames == 0}
        matches = self._match_detections(normalized)
        matched_detection_indexes = {detection_index for detection_index, _ in matches}
        matched_track_ids = {track_id for _, track_id in matches}

        for detection_index, track_id in matches:
            detection = normalized[detection_index]
            track = self._tracks[track_id]
            previous_bbox = dict(track.bbox)
            previous_region = _region(_center(previous_bbox))
            distance = _distance(_center(previous_bbox), _center(detection["bbox"]))
            track.bbox = dict(detection["bbox"])
            track.confidence = float(detection["confidence"])
            track.last_seen_frame = frame_id
            track.last_observed_at = observed_at
            track.missing_frames = 0
            current_region = _region(_center(track.bbox))
            if distance >= self.move_threshold or previous_region != current_region:
                events.append(
                    _event(
                        scene_id="",
                        event_type="moved",
                        observed_at=observed_at,
                        frame_id=frame_id,
                        track=track,
                        confidence=track.confidence,
                        from_region=previous_region,
                        to_region=current_region,
                        distance=distance,
                    )
                )

        for detection_index, detection in enumerate(normalized):
            if detection_index in matched_detection_indexes:
                continue
            track = self._new_track(detection, frame_id=frame_id, observed_at=observed_at)
            self._tracks[track.track_id] = track
            matched_track_ids.add(track.track_id)
            events.append(
                _event(
                    scene_id="",
                    event_type="appeared",
                    observed_at=observed_at,
                    frame_id=frame_id,
                    track=track,
                    confidence=track.confidence,
                    from_region="",
                    to_region=_region(_center(track.bbox)),
                    distance=0.0,
                )
            )

        for track_id, track in list(self._tracks.items()):
            if track_id in matched_track_ids:
                continue
            track.missing_frames += 1
            if track.missing_frames > self.max_missing_frames:
                events.append(
                    _event(
                        scene_id="",
                        event_type="disappeared",
                        observed_at=observed_at,
                        frame_id=frame_id,
                        track=track,
                        confidence=track.confidence,
                        from_region=_region(_center(track.bbox)),
                        to_region="",
                        distance=0.0,
                    )
                )
                del self._tracks[track_id]

        active_tracks = sorted(
            [track for track in self._tracks.values() if track.missing_frames == 0],
            key=lambda item: item.track_id,
        )
        attention = _attention_track(active_tracks)
        if attention is not None:
            events.append(
                _event(
                    scene_id="",
                    event_type="attention",
                    observed_at=observed_at,
                    frame_id=frame_id,
                    track=attention,
                    confidence=attention.confidence,
                    from_region="",
                    to_region=_region(_center(attention.bbox)),
                    distance=0.0,
                )
            )

        objects = [_track_object(track) for track in active_tracks]
        relationships = _relationships(objects)
        summary = _summary(objects, events)
        scene_id = _scene_id(frame_id, observed_at, objects, relationships)
        scene_snapshot = {
            "sceneId": scene_id,
            "observedAt": observed_at,
            "frameId": frame_id,
            "objects": objects,
            "relationships": relationships,
            "attention": _attention_object(attention),
            "summary": summary,
            "metadata": {
                "frameId": frame_id,
                "realtime": True,
                "simulator": "software",
                "trackCount": len(objects),
            },
        }
        for event in events:
            event["sceneId"] = scene_id
            event["eventId"] = f"{scene_id}:{event['eventType']}:{event['subject']['trackId']}"
        return {
            "frameId": frame_id,
            "observedAt": observed_at,
            "events": events,
            "sceneSnapshot": scene_snapshot,
            "sceneGraphSummary": summary,
        }

    def _match_detections(self, detections: list[dict[str, Any]]) -> list[tuple[int, str]]:
        candidates: list[tuple[float, int, str]] = []
        active_tracks = [track for track in self._tracks.values() if track.missing_frames <= self.max_missing_frames]
        for detection_index, detection in enumerate(detections):
            for track in active_tracks:
                if detection["label"] != track.label:
                    continue
                distance = _distance(_center(detection["bbox"]), _center(track.bbox))
                if distance <= self.match_distance:
                    candidates.append((distance, detection_index, track.track_id))

        matches: list[tuple[int, str]] = []
        used_detections: set[int] = set()
        used_tracks: set[str] = set()
        for _, detection_index, track_id in sorted(candidates):
            if detection_index in used_detections or track_id in used_tracks:
                continue
            matches.append((detection_index, track_id))
            used_detections.add(detection_index)
            used_tracks.add(track_id)
        return matches

    def _new_track(self, detection: dict[str, Any], *, frame_id: str, observed_at: str) -> _Track:
        label = str(detection["label"])
        next_id = self._next_ids.get(label, 0) + 1
        self._next_ids[label] = next_id
        return _Track(
            track_id=f"{label}-{next_id:03d}",
            label=label,
            bbox=dict(detection["bbox"]),
            confidence=float(detection["confidence"]),
            first_seen_frame=frame_id,
            last_seen_frame=frame_id,
            last_observed_at=observed_at,
        )


def to_eiprotocol_scene_content(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Map a simulator result to VisionSceneObservation content."""

    scene = _scene_snapshot(snapshot)
    return {
        "sceneId": str(scene.get("sceneId", "")),
        "observedAt": str(scene.get("observedAt", "")),
        "summary": str(snapshot.get("sceneGraphSummary") or scene.get("summary") or ""),
        "objects": [dict(item) for item in _dict_list(scene.get("objects"))],
        "relationships": [dict(item) for item in _dict_list(scene.get("relationships"))],
        "environment": {"source": "realtime_vision_simulator"},
        "imageUrl": "",
        "metadata": dict(scene.get("metadata") if isinstance(scene.get("metadata"), Mapping) else {}),
    }


def to_eiprotocol_event_contents(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Map simulator events to VisionEventObservation content dictionaries."""

    contents: list[dict[str, Any]] = []
    for event in _dict_list(snapshot.get("events")):
        contents.append(
            {
                "eventId": str(event.get("eventId", "")),
                "eventType": str(event.get("eventType", "")),
                "observedAt": str(event.get("observedAt", "")),
                "sceneId": str(event.get("sceneId", "")),
                "subject": dict(event.get("subject") if isinstance(event.get("subject"), Mapping) else {}),
                "confidence": event.get("confidence"),
                "details": dict(event.get("details") if isinstance(event.get("details"), Mapping) else {}),
                "metadata": dict(event.get("metadata") if isinstance(event.get("metadata"), Mapping) else {}),
            }
        )
    return contents


def _normalize_detection(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    label = str(raw.get("label") or raw.get("name") or raw.get("class") or "").strip()
    bbox = _normalize_bbox(raw.get("bbox"))
    if not label or bbox is None:
        return None
    return {
        "label": label,
        "bbox": bbox,
        "confidence": _coerce_float(raw.get("confidence", raw.get("score", 0.0))),
    }


def _normalize_bbox(raw: Any) -> dict[str, float] | None:
    if not isinstance(raw, Mapping):
        return None
    try:
        x_min = _clip01(float(raw.get("x_min", raw.get("xmin", raw.get("left", 0.0)))))
        y_min = _clip01(float(raw.get("y_min", raw.get("ymin", raw.get("top", 0.0)))))
        x_max = _clip01(float(raw.get("x_max", raw.get("xmax", raw.get("right", 0.0)))))
        y_max = _clip01(float(raw.get("y_max", raw.get("ymax", raw.get("bottom", 0.0)))))
    except (TypeError, ValueError):
        return None
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if y_max < y_min:
        y_min, y_max = y_max, y_min
    return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}


def _event(
    *,
    scene_id: str,
    event_type: str,
    observed_at: str,
    frame_id: str,
    track: _Track,
    confidence: float,
    from_region: str,
    to_region: str,
    distance: float,
) -> dict[str, Any]:
    return {
        "eventId": f"{scene_id}:{event_type}:{track.track_id}" if scene_id else "",
        "eventType": event_type,
        "observedAt": observed_at,
        "sceneId": scene_id,
        "subject": {"trackId": track.track_id, "label": track.label},
        "confidence": round(float(confidence), 3),
        "details": {
            "fromRegion": from_region,
            "toRegion": to_region,
            "distance": round(float(distance), 3),
        },
        "metadata": {"frameId": frame_id},
    }


def _track_object(track: _Track) -> dict[str, Any]:
    center = _center(track.bbox)
    return {
        "trackId": track.track_id,
        "label": track.label,
        "confidence": round(float(track.confidence), 3),
        "bbox": dict(track.bbox),
        "center": {"x": round(center[0], 3), "y": round(center[1], 3)},
        "region": _region(center),
        "missingFrames": track.missing_frames,
    }


def _relationships(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    for subject in objects:
        for obj in objects:
            if subject["trackId"] == obj["trackId"]:
                continue
            subject_center = _object_center(subject)
            object_center = _object_center(obj)
            dx = object_center[0] - subject_center[0]
            dy = object_center[1] - subject_center[1]
            for relation in _relation_types(dx, dy):
                relations.append(
                    {
                        "subjectId": subject["trackId"],
                        "subjectLabel": subject["label"],
                        "relation": relation,
                        "objectId": obj["trackId"],
                        "objectLabel": obj["label"],
                    }
                )
    return sorted(relations, key=lambda item: (item["subjectId"], item["relation"], item["objectId"]))


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


def _summary(objects: list[dict[str, Any]], events: list[dict[str, Any]]) -> str:
    if objects:
        labels: dict[str, int] = {}
        for obj in objects:
            label = str(obj.get("label", ""))
            labels[label] = labels.get(label, 0) + 1
        observed = ", ".join(
            f"{count} {label}" if count > 1 else label for label, count in sorted(labels.items())
        )
    else:
        observed = "empty scene"
    event_types = sorted({str(event.get("eventType")) for event in events if event.get("eventType")})
    if event_types:
        return f"Observed {observed}; realtime events: {', '.join(event_types)}"
    return f"Observed {observed}; realtime events: none"


def _attention_track(tracks: list[_Track]) -> _Track | None:
    if not tracks:
        return None
    return max(tracks, key=lambda track: (_area(track.bbox) * max(track.confidence, 0.0), track.confidence, track.track_id))


def _attention_object(track: _Track | None) -> dict[str, Any]:
    if track is None:
        return {}
    return {
        "trackId": track.track_id,
        "label": track.label,
        "confidence": round(float(track.confidence), 3),
        "region": _region(_center(track.bbox)),
    }


def _scene_snapshot(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    scene = snapshot.get("sceneSnapshot")
    return scene if isinstance(scene, Mapping) else snapshot


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _scene_id(frame_id: str, observed_at: str, objects: list[dict[str, Any]], relationships: list[dict[str, Any]]) -> str:
    payload = {
        "frameId": frame_id,
        "observedAt": observed_at,
        "objects": [
            {"trackId": item["trackId"], "label": item["label"], "region": item["region"]}
            for item in objects
        ],
        "relationships": relationships,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    return f"scene_rt_{digest}"


def _center(bbox: Mapping[str, float]) -> tuple[float, float]:
    return ((float(bbox["x_min"]) + float(bbox["x_max"])) / 2.0, (float(bbox["y_min"]) + float(bbox["y_max"])) / 2.0)


def _object_center(obj: Mapping[str, Any]) -> tuple[float, float]:
    center = obj.get("center")
    if isinstance(center, Mapping):
        return (_coerce_float(center.get("x")), _coerce_float(center.get("y")))
    bbox = obj.get("bbox")
    if isinstance(bbox, Mapping):
        return _center(bbox)  # type: ignore[arg-type]
    return (0.0, 0.0)


def _region(center: tuple[float, float]) -> str:
    x_name = "left" if center[0] < 0.33 else "center" if center[0] < 0.66 else "right"
    y_name = "top" if center[1] < 0.25 else "middle" if center[1] < 0.75 else "bottom"
    return f"{x_name}_{y_name}"


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _area(bbox: Mapping[str, float]) -> float:
    return max(0.0, float(bbox["x_max"]) - float(bbox["x_min"])) * max(0.0, float(bbox["y_max"]) - float(bbox["y_min"]))


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


__all__ = [
    "RealtimeVisionSimulator",
    "to_eiprotocol_event_contents",
    "to_eiprotocol_scene_content",
]
