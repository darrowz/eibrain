"""Realtime eye scene/event aggregation bridge."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import hashlib
import json
import math
from typing import Any

try:  # pragma: no cover - exercised in eihead standalone export tests.
    from eibrain.cognition.vision_realtime import (
        RealtimeVisionSimulator,
        to_eiprotocol_event_contents,
        to_eiprotocol_scene_content,
    )
except ModuleNotFoundError:  # pragma: no cover - standalone eihead has no full eibrain cognition package.
    RealtimeVisionSimulator = None  # type: ignore[assignment]
    to_eiprotocol_event_contents = None  # type: ignore[assignment]
    to_eiprotocol_scene_content = None  # type: ignore[assignment]

from .realtime import COMPAT_STATIC_FRAME_MODE, REALTIME_STREAM_MODE


class RealtimeVisionSceneBridge:
    """Convert realtime eye observations into scene snapshots and events."""

    def __init__(
        self,
        *,
        simulator: RealtimeVisionSimulator | None = None,
        match_distance: float = 0.35,
        move_threshold: float = 0.12,
        max_missing_frames: int = 1,
    ) -> None:
        self.simulator = simulator or _new_simulator(
            match_distance=match_distance,
            move_threshold=move_threshold,
            max_missing_frames=max_missing_frames,
        )
        self.latest_scene_id = ""

    def update(self, observation_or_status: Mapping[str, Any]) -> dict[str, Any]:
        """Aggregate one service observation/status dict into protocol content."""

        observation = dict(observation_or_status)
        frame_id = _frame_id(observation)
        observed_at = _observed_at(observation)
        live, reason = _live_state(observation)
        if not live:
            return self._non_live_result(frame_id=frame_id, observed_at=observed_at, reason=reason)

        snapshot = self.simulator.update(
            frame_id=frame_id,
            observed_at=observed_at,
            detections=_detections(observation),
        )
        scene_snapshot = _scene_content(snapshot)
        event_contents = _event_contents(snapshot)
        self.latest_scene_id = str(scene_snapshot.get("sceneId", ""))
        object_count = len(scene_snapshot.get("objects", []))
        stable_target = _stable_target_from_scene(scene_snapshot)
        last_event = dict(event_contents[-1]) if event_contents else None
        diagnostics = _diagnostics(
            observation,
            track_count=object_count,
            stable_target=stable_target,
            event_count=len(event_contents),
            last_event=last_event,
        )

        return {
            "kind": "realtime_vision_scene_bridge",
            "mode": REALTIME_STREAM_MODE,
            "status": "ok",
            "stream_ready": True,
            "not_wired": False,
            "stale": False,
            "live": True,
            "reason": "live",
            "frame_id": frame_id,
            "observed_at": observed_at,
            "latest_scene_id": self.latest_scene_id,
            "scene_id": self.latest_scene_id,
            "scene": scene_snapshot,
            "scene_snapshot": scene_snapshot,
            "scene_summary": str(snapshot.get("sceneGraphSummary", "")),
            "sceneGraphSummary": str(snapshot.get("sceneGraphSummary", "")),
            "event_summary": str(snapshot.get("eventSummary") or scene_snapshot.get("eventSummary") or ""),
            "event_contents": event_contents,
            "events": event_contents,
            "tracks": [dict(item) for item in scene_snapshot.get("objects", []) if isinstance(item, Mapping)],
            "target": _target_from_scene(scene_snapshot),
            "stable_target": stable_target,
            "object_count": object_count,
            "track_count": object_count,
            "event_count": len(event_contents),
            "last_event": last_event,
            "diagnostics": diagnostics,
        }

    def _non_live_result(self, *, frame_id: str, observed_at: str, reason: str) -> dict[str, Any]:
        scene_snapshot = {
            "sceneId": "",
            "observedAt": observed_at,
            "summary": f"Realtime vision observation is non-live: {reason}",
            "objects": [],
            "relationships": [],
            "environment": {"source": "eihead.eye.scene", "live": False},
            "imageUrl": "",
            "metadata": {
                "frameId": frame_id,
                "realtime": False,
                "reason": reason,
                "trackCount": 0,
            },
        }
        return {
            "kind": "realtime_vision_scene_bridge",
            "mode": REALTIME_STREAM_MODE,
            "status": reason,
            "stream_ready": False,
            "not_wired": reason in {"not_wired", "placeholder"},
            "stale": reason == "stale",
            "live": False,
            "reason": reason,
            "frame_id": frame_id,
            "observed_at": observed_at,
            "latest_scene_id": self.latest_scene_id,
            "scene_id": "",
            "scene": scene_snapshot,
            "scene_snapshot": scene_snapshot,
            "scene_summary": scene_snapshot["summary"],
            "sceneGraphSummary": scene_snapshot["summary"],
            "event_summary": "",
            "event_contents": [],
            "events": [],
            "tracks": [],
            "target": None,
            "stable_target": None,
            "object_count": 0,
            "track_count": 0,
            "event_count": 0,
            "last_event": None,
            "diagnostics": {
                "fps": 0.0,
                "frame_age": None,
                "frame_age_s": None,
                "track_count": 0,
                "stable_target": None,
                "event_count": 0,
                "last_event": None,
            },
        }


def _live_state(observation: Mapping[str, Any]) -> tuple[bool, str]:
    status = str(observation.get("status", "")).strip().lower()
    mode = str(observation.get("mode", "") or REALTIME_STREAM_MODE).strip().lower()
    backend = str(observation.get("backend", "")).strip().lower()

    if _truthy(observation.get("not_wired")) or status == "not_wired":
        return False, "not_wired"
    if _truthy(observation.get("placeholder")) or backend == "placeholder":
        return False, "placeholder"
    if (
        _truthy(observation.get("compatibility_mode"))
        or mode == COMPAT_STATIC_FRAME_MODE
        or status in {"compat_static", "compat_static_frame", "compat_static_frame_test_only"}
    ):
        return False, "compat_static"
    if status == "static":
        return False, "static"
    if _truthy(observation.get("stale")) or status == "stale":
        return False, "stale"
    return True, "live"


def _frame_id(observation: Mapping[str, Any]) -> str:
    return str(observation.get("frame_id") or observation.get("frameId") or observation.get("last_frame_id") or "")


def _observed_at(observation: Mapping[str, Any]) -> str:
    for key in ("observed_at", "observedAt", "captured_at", "capturedAt"):
        value = observation.get(key)
        if value:
            return str(value)
    for key in ("captured_at_ts", "last_frame_captured_at_ts", "timestamp", "ts"):
        value = observation.get(key)
        if value is None:
            continue
        try:
            return datetime.fromtimestamp(float(value), tz=UTC).isoformat(timespec="milliseconds")
        except (OSError, OverflowError, TypeError, ValueError):
            continue
    return ""


def _detections(observation: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = observation.get("detections")
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _target_from_scene(scene_snapshot: Mapping[str, Any]) -> dict[str, Any] | None:
    stable_target = _stable_target_from_scene(scene_snapshot)
    if stable_target:
        return stable_target
    attention = scene_snapshot.get("attention")
    if isinstance(attention, Mapping) and attention:
        track_id = str(attention.get("trackId") or "")
        for item in scene_snapshot.get("objects", []):
            if isinstance(item, Mapping) and str(item.get("trackId") or "") == track_id:
                return _target_from_object(item)
    for item in scene_snapshot.get("objects", []):
        if isinstance(item, Mapping):
            return _target_from_object(item)
    return None


def _stable_target_from_scene(scene_snapshot: Mapping[str, Any]) -> dict[str, Any] | None:
    for key in ("stableTarget", "stable_target", "attention"):
        value = scene_snapshot.get(key)
        if not isinstance(value, Mapping) or not value:
            continue
        track_id = str(value.get("trackId") or value.get("track_id") or "")
        if track_id:
            for item in scene_snapshot.get("objects", []):
                if isinstance(item, Mapping) and str(item.get("trackId") or item.get("track_id") or "") == track_id:
                    return _target_from_object(item)
        return _target_from_object(value)
    return None


def _target_from_object(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "track_id": item.get("trackId"),
        "trackId": item.get("trackId"),
        "label": item.get("label"),
        "score": item.get("confidence"),
        "center": dict(item.get("center")) if isinstance(item.get("center"), Mapping) else None,
        "bbox": dict(item.get("bbox")) if isinstance(item.get("bbox"), Mapping) else None,
        "temporalState": item.get("temporalState"),
    }


def _diagnostics(
    observation: Mapping[str, Any],
    *,
    track_count: int,
    stable_target: Mapping[str, Any] | None,
    event_count: int,
    last_event: Mapping[str, Any] | None,
) -> dict[str, Any]:
    frame_age = _number_or_none(
        observation.get("last_frame_age", observation.get("last_frame_age_s", observation.get("frame_age")))
    )
    return {
        "fps": _number_or_zero(observation.get("fps")),
        "frame_age": frame_age,
        "frame_age_s": frame_age,
        "track_count": int(track_count),
        "stable_target": dict(stable_target) if isinstance(stable_target, Mapping) else None,
        "event_count": int(event_count),
        "last_event": dict(last_event) if isinstance(last_event, Mapping) else None,
    }


def _number_or_zero(value: Any) -> float:
    number = _number_or_none(value)
    return 0.0 if number is None else number


def _number_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _new_simulator(*, match_distance: float, move_threshold: float, max_missing_frames: int) -> Any:
    if RealtimeVisionSimulator is not None:
        return RealtimeVisionSimulator(
            match_distance=match_distance,
            move_threshold=move_threshold,
            max_missing_frames=max_missing_frames,
        )
    return _FallbackRealtimeVisionSimulator(
        match_distance=match_distance,
        move_threshold=move_threshold,
        max_missing_frames=max_missing_frames,
    )


def _scene_content(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    if to_eiprotocol_scene_content is not None:
        return to_eiprotocol_scene_content(snapshot)
    scene = snapshot.get("sceneSnapshot")
    scene = scene if isinstance(scene, Mapping) else snapshot
    return {
        "sceneId": str(scene.get("sceneId", "")),
        "observedAt": str(scene.get("observedAt", "")),
        "summary": str(snapshot.get("sceneGraphSummary") or scene.get("summary") or ""),
        "objects": [dict(item) for item in _dict_list(scene.get("objects"))],
        "relationships": [dict(item) for item in _dict_list(scene.get("relationships"))],
        "environment": {"source": "eihead.eye.scene"},
        "imageUrl": "",
        "metadata": dict(scene.get("metadata")) if isinstance(scene.get("metadata"), Mapping) else {},
        "attention": dict(scene.get("attention")) if isinstance(scene.get("attention"), Mapping) else {},
        "stableTarget": dict(scene.get("stableTarget")) if isinstance(scene.get("stableTarget"), Mapping) else {},
        "eventSummary": str(snapshot.get("eventSummary") or scene.get("eventSummary") or ""),
    }


def _event_contents(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    if to_eiprotocol_event_contents is not None:
        return to_eiprotocol_event_contents(snapshot)
    contents: list[dict[str, Any]] = []
    for event in _dict_list(snapshot.get("events")):
        contents.append(
            {
                "eventId": str(event.get("eventId", "")),
                "eventType": str(event.get("eventType", "")),
                "observedAt": str(event.get("observedAt", "")),
                "sceneId": str(event.get("sceneId", "")),
                "subject": dict(event.get("subject")) if isinstance(event.get("subject"), Mapping) else {},
                "confidence": event.get("confidence"),
                "details": dict(event.get("details")) if isinstance(event.get("details"), Mapping) else {},
                "metadata": dict(event.get("metadata")) if isinstance(event.get("metadata"), Mapping) else {},
            }
        )
    return contents


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


class _FallbackRealtimeVisionSimulator:
    """Small eihead-local tracker used when the full eibrain package is absent."""

    def __init__(self, *, match_distance: float, move_threshold: float, max_missing_frames: int) -> None:
        self.match_distance = float(match_distance)
        self.move_threshold = float(move_threshold)
        self.max_missing_frames = int(max_missing_frames)
        self._tracks: dict[str, dict[str, Any]] = {}
        self._next_ids: dict[str, int] = {}

    def update(self, *, frame_id: str, observed_at: str, detections: list[Mapping[str, Any]]) -> dict[str, Any]:
        normalized = [item for item in (_normalize_detection(item) for item in detections) if item is not None]
        events: list[dict[str, Any]] = []
        matches = self._match(normalized)
        matched_detection_indexes = {detection_index for detection_index, _ in matches}
        matched_track_ids = {track_id for _, track_id in matches}

        for detection_index, track_id in matches:
            detection = normalized[detection_index]
            track = self._tracks[track_id]
            previous_bbox = dict(track["bbox"])
            previous_region = _region(_center(previous_bbox))
            distance = _distance(_center(previous_bbox), _center(detection["bbox"]))
            track.update(
                {
                    "bbox": dict(detection["bbox"]),
                    "confidence": detection["confidence"],
                    "last_seen_frame": frame_id,
                    "last_observed_at": observed_at,
                    "missing_frames": 0,
                }
            )
            current_region = _region(_center(track["bbox"]))
            if distance >= self.move_threshold or previous_region != current_region:
                events.append(
                    _fallback_event(
                        event_type="moved",
                        observed_at=observed_at,
                        frame_id=frame_id,
                        track=track,
                        from_region=previous_region,
                        to_region=current_region,
                        distance=distance,
                    )
                )

        for detection_index, detection in enumerate(normalized):
            if detection_index in matched_detection_indexes:
                continue
            track = self._new_track(detection, frame_id=frame_id, observed_at=observed_at)
            self._tracks[str(track["trackId"])] = track
            matched_track_ids.add(str(track["trackId"]))
            events.append(
                _fallback_event(
                    event_type="appeared",
                    observed_at=observed_at,
                    frame_id=frame_id,
                    track=track,
                    from_region="",
                    to_region=_region(_center(track["bbox"])),
                    distance=0.0,
                )
            )

        for track_id, track in list(self._tracks.items()):
            if track_id in matched_track_ids:
                continue
            track["missing_frames"] = int(track.get("missing_frames", 0)) + 1
            if int(track["missing_frames"]) > self.max_missing_frames:
                events.append(
                    _fallback_event(
                        event_type="disappeared",
                        observed_at=observed_at,
                        frame_id=frame_id,
                        track=track,
                        from_region=_region(_center(track["bbox"])),
                        to_region="",
                        distance=0.0,
                    )
                )
                del self._tracks[track_id]

        active_tracks = [track for track in self._tracks.values() if int(track.get("missing_frames", 0)) == 0]
        attention = max(active_tracks, key=lambda item: (float(item["confidence"]), str(item["trackId"])), default=None)
        if attention is not None:
            events.append(
                _fallback_event(
                    event_type="attention",
                    observed_at=observed_at,
                    frame_id=frame_id,
                    track=attention,
                    from_region="",
                    to_region=_region(_center(attention["bbox"])),
                    distance=0.0,
                )
            )

        objects = [_track_object(track) for track in sorted(active_tracks, key=lambda item: str(item["trackId"]))]
        summary = _fallback_summary(objects, events)
        scene_id = _fallback_scene_id(frame_id, observed_at, objects)
        for event in events:
            event["sceneId"] = scene_id
            event["eventId"] = f"{scene_id}:{event['eventType']}:{event['subject']['trackId']}"
        return {
            "frameId": frame_id,
            "observedAt": observed_at,
            "events": events,
            "sceneSnapshot": {
                "sceneId": scene_id,
                "observedAt": observed_at,
                "frameId": frame_id,
                "objects": objects,
                "relationships": [],
                "attention": _track_object(attention) if attention else {},
                "summary": summary,
                "metadata": {"frameId": frame_id, "realtime": True, "simulator": "eihead_local", "trackCount": len(objects)},
            },
            "sceneGraphSummary": summary,
        }

    def _match(self, detections: list[dict[str, Any]]) -> list[tuple[int, str]]:
        candidates: list[tuple[float, int, str]] = []
        for detection_index, detection in enumerate(detections):
            for track_id, track in self._tracks.items():
                if detection["label"] != track["label"]:
                    continue
                distance = _distance(_center(detection["bbox"]), _center(track["bbox"]))
                if distance <= self.match_distance:
                    candidates.append((distance, detection_index, track_id))
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

    def _new_track(self, detection: dict[str, Any], *, frame_id: str, observed_at: str) -> dict[str, Any]:
        label = str(detection["label"])
        next_id = self._next_ids.get(label, 0) + 1
        self._next_ids[label] = next_id
        return {
            "trackId": f"{label}-{next_id:03d}",
            "label": label,
            "bbox": dict(detection["bbox"]),
            "confidence": float(detection["confidence"]),
            "first_seen_frame": frame_id,
            "last_seen_frame": frame_id,
            "last_observed_at": observed_at,
            "missing_frames": 0,
        }


def _normalize_detection(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    label = str(raw.get("label") or raw.get("name") or raw.get("class") or "").strip()
    bbox = _normalize_bbox(raw.get("bbox"))
    if not label or bbox is None:
        return None
    return {"label": label, "bbox": bbox, "confidence": _coerce_float(raw.get("confidence", raw.get("score", 0.0)))}


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


def _fallback_event(
    *,
    event_type: str,
    observed_at: str,
    frame_id: str,
    track: Mapping[str, Any],
    from_region: str,
    to_region: str,
    distance: float,
) -> dict[str, Any]:
    return {
        "eventId": "",
        "eventType": event_type,
        "observedAt": observed_at,
        "sceneId": "",
        "subject": {"trackId": track["trackId"], "label": track["label"]},
        "confidence": round(float(track["confidence"]), 3),
        "details": {"fromRegion": from_region, "toRegion": to_region, "distance": round(float(distance), 3)},
        "metadata": {"frameId": frame_id},
    }


def _track_object(track: Mapping[str, Any]) -> dict[str, Any]:
    center = _center(track["bbox"])
    return {
        "trackId": track["trackId"],
        "label": track["label"],
        "confidence": round(float(track["confidence"]), 3),
        "bbox": dict(track["bbox"]),
        "center": {"x": round(center[0], 3), "y": round(center[1], 3)},
        "region": _region(center),
        "missingFrames": int(track.get("missing_frames", 0)),
    }


def _fallback_summary(objects: list[dict[str, Any]], events: list[dict[str, Any]]) -> str:
    observed = ", ".join(sorted({str(item.get("label")) for item in objects})) if objects else "empty scene"
    event_types = sorted({str(event.get("eventType")) for event in events if event.get("eventType")})
    return f"Observed {observed}; realtime events: {', '.join(event_types) if event_types else 'none'}"


def _fallback_scene_id(frame_id: str, observed_at: str, objects: list[dict[str, Any]]) -> str:
    payload = {"frameId": frame_id, "observedAt": observed_at, "objects": objects}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    return f"scene_rt_{digest}"


def _center(bbox: Mapping[str, Any]) -> tuple[float, float]:
    return ((float(bbox["x_min"]) + float(bbox["x_max"])) / 2.0, (float(bbox["y_min"]) + float(bbox["y_max"])) / 2.0)


def _region(center: tuple[float, float]) -> str:
    x_name = "left" if center[0] < 0.33 else "center" if center[0] < 0.66 else "right"
    y_name = "top" if center[1] < 0.25 else "middle" if center[1] < 0.75 else "bottom"
    return f"{x_name}_{y_name}"


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


__all__ = ["RealtimeVisionSceneBridge"]
