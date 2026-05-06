from __future__ import annotations

import json

from eibrain.cognition.vision_events import VisionEventShaper, shape_vision_event_contents


OBSERVED_AT = "2026-05-06T10:00:00.000+08:00"


def test_shapes_scene_lifecycle_and_motion_deltas_into_event_contents() -> None:
    shaper = VisionEventShaper(source="eye.realtime", dedupe_window_ms=500, movement_threshold=0.05)

    events = shaper.shape(
        scene_delta={
            "sceneId": "scene-front-1",
            "frameId": "frame-001",
            "appeared": [
                {
                    "trackId": "person-001",
                    "label": "person",
                    "confidence": 0.9349,
                    "region": "left_middle",
                }
            ],
            "left": [
                {
                    "track_id": "person-002",
                    "label": "person",
                    "confidence": 0.72,
                    "region": "right_middle",
                }
            ],
            "moved": [
                {
                    "trackId": "cup-001",
                    "label": "cup",
                    "confidence": 0.8119,
                    "fromRegion": "left_middle",
                    "toRegion": "center_middle",
                    "distance": 0.22,
                }
            ],
        },
        timestamp=OBSERVED_AT,
        freshness_ms=33.6,
    )

    assert [event["eventType"] for event in events] == [
        "person_appeared",
        "person_left",
        "object_moved",
    ]
    assert events[0]["eventId"] == "scene-front-1:person_appeared:person-001"
    assert events[0]["subject"] == {"trackId": "person-001", "label": "person"}
    assert events[0]["trackId"] == "person-001"
    assert events[0]["confidence"] == 0.935
    assert events[0]["timestamp"] == OBSERVED_AT
    assert events[0]["observedAt"] == OBSERVED_AT
    assert events[0]["source"] == "eye.realtime"
    assert events[0]["freshnessMs"] == 33.6
    assert events[0]["diagnostics"]["frameId"] == "frame-001"
    assert events[2]["details"] == {
        "fromRegion": "left_middle",
        "toRegion": "center_middle",
        "distance": 0.22,
    }
    json.dumps(events)


def test_shapes_target_lock_lost_attention_and_follow_state_changes() -> None:
    shaper = VisionEventShaper(source={"domain": "eihead", "instanceId": "honjia"})

    locked = shaper.shape(
        target_delta={
            "current": {"trackId": "person-001", "label": "person", "confidence": 0.91},
            "previous": None,
            "isLocked": True,
            "reason": "initial_lock",
            "diagnostics": {"candidate_count": 1},
        },
        timestamp=OBSERVED_AT,
        freshness_ms=12,
    )
    changed = shaper.shape(
        tracking_delta={
            "attention": {
                "previous": {"trackId": "person-001", "label": "person"},
                "current": {"trackId": "cup-001", "label": "cup", "confidence": 0.64},
                "reason": "salience_shift",
            },
            "followState": {
                "previous": "idle",
                "current": "following",
                "trackId": "cup-001",
                "label": "cup",
                "confidence": 0.7,
            },
        },
        timestamp="2026-05-06T10:00:00.100+08:00",
        freshness_ms=25,
    )
    lost = shaper.shape(
        target_delta={
            "previous": {"trackId": "person-001", "label": "person", "confidence": 0.91},
            "current": None,
            "isLocked": False,
            "reason": "target_lost_timeout",
        },
        timestamp="2026-05-06T10:00:00.250+08:00",
        freshness_ms=48,
    )

    events = locked + changed + lost

    assert [event["eventType"] for event in events] == [
        "target_locked",
        "attention_changed",
        "follow_state_changed",
        "target_lost",
    ]
    assert events[0]["subject"] == {"trackId": "person-001", "label": "person"}
    assert events[0]["diagnostics"]["reason"] == "initial_lock"
    assert events[0]["diagnostics"]["candidate_count"] == 1
    assert events[1]["subject"] == {"trackId": "cup-001", "label": "cup"}
    assert events[1]["diagnostics"]["previousSubject"] == {"trackId": "person-001", "label": "person"}
    assert events[2]["details"]["previousState"] == "idle"
    assert events[2]["details"]["currentState"] == "following"
    assert events[3]["diagnostics"]["reason"] == "target_lost_timeout"
    json.dumps(events)


def test_throttles_duplicate_events_but_allows_changed_subject_and_later_repeats() -> None:
    shaper = VisionEventShaper(dedupe_window_ms=500)
    scene_delta = {
        "sceneId": "scene-front-2",
        "appeared": [{"trackId": "person-001", "label": "person", "confidence": 0.9}],
    }

    first = shaper.shape(scene_delta=scene_delta, timestamp_ms=1_000)
    duplicate = shaper.shape(scene_delta=scene_delta, timestamp_ms=1_200)
    changed_subject = shaper.shape(
        scene_delta={
            "sceneId": "scene-front-2",
            "appeared": [{"trackId": "person-002", "label": "person", "confidence": 0.88}],
        },
        timestamp_ms=1_250,
    )
    later = shaper.shape(scene_delta=scene_delta, timestamp_ms=1_600)

    assert [event["eventType"] for event in first] == ["person_appeared"]
    assert duplicate == []
    assert [event["trackId"] for event in changed_subject] == ["person-002"]
    assert [event["trackId"] for event in later] == ["person-001"]


def test_skips_small_motion_to_avoid_frame_spam() -> None:
    shaper = VisionEventShaper(movement_threshold=0.1)

    events = shaper.shape(
        scene_delta={
            "sceneId": "scene-front-3",
            "moved": [
                {
                    "trackId": "cup-001",
                    "label": "cup",
                    "confidence": 0.8,
                    "fromRegion": "center_middle",
                    "toRegion": "center_middle",
                    "distance": 0.03,
                }
            ],
        },
        timestamp=OBSERVED_AT,
    )

    assert events == []


def test_function_helper_is_stateless_and_json_serializable() -> None:
    events = shape_vision_event_contents(
        scene_delta={
            "sceneId": "scene-front-4",
            "appeared": [{"trackId": "person-003", "label": "person", "confidence": 0.83}],
        },
        timestamp=OBSERVED_AT,
        source="eye.helper",
        freshness_ms=10,
    )

    assert events[0]["eventType"] == "person_appeared"
    assert events[0]["source"] == "eye.helper"
    assert json.loads(json.dumps(events)) == events


def test_derives_scene_events_from_previous_current_objects() -> None:
    shaper = VisionEventShaper(movement_threshold=0.08)

    events = shaper.shape(
        scene_delta={
            "sceneId": "scene-front-5",
            "frameId": "frame-005",
            "previous": {
                "objects": [
                    {
                        "trackId": "person-left",
                        "label": "person",
                        "confidence": 0.76,
                        "center": {"x": 0.20, "y": 0.50},
                        "region": "left_middle",
                    },
                    {
                        "trackId": "cup-001",
                        "label": "cup",
                        "confidence": 0.80,
                        "center": {"x": 0.20, "y": 0.50},
                        "region": "left_middle",
                    },
                ]
            },
            "current": {
                "objects": [
                    {
                        "trackId": "person-new",
                        "label": "person",
                        "confidence": 0.91,
                        "center": {"x": 0.70, "y": 0.50},
                        "region": "right_middle",
                    },
                    {
                        "trackId": "cup-001",
                        "label": "cup",
                        "confidence": 0.82,
                        "center": {"x": 0.38, "y": 0.50},
                        "region": "center_middle",
                    },
                ]
            },
        },
        timestamp=OBSERVED_AT,
    )

    assert [event["eventType"] for event in events] == [
        "person_appeared",
        "person_left",
        "object_moved",
    ]
    assert events[2]["trackId"] == "cup-001"
    assert events[2]["details"]["fromRegion"] == "left_middle"
    assert events[2]["details"]["toRegion"] == "center_middle"


def test_accepts_snapshot_style_target_and_tracking_deltas() -> None:
    shaper = VisionEventShaper()

    locked = shaper.shape(
        target_delta={
            "track_id": "person-locked",
            "label": "person",
            "confidence": 0.86,
            "is_locked": True,
            "lock_state": "locked",
            "switch_reason": "initial_lock",
            "diagnostics": {"candidate_count": 2},
        },
        timestamp=OBSERVED_AT,
    )
    tracking = shaper.shape(
        tracking_delta={
            "previous": {
                "follow_state": "idle",
                "attention": {"trackId": "person-locked", "label": "person"},
            },
            "current": {
                "follow_state": "following",
                "attention": {"trackId": "cup-locked", "label": "cup", "confidence": 0.74},
                "trackId": "cup-locked",
                "label": "cup",
            },
        },
        timestamp="2026-05-06T10:00:00.100+08:00",
    )
    lost = shaper.shape(
        target_delta={
            "track_id": None,
            "label": None,
            "is_locked": False,
            "lock_state": "unlocked",
            "switch_reason": "target_lost_timeout",
        },
        timestamp="2026-05-06T10:00:00.200+08:00",
    )

    assert [event["eventType"] for event in locked + tracking + lost] == [
        "target_locked",
        "attention_changed",
        "follow_state_changed",
        "target_lost",
    ]
    assert locked[0]["diagnostics"]["switchReason"] == "initial_lock"
    assert lost[0]["trackId"] == "person-locked"
