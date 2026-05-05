from __future__ import annotations

import pytest


def test_select_tracking_target_prefers_trackable_labels_and_reports_error() -> None:
    from eihead.eye.tracking import select_tracking_target

    detections = [
        {"label": "dog", "score": 0.99, "bbox": [10, 20, 620, 460], "track_id": "dog-1"},
        {"label": "person", "score": 0.72, "bbox": [260, 100, 420, 340], "track_id": "person-1"},
        {"label": "face", "score": 0.71, "bbox": [284, 132, 364, 244], "track_id": "face-1"},
    ]

    target = select_tracking_target(detections, frame_width=640, frame_height=480, frame_id="frame-7")

    assert target is not None
    assert target.label == "person"
    assert target.track_id == "person-1"
    assert target.frame_id == "frame-7"
    assert target.bbox == (260.0, 100.0, 420.0, 340.0)
    assert target.center_x == pytest.approx(340.0)
    assert target.center_y == pytest.approx(220.0)
    assert target.horizontal_error == pytest.approx((340.0 - 320.0) / 320.0)
    assert target.score == pytest.approx(0.72)


def test_select_tracking_target_accepts_realtime_bbox_variants() -> None:
    from eihead.eye.tracking import select_tracking_target

    detections = [
        {
            "label": "person",
            "score": 0.90,
            "bbox": {"x_min": 0.25, "y_min": 0.25, "x_max": 0.75, "y_max": 0.75},
            "trackId": "normalized-person",
        },
        {
            "label": "face",
            "score": 0.89,
            "bbox": {"x": 300, "y": 160, "w": 80, "h": 80},
            "trackId": "xywh-face",
        },
    ]

    target = select_tracking_target(detections, frame_width=640, frame_height=480)

    assert target is not None
    assert target.track_id == "normalized-person"
    assert target.bbox == (160.0, 120.0, 480.0, 360.0)
    assert target.horizontal_error == pytest.approx(0.0)


def test_select_tracking_target_uses_score_area_and_center_for_stable_ranking() -> None:
    from eihead.eye.tracking import select_tracking_target

    detections = [
        {"label": "person", "score": 0.80, "bbox": [500, 120, 620, 420], "track_id": "edge"},
        {"label": "person", "score": 0.82, "bbox": [240, 130, 400, 430], "track_id": "center"},
        {"label": "face", "score": 0.60, "bbox": [300, 180, 340, 220], "track_id": "tiny-face"},
    ]

    target = select_tracking_target(detections, frame_width=640, frame_height=480)

    assert target is not None
    assert target.track_id == "center"
    assert target.horizontal_error == pytest.approx(0.0)


def test_select_tracking_target_returns_none_for_empty_or_invalid_detections() -> None:
    from eihead.eye.tracking import select_tracking_target

    assert select_tracking_target([], frame_width=640, frame_height=480) is None
    assert select_tracking_target(
        [{"label": "person", "score": 0.9, "bbox": [10, 10, 10, 40]}],
        frame_width=640,
        frame_height=480,
    ) is None


def test_plan_pan_follow_action_outputs_smoothed_pan_only_command() -> None:
    from eihead.eye.tracking import TrackingTarget
    from eihead.neck.vision_follow import VisionFollowState, plan_pan_follow_action

    state = VisionFollowState(current_pan_deg=90.0, last_commanded_pan_deg=90.0)
    target = TrackingTarget(
        bbox=(400.0, 100.0, 560.0, 340.0),
        center_x=480.0,
        center_y=220.0,
        horizontal_error=0.5,
        score=0.92,
        label="person",
        track_id="person-1",
        frame_id="frame-8",
    )

    action = plan_pan_follow_action(target, state=state)

    assert action.mode == "track"
    assert action.pan_delta_deg == pytest.approx(5.0)
    assert action.pan_deg == pytest.approx(95.0)
    assert action.tilt_deg is None
    assert action.reason == "tracking"
    assert state.smoothed_error == pytest.approx(0.25)
    assert state.last_commanded_pan_deg == pytest.approx(95.0)
    assert state.current_pan_deg == pytest.approx(95.0)
    assert state.lost_frames == 0


def test_plan_pan_follow_action_suppresses_deadband_and_tiny_angle_changes() -> None:
    from eihead.eye.tracking import TrackingTarget
    from eihead.neck.vision_follow import VisionFollowConfig, VisionFollowState, plan_pan_follow_action

    state = VisionFollowState(current_pan_deg=90.0, last_commanded_pan_deg=90.0)
    tiny_target = TrackingTarget(
        bbox=(326.4, 100.0, 390.4, 240.0),
        center_x=358.4,
        center_y=170.0,
        horizontal_error=0.12,
        score=0.8,
        label="face",
        track_id=None,
        frame_id=None,
    )

    deadband = plan_pan_follow_action(tiny_target, state=state)
    assert deadband.mode == "hold"
    assert deadband.pan_delta_deg == 0.0
    assert deadband.reason == "deadband"

    small_step = plan_pan_follow_action(
        tiny_target,
        state=state,
        config=VisionFollowConfig(deadband=0.02, min_angle_delta_deg=2.0),
    )
    assert small_step.mode == "hold"
    assert small_step.pan_delta_deg == 0.0
    assert small_step.reason == "min_angle_delta"


def test_plan_pan_follow_action_holds_then_decays_when_target_is_lost() -> None:
    from eihead.neck.vision_follow import VisionFollowState, plan_pan_follow_action

    state = VisionFollowState(current_pan_deg=100.0, last_commanded_pan_deg=100.0, smoothed_error=0.4)

    first_lost = plan_pan_follow_action(None, state=state)
    second_lost = plan_pan_follow_action(None, state=state)
    third_lost = plan_pan_follow_action(None, state=state)
    fourth_lost = plan_pan_follow_action(None, state=state)

    assert first_lost.mode == "hold"
    assert first_lost.reason == "target_lost_hold"
    assert first_lost.pan_deg == pytest.approx(100.0)
    assert second_lost.mode == "hold"
    assert third_lost.mode == "decay"
    assert third_lost.reason == "target_lost_decay"
    assert third_lost.pan_delta_deg == pytest.approx(-2.0)
    assert third_lost.pan_deg == pytest.approx(98.0)
    assert fourth_lost.mode == "decay"
    assert fourth_lost.pan_deg == pytest.approx(96.0)
    assert state.smoothed_error == pytest.approx(0.0)
