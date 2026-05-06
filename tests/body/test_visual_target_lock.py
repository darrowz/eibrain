from __future__ import annotations


def _det(
    track_id: str,
    label: str,
    confidence: float,
    bbox: dict[str, float],
    **extra: object,
) -> dict[str, object]:
    detection: dict[str, object] = {
        "track_id": track_id,
        "label": label,
        "confidence": confidence,
        "bbox": bbox,
    }
    detection.update(extra)
    return detection


def test_visual_target_lock_keeps_existing_person_across_minor_score_jitter() -> None:
    from eibrain.body.visual_target_lock import (
        VisualTargetLockConfig,
        VisualTargetLockSelector,
    )

    selector = VisualTargetLockSelector(
        VisualTargetLockConfig(switch_hysteresis=0.75)
    )

    first = selector.select(
        [
            _det("person-1", "person", 0.88, _bbox(0.36, 0.20, 0.56, 0.72)),
            _det("person-2", "person", 0.76, _bbox(0.62, 0.20, 0.82, 0.72)),
        ],
        now_ts=1.0,
    )
    second = selector.select(
        [
            _det("person-1", "person", 0.86, _bbox(0.37, 0.20, 0.57, 0.72)),
            _det("person-2", "person", 0.94, _bbox(0.62, 0.20, 0.84, 0.74)),
        ],
        now_ts=1.2,
    )

    assert first.track_id == "person-1"
    assert second.track_id == "person-1"
    assert second.is_locked is True
    assert second.lock_state == "locked"
    assert second.switch_reason == "maintained"
    assert second.center == {"x": 0.47, "y": 0.46}


def test_visual_target_lock_prioritizes_registered_identity_over_faces() -> None:
    from eibrain.body.visual_target_lock import VisualTargetLockSelector

    selector = VisualTargetLockSelector()

    result = selector.select(
        [
            _det("face-1", "face", 0.99, _bbox(0.44, 0.16, 0.54, 0.36)),
            _det(
                "person-registered",
                "person",
                0.58,
                _bbox(0.12, 0.18, 0.38, 0.86),
                person_id="alice",
            ),
        ],
        now_ts=2.0,
    )

    assert result.track_id == "person-registered"
    assert result.label == "person"
    assert result.target is not None
    assert result.target["person_id"] == "alice"
    assert result.diagnostics["selected_priority"] == "identity"
    assert result.switch_reason == "initial_lock"


def test_visual_target_lock_holds_then_releases_briefly_lost_target() -> None:
    from eibrain.body.visual_target_lock import (
        VisualTargetLockConfig,
        VisualTargetLockSelector,
    )

    selector = VisualTargetLockSelector(VisualTargetLockConfig(lost_timeout=0.8))

    locked = selector.select(
        [_det("face-1", "face", 0.95, _bbox(0.42, 0.12, 0.58, 0.38))],
        now_ts=10.0,
    )
    held = selector.select([], now_ts=10.4)
    released = selector.select([], now_ts=10.9)

    assert locked.track_id == "face-1"
    assert locked.lock_id == "face-1"
    assert held.track_id == "face-1"
    assert held.lock_id == "face-1"
    assert held.lock_state == "lost_hold"
    assert held.switch_reason == "target_lost_hold"
    assert held.is_locked is True
    assert released.track_id is None
    assert released.lock_id is None
    assert released.target is None
    assert released.lock_state == "unlocked"
    assert released.switch_reason == "target_lost_timeout"
    assert released.is_locked is False


def test_visual_target_lock_switches_to_stronger_target_after_age_and_hysteresis() -> None:
    from eibrain.body.visual_target_lock import (
        VisualTargetLockConfig,
        VisualTargetLockSelector,
    )

    selector = VisualTargetLockSelector(
        VisualTargetLockConfig(switch_hysteresis=0.2, target_age=0.3)
    )

    first = selector.select(
        [_det("person-1", "person", 0.82, _bbox(0.36, 0.18, 0.60, 0.84))],
        now_ts=20.0,
    )
    too_new = selector.select(
        [
            _det("person-1", "person", 0.80, _bbox(0.36, 0.18, 0.60, 0.84)),
            _det("face-1", "face", 0.98, _bbox(0.62, 0.10, 0.78, 0.36)),
        ],
        now_ts=20.1,
    )
    switched = selector.select(
        [
            _det("person-1", "person", 0.78, _bbox(0.36, 0.18, 0.60, 0.84)),
            _det("face-1", "face", 0.99, _bbox(0.62, 0.10, 0.78, 0.36)),
        ],
        now_ts=20.5,
    )

    assert first.track_id == "person-1"
    assert too_new.track_id == "person-1"
    assert too_new.switch_reason == "candidate_too_new"
    assert switched.track_id == "face-1"
    assert switched.lock_state == "switched"
    assert switched.switch_reason == "stronger_target"
    assert switched.diagnostics["previous_track_id"] == "person-1"


def test_visual_target_lock_filters_low_confidence_candidates() -> None:
    from eibrain.body.visual_target_lock import (
        VisualTargetLockConfig,
        VisualTargetLockSelector,
    )

    selector = VisualTargetLockSelector(VisualTargetLockConfig(min_confidence=0.6))

    result = selector.select(
        [_det("face-weak", "face", 0.59, _bbox(0.40, 0.12, 0.56, 0.36))],
        now_ts=30.0,
    )

    assert result.target is None
    assert result.track_id is None
    assert result.is_locked is False
    assert result.lock_state == "unlocked"
    assert result.switch_reason == "no_eligible_target"
    assert result.diagnostics["filtered_low_confidence"] == 1


def test_visual_target_lock_reports_empty_unlocked_state_without_targets() -> None:
    from eibrain.body.visual_target_lock import VisualTargetLockSelector

    selector = VisualTargetLockSelector()

    result = selector.select([], now_ts=40.0)

    assert result.track_id is None
    assert result.lock_id is None
    assert result.label is None
    assert result.bbox is None
    assert result.center is None
    assert result.confidence == 0.0
    assert result.target is None
    assert result.is_locked is False
    assert result.lock_state == "unlocked"
    assert result.switch_reason == "no_target"
    assert result.diagnostics["candidate_count"] == 0


def test_visual_target_lock_keeps_lock_id_stable_when_tracker_id_jitters() -> None:
    from eibrain.body.visual_target_lock import VisualTargetLockSelector

    selector = VisualTargetLockSelector()

    first = selector.select(
        [_det("person-1", "person", 0.84, _bbox(0.34, 0.18, 0.58, 0.84))],
        now_ts=50.0,
    )
    relinked = selector.select(
        [_det("person-1b", "person", 0.83, _bbox(0.35, 0.18, 0.59, 0.84))],
        now_ts=50.2,
    )

    assert first.track_id == "person-1"
    assert first.lock_id == "person-1"
    assert relinked.track_id == "person-1b"
    assert relinked.lock_id == "person-1"
    assert relinked.lock_state == "locked"
    assert relinked.switch_reason == "maintained"
    assert relinked.diagnostics["lock_track_id"] == "person-1"


def _bbox(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> dict[str, float]:
    return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}
