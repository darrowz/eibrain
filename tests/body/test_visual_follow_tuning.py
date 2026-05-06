from __future__ import annotations

import pytest


def _recommend(**overrides):
    from eibrain.body.visual_follow_tuning import (
        VisualFollowTuningTelemetry,
        recommend_visual_follow_tuning,
    )

    telemetry = {
        "filtered_error": 0.22,
        "stable_error_count": 4,
        "suppress_reason": None,
        "action_interval_s": 0.5,
        "fps": 24.0,
        "target_freshness_s": 0.05,
        "pan_proof_dx": -0.08,
        "pan_min": 40,
        "pan_max": 140,
        "current_angle": 90,
    }
    telemetry.update(overrides)
    return recommend_visual_follow_tuning(VisualFollowTuningTelemetry(**telemetry))


def test_visual_follow_tuning_widens_deadband_for_excessive_jitter() -> None:
    recommendation = _recommend(
        filtered_error=0.035,
        stable_error_count=0,
        suppress_reason="inside_deadband",
        pan_proof_dx=0.015,
    )

    assert recommendation.reason == "jitter_too_sensitive"
    assert recommendation.safe_to_apply is True
    assert recommendation.deadband > 0.08
    assert recommendation.step_gain < 24.0
    assert recommendation.max_step < 8
    assert recommendation.min_interval > 0.45
    assert recommendation.hold_frames >= 3


def test_visual_follow_tuning_gets_more_aggressive_when_head_does_not_move() -> None:
    recommendation = _recommend(
        filtered_error=0.32,
        stable_error_count=8,
        suppress_reason="rate_limited",
        action_interval_s=0.8,
        pan_proof_dx=0.004,
    )

    assert recommendation.reason == "underresponsive_no_motion"
    assert recommendation.safe_to_apply is True
    assert recommendation.deadband < 0.08
    assert recommendation.step_gain > 24.0
    assert recommendation.max_step > 8
    assert recommendation.min_interval < 0.45
    assert recommendation.hold_frames == 1


def test_visual_follow_tuning_dampens_when_pan_proof_overshoots() -> None:
    recommendation = _recommend(
        filtered_error=0.08,
        stable_error_count=3,
        pan_proof_dx=-0.24,
    )

    assert recommendation.reason == "overshoot"
    assert recommendation.safe_to_apply is True
    assert recommendation.deadband >= 0.08
    assert recommendation.step_gain < 24.0
    assert recommendation.max_step < 8
    assert recommendation.min_interval > 0.45
    assert recommendation.hold_frames >= 2


def test_visual_follow_tuning_refuses_to_apply_stale_target_recommendations() -> None:
    recommendation = _recommend(
        filtered_error=0.3,
        stable_error_count=6,
        target_freshness_s=1.4,
        pan_proof_dx=-0.1,
    )

    assert recommendation.reason == "target_stale"
    assert recommendation.safe_to_apply is False
    assert recommendation.deadband == pytest.approx(0.08)
    assert recommendation.step_gain == pytest.approx(24.0)
    assert recommendation.max_step == 8
    assert recommendation.min_interval == pytest.approx(0.45)
    assert recommendation.hold_frames == 2


def test_visual_follow_tuning_slows_commands_when_fps_is_low() -> None:
    recommendation = _recommend(
        filtered_error=0.22,
        stable_error_count=5,
        fps=7.5,
        pan_proof_dx=-0.05,
    )

    assert recommendation.reason == "fps_low"
    assert recommendation.safe_to_apply is True
    assert recommendation.step_gain < 24.0
    assert recommendation.max_step <= 6
    assert recommendation.min_interval > 0.45
    assert recommendation.hold_frames >= 3


def test_visual_follow_tuning_refuses_aggressive_changes_near_pan_boundary() -> None:
    recommendation = _recommend(
        filtered_error=0.31,
        stable_error_count=7,
        current_angle=139,
        pan_proof_dx=-0.02,
    )

    assert recommendation.reason == "near_pan_boundary"
    assert recommendation.safe_to_apply is False
    assert recommendation.deadband >= 0.08
    assert recommendation.step_gain < 24.0
    assert recommendation.max_step <= 2
    assert recommendation.min_interval >= 0.45


def test_visual_follow_tuning_flags_bias_confirmation_jitter() -> None:
    recommendation = _recommend(
        filtered_error=0.07,
        stable_error_count=1,
        suppress_reason="bias_not_confirmed",
        pan_proof_dx=0.01,
    )

    assert recommendation.reason == "jitter_too_sensitive"
    assert recommendation.safe_to_apply is True
    assert recommendation.deadband > 0.08
    assert recommendation.min_interval > 0.45
