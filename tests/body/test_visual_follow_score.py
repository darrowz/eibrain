from __future__ import annotations

import pytest


def test_visual_follow_score_successfully_scores_converged_command() -> None:
    from eibrain.body.visual_follow_score import (
        VisualFollowScoreConfig,
        score_visual_follow,
    )

    result = score_visual_follow(
        before_error=0.32,
        after_error=0.04,
        command_angle_delta=8.0,
        target_age_s=0.08,
        action_elapsed_s=0.18,
        settle_time_s=0.28,
        config=VisualFollowScoreConfig(settle_tolerance=0.06),
    )

    assert result.success is True
    assert result.score >= 0.85
    assert result.error_reduced is True
    assert result.overshoot is False
    assert result.settled is True
    assert result.reason == "settled_reduced_error"
    assert result.metrics["error_reduction"] == pytest.approx(0.28)
    assert result.metrics["error_reduction_ratio"] == pytest.approx(0.875)


def test_visual_follow_score_rejects_stale_target() -> None:
    from eibrain.body.visual_follow_score import (
        VisualFollowScoreConfig,
        score_visual_follow,
    )

    result = score_visual_follow(
        before_error=0.28,
        after_error=0.03,
        command_angle_delta=6.0,
        target_age_s=1.4,
        action_elapsed_s=0.15,
        settle_time_s=0.2,
        config=VisualFollowScoreConfig(max_target_age_s=0.5),
    )

    assert result.success is False
    assert result.score == 0.0
    assert result.error_reduced is True
    assert result.overshoot is False
    assert result.settled is True
    assert result.reason == "target_stale"
    assert result.metrics["target_fresh"] is False


def test_visual_follow_score_marks_suppressed_jitter_as_unscored_hold() -> None:
    from eibrain.body.visual_follow_score import score_visual_follow

    result = score_visual_follow(
        before_error=0.05,
        after_error=0.045,
        command_angle_delta=0.0,
        target_age_s=0.05,
        action_elapsed_s=0.0,
        settle_time_s=0.0,
        suppressed=True,
        suppressed_reason="inside_deadband",
    )

    assert result.success is False
    assert result.score == 0.0
    assert result.error_reduced is False
    assert result.overshoot is False
    assert result.settled is True
    assert result.reason == "suppressed_inside_deadband"
    assert result.metrics["suppressed"] is True
    assert result.metrics["held"] is True


def test_visual_follow_score_penalizes_overshoot() -> None:
    from eibrain.body.visual_follow_score import (
        VisualFollowScoreConfig,
        score_visual_follow,
    )

    result = score_visual_follow(
        before_error=0.18,
        after_error=-0.12,
        command_angle_delta=10.0,
        target_age_s=0.05,
        action_elapsed_s=0.2,
        settle_time_s=0.35,
        config=VisualFollowScoreConfig(settle_tolerance=0.05, overshoot_tolerance=0.04),
    )

    assert result.success is False
    assert 0.0 < result.score < 0.8
    assert result.error_reduced is True
    assert result.overshoot is True
    assert result.settled is False
    assert result.reason == "overshot_target"
    assert result.metrics["crossed_center"] is True


def test_visual_follow_score_treats_no_command_hold_as_success_when_settled() -> None:
    from eibrain.body.visual_follow_score import (
        VisualFollowScoreConfig,
        score_visual_follow,
    )

    result = score_visual_follow(
        before_error=0.025,
        after_error=0.02,
        command_angle_delta=0.0,
        target_age_s=0.04,
        action_elapsed_s=0.0,
        settle_time_s=0.0,
        held=True,
        config=VisualFollowScoreConfig(settle_tolerance=0.05),
    )

    assert result.success is True
    assert result.score == pytest.approx(1.0)
    assert result.error_reduced is False
    assert result.overshoot is False
    assert result.settled is True
    assert result.reason == "held_settled"
    assert result.metrics["held"] is True
    assert result.metrics["commanded"] is False
