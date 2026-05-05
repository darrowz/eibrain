from __future__ import annotations


def test_neck_fusion_holds_inside_center_deadband() -> None:
    from eibrain.body.neck_fusion import NeckFusionConfig, NeckFusionPolicy

    policy = NeckFusionPolicy(NeckFusionConfig(home_angle=90, deadband=0.08))

    recommendation = policy.recommend(
        target_x=0.53,
        score=0.96,
        current_angle=90,
        last_action=None,
        now_ts=1.0,
    )

    assert recommendation.action == "hold"
    assert recommendation.target_angle == 90
    assert recommendation.reason == "inside_deadband"


def test_neck_fusion_holds_low_confidence_target() -> None:
    from eibrain.body.neck_fusion import NeckFusionConfig, NeckFusionPolicy

    policy = NeckFusionPolicy(NeckFusionConfig(home_angle=90, min_confidence=0.5))

    recommendation = policy.recommend(
        target_x=0.84,
        score=0.31,
        current_angle=90,
        last_action=None,
        now_ts=1.0,
    )

    assert recommendation.action == "hold"
    assert recommendation.target_angle == 90
    assert recommendation.reason == "low_confidence"


def test_neck_fusion_requires_consecutive_offset_before_pan() -> None:
    from eibrain.body.neck_fusion import NeckFusionConfig, NeckFusionPolicy

    policy = NeckFusionPolicy(
        NeckFusionConfig(
            home_angle=90,
            deadband=0.08,
            hysteresis=0.03,
            consecutive_bias_required=2,
            pan_step_gain=30.0,
            max_step_degrees=10,
        )
    )

    first = policy.recommend(
        target_x=0.82,
        score=0.94,
        current_angle=90,
        last_action=None,
        now_ts=1.0,
    )
    second = policy.recommend(
        target_x=0.84,
        score=0.95,
        current_angle=90,
        last_action=first.last_action,
        now_ts=1.2,
    )

    assert first.action == "hold"
    assert first.reason == "bias_not_confirmed"
    assert second.action == "pan_right"
    assert second.target_angle > 90
    assert second.reason == "offset_confirmed"


def test_neck_fusion_rate_limits_followup_pan_commands() -> None:
    from eibrain.body.neck_fusion import NeckFusionConfig, NeckFusionPolicy

    policy = NeckFusionPolicy(
        NeckFusionConfig(
            home_angle=90,
            deadband=0.08,
            consecutive_bias_required=2,
            min_command_interval_s=0.5,
        )
    )

    first = policy.recommend(
        target_x=0.80,
        score=0.95,
        current_angle=90,
        last_action=None,
        now_ts=1.0,
    )
    second = policy.recommend(
        target_x=0.82,
        score=0.96,
        current_angle=90,
        last_action=first.last_action,
        now_ts=1.1,
    )
    third = policy.recommend(
        target_x=0.86,
        score=0.96,
        current_angle=second.target_angle,
        last_action=second.last_action,
        now_ts=1.3,
    )

    assert second.action == "pan_right"
    assert third.action == "hold"
    assert third.reason == "rate_limited"
    assert third.target_angle == second.target_angle


def test_neck_fusion_recenters_after_target_is_missing_for_delay() -> None:
    from eibrain.body.neck_fusion import NeckFusionConfig, NeckFusionPolicy

    policy = NeckFusionPolicy(
        NeckFusionConfig(
            home_angle=90,
            deadband=0.08,
            consecutive_bias_required=2,
            recenter_after_missing_s=0.6,
        )
    )

    first = policy.recommend(
        target_x=0.18,
        score=0.93,
        current_angle=90,
        last_action=None,
        now_ts=1.0,
    )
    second = policy.recommend(
        target_x=0.16,
        score=0.94,
        current_angle=90,
        last_action=first.last_action,
        now_ts=1.2,
    )
    missing_hold = policy.recommend(
        target_x=None,
        score=0.0,
        current_angle=second.target_angle,
        last_action=second.last_action,
        now_ts=1.5,
    )
    recenter = policy.recommend(
        target_x=None,
        score=0.0,
        current_angle=second.target_angle,
        last_action=missing_hold.last_action,
        now_ts=1.9,
    )

    assert second.action == "pan_left"
    assert missing_hold.action == "hold"
    assert missing_hold.reason == "target_missing_hold"
    assert recenter.action == "recenter"
    assert recenter.target_angle == 90
    assert recenter.reason == "target_missing_recenter"
