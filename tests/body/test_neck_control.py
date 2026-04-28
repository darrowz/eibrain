from __future__ import annotations


def test_neck_policy_filters_non_face_tracking_label() -> None:
    from eibrain.body.neck_control import NeckControlState, NeckIntent, NeckPolicy

    state = NeckControlState(last_angle=90, desired_angle=90)
    decision = NeckPolicy().decide(
        intent=NeckIntent(source="eye.tracking", target_name="potted plant", target_x=0.1, created_at_ts=1.0),
        state=state,
        now_ts=1.1,
    )

    assert decision.should_command is False
    assert decision.reason == "label_not_trackable"
    assert state.state == "holding"


def test_neck_policy_can_be_configured_to_track_faces_only() -> None:
    from eibrain.body.neck_control import NeckControlConfig, NeckControlState, NeckIntent, NeckPolicy

    policy = NeckPolicy(NeckControlConfig(allowed_tracking_labels=("face",)))
    state = NeckControlState(last_angle=90, desired_angle=90)
    person = policy.decide(
        intent=NeckIntent(source="eye.tracking", target_name="person", target_x=0.9, created_at_ts=1.0),
        state=state,
        now_ts=1.1,
    )
    face = policy.decide(
        intent=NeckIntent(source="eye.tracking", target_name="face", target_x=0.9, created_at_ts=1.2),
        state=state,
        now_ts=1.3,
    )

    assert person.should_command is False
    assert person.reason == "label_not_trackable"
    assert face.should_command is True


def test_neck_policy_commands_face_with_rate_limit() -> None:
    from eibrain.body.neck_control import NeckControlConfig, NeckControlState, NeckIntent, NeckPolicy

    policy = NeckPolicy(NeckControlConfig(min_command_interval_s=1.0, deadband=0.05, max_step=8))
    state = NeckControlState(last_angle=90, desired_angle=90)
    first = policy.decide(
        intent=NeckIntent(source="eye.tracking", target_name="face", target_x=0.9, created_at_ts=1.0),
        state=state,
        now_ts=1.1,
    )
    second = policy.decide(
        intent=NeckIntent(source="eye.tracking", target_name="face", target_x=0.95, created_at_ts=1.2),
        state=state,
        now_ts=1.3,
    )

    assert first.should_command is True
    assert first.angle > 90
    assert state.last_commanded_angle == first.angle
    assert second.should_command is False
    assert second.reason == "min_interval"


def test_neck_policy_suppresses_same_angle_and_expired_intent() -> None:
    from eibrain.body.neck_control import NeckControlState, NeckIntent, NeckPolicy

    policy = NeckPolicy()
    state = NeckControlState(last_angle=90, desired_angle=90, last_commanded_angle=90)
    same = policy.decide(
        intent=NeckIntent(source="manual_override", target_name="manual", target_angle=90, created_at_ts=1.0),
        state=state,
        now_ts=1.1,
    )
    expired = policy.decide(
        intent=NeckIntent(source="eye.tracking", target_name="face", target_x=0.9, ttl_s=0.1, created_at_ts=1.0),
        state=state,
        now_ts=2.0,
    )

    assert same.should_command is False
    assert same.reason == "same_angle"
    assert expired.should_command is False
    assert expired.reason == "intent_expired"
