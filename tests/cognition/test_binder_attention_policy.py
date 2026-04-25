from __future__ import annotations


def test_binder_builds_audio_moment_from_embodied_state() -> None:
    from eibrain.cognition.fusion.binder import ObservationBinder
    from eibrain.state.embodied import EmbodiedState

    state = EmbodiedState.create_default().with_transcript(
        text="hello",
        session_id="s1",
        actor_id="user-1",
        ts=1.0,
    )

    moment = ObservationBinder().bind(state)

    assert moment.transcript == "hello"
    assert moment.session_id == "s1"
    assert moment.actor_id == "user-1"
    assert moment.modalities == ("audio_text",)
    assert moment.query_text == "hello"


def test_attention_scores_audio_as_reply_worthy() -> None:
    from eibrain.cognition.attention.manager import AttentionManager
    from eibrain.cognition.fusion.binder import ObservationBinder
    from eibrain.state.embodied import EmbodiedState

    state = EmbodiedState.create_default().with_transcript(
        text="hello",
        session_id="s1",
        actor_id="user-1",
        ts=1.0,
    )

    salience = AttentionManager().evaluate(ObservationBinder().bind(state))

    assert salience.score >= 0.5
    assert salience.should_recall is True
    assert salience.should_reply is True
    assert salience.should_writeback is True


def test_attention_scores_visual_focus_as_orient_only() -> None:
    from eibrain.cognition.attention.manager import AttentionManager
    from eibrain.cognition.fusion.binder import ObservationBinder
    from eibrain.state.embodied import EmbodiedState

    state = EmbodiedState.create_default().with_visual_focus(
        target_name="person",
        actor_id="user-1",
        summary="a person is near the camera",
        target_x=0.7,
        ts=1.0,
    )

    salience = AttentionManager().evaluate(ObservationBinder().bind(state))

    assert salience.should_recall is True
    assert salience.should_reply is False
    assert salience.should_orient is True


def test_policy_decides_reply_or_orient_from_salience() -> None:
    from eibrain.cognition.attention.manager import AttentionManager
    from eibrain.cognition.fusion.binder import ObservationBinder
    from eibrain.cognition.policy.engine import PolicyEngine
    from eibrain.state.embodied import EmbodiedState

    binder = ObservationBinder()
    attention = AttentionManager()
    policy = PolicyEngine()
    audio_state = EmbodiedState.create_default().with_transcript(
        text="hello",
        session_id="s1",
        actor_id="user-1",
        ts=1.0,
    )
    visual_state = EmbodiedState.create_default().with_visual_focus(
        target_name="person",
        actor_id="user-1",
        summary="a person is near the camera",
        ts=1.0,
    )

    audio_decision = policy.decide(
        state=audio_state,
        moment=binder.bind(audio_state),
        salience=attention.evaluate(binder.bind(audio_state)),
    )
    visual_decision = policy.decide(
        state=visual_state,
        moment=binder.bind(visual_state),
        salience=attention.evaluate(binder.bind(visual_state)),
    )

    assert audio_decision.decision_type == "reply"
    assert audio_decision.should_reply is True
    assert visual_decision.decision_type == "orient"
    assert visual_decision.should_reply is False
    assert visual_decision.should_orient is True
