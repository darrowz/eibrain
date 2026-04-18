from __future__ import annotations


def test_interrupt_skill_compiles_stop_speech_action() -> None:
    from eibrain.protocol.intents import PauseIntent
    from eibrain.skills.interrupt import InterruptForUserSkill

    skill = InterruptForUserSkill()
    actions = skill.compile(
        PauseIntent(
            ts=1.0,
            source="planner",
            session_id="s1",
            reason="user_interrupt",
            priority=100,
        )
    )

    assert len(actions) == 1
    assert actions[0].kind == "stop_speech_action"


def test_orient_skill_compiles_move_head_action() -> None:
    from eibrain.protocol.intents import OrientIntent
    from eibrain.skills.orient import OrientToSpeakerSkill

    skill = OrientToSpeakerSkill()
    actions = skill.compile(
        OrientIntent(
            ts=1.0,
            source="planner",
            session_id="s1",
            reason="track_speaker",
            priority=5,
            target_id="user-1",
            target_name="speaker",
            target_x=0.7,
        )
    )

    assert len(actions) == 1
    assert actions[0].kind == "move_head_action"
    assert actions[0].target_id == "user-1"
    assert actions[0].target_x == 0.7
