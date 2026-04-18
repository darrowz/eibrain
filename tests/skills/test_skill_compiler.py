from __future__ import annotations


def test_skill_compiler_maps_speak_and_orient_intents_to_actions() -> None:
    from eibrain.protocol.intents import OrientIntent, SpeakIntent
    from eibrain.skills.compiler import SkillCompiler

    compiler = SkillCompiler()
    actions = compiler.compile(
        [
            SpeakIntent(
                ts=1.0,
                source="planner",
                session_id="s1",
                reason="reply",
                priority=10,
                text="hello",
            ),
            OrientIntent(
                ts=1.0,
                source="planner",
                session_id="s1",
                reason="track",
                priority=5,
                target_id="user-1",
                target_name="speaker",
            ),
        ]
    )

    assert len(actions) == 2
    assert actions[0].kind == "play_speech_action"
    assert actions[0].text == "hello"
    assert actions[1].kind == "move_head_action"
