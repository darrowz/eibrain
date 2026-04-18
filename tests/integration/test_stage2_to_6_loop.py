from __future__ import annotations


def test_stage2_to_6_loop_updates_memory_learning_and_operator_report() -> None:
    from apps.body_runtime.app import BodyRuntimeApp
    from apps.cognitive_runtime.app import CognitiveRuntimeApp
    from apps.operator_console.app import OperatorConsoleApp

    body = BodyRuntimeApp()
    cognitive = CognitiveRuntimeApp()
    operator = OperatorConsoleApp()

    observation = body.simulate_transcript(
        text="hello eibrain",
        session_id="session-42",
        actor_id="user-1",
    )
    actions = cognitive.handle_observation(observation)
    outcomes = body.dispatch_actions(actions)

    report = operator.build_status_report(
        body_snapshot=body.snapshot(),
        cognitive_snapshot=cognitive.snapshot(),
        traces=cognitive.traces,
    )

    assert actions[0].kind == "play_speech_action"
    assert outcomes[0].kind == "speech_playback_completed"
    assert report["trace_count"] >= 1
    assert report["cognition"]["learning_decision"] in {"keep_policy", "adjust_policy"}
