from __future__ import annotations


def test_engagement_state_machine_transitions_for_voice_loop() -> None:
    from eibrain.cognition.policy.engine import EngagementStateMachine

    machine = EngagementStateMachine()

    assert machine.next_phase(current="idle", event="speech_started") == "listening"
    assert machine.next_phase(current="listening", event="transcript_final") == "thinking"
    assert machine.next_phase(current="thinking", event="reply_started") == "speaking"
    assert machine.next_phase(current="speaking", event="user_interrupt") == "interrupted"
    assert machine.next_phase(current="interrupted", event="listen_resumed") == "listening"

