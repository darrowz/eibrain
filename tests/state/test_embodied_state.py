from __future__ import annotations


def test_embodied_state_exposes_phase1_domains() -> None:
    from eibrain.body.health.capability_matrix import CapabilityMatrix
    from eibrain.state.embodied import EmbodiedState

    state = EmbodiedState.create_default()

    assert state.body.organs == {}
    assert state.world.current_speaker_id is None
    assert state.self_state.mode == "idle"
    assert state.session.active_session_id is None
    assert state.engagement.phase == "idle"
    assert isinstance(state.capabilities, CapabilityMatrix)


def test_embodied_state_can_transition_to_listening() -> None:
    from eibrain.protocol.observations import AudioTranscriptFinal
    from eibrain.state.embodied import EmbodiedState

    state = EmbodiedState.create_default()
    next_state = state.with_observation(
        AudioTranscriptFinal(
            ts=10.0,
            source="ear.asr",
            text="hello there",
            session_id="session-1",
            actor_id="user-1",
        )
    )

    assert next_state.world.last_transcript == "hello there"
    assert next_state.world.current_speaker_id == "user-1"
    assert next_state.session.active_session_id == "session-1"
    assert next_state.engagement.phase == "listening"
    assert next_state.self_state.mode == "attending"
