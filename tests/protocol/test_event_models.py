from __future__ import annotations


def test_observation_event_to_dict_is_serializable() -> None:
    from eibrain.protocol.events import ObservationEvent

    event = ObservationEvent(
        ts=1.0,
        source="ear.asr",
        session_id="s1",
        actor_id="user-1",
        modality="audio_text",
        text="hello",
        payload={"language": "en"},
    )

    assert event.to_dict()["kind"] == "observation_event"
    assert event.to_dict()["payload"] == {"language": "en"}


def test_salience_decision_defaults_are_safe() -> None:
    from eibrain.protocol.events import SalienceDecision

    decision = SalienceDecision(score=0.0, reason="idle")

    assert decision.should_recall is True
    assert decision.should_reply is False
    assert decision.should_writeback is False
