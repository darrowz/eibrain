from __future__ import annotations

from importlib import import_module


def test_protocol_modules_are_importable() -> None:
    expected_modules = [
        "eibrain.protocol",
        "eibrain.protocol.base",
        "eibrain.protocol.observations",
        "eibrain.protocol.intents",
        "eibrain.protocol.actions",
        "eibrain.protocol.outcomes",
        "eibrain.protocol.envelopes",
    ]

    for module_name in expected_modules:
        module = import_module(module_name)
        assert module is not None


def test_protocol_models_expose_serializable_payloads() -> None:
    from eibrain.protocol.actions import PlaySpeechAction
    from eibrain.protocol.envelopes import Envelope
    from eibrain.protocol.intents import SpeakIntent
    from eibrain.protocol.observations import AudioTranscriptFinal
    from eibrain.protocol.outcomes import SpeechPlaybackCompleted

    observation = AudioTranscriptFinal(
        ts=1.0,
        source="ear.asr",
        text="hello",
        session_id="s1",
    )
    intent = SpeakIntent(
        ts=2.0,
        source="planner",
        reason="reply",
        priority=10,
        text="hi there",
        session_id="s1",
    )
    action = PlaySpeechAction(
        ts=3.0,
        source="reply-skill",
        text="hi there",
        session_id="s1",
    )
    outcome = SpeechPlaybackCompleted(
        ts=4.0,
        source="mouth.playback",
        session_id="s1",
    )
    envelope = Envelope.wrap(channel="observations", payload=observation)

    assert observation.to_dict()["kind"] == "audio_transcript_final"
    assert intent.to_dict()["kind"] == "speak_intent"
    assert action.to_dict()["kind"] == "play_speech_action"
    assert outcome.to_dict()["kind"] == "speech_playback_completed"
    assert envelope.channel == "observations"
    assert envelope.payload["session_id"] == "s1"
