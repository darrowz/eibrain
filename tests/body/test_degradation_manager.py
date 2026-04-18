from __future__ import annotations


def test_degradation_manager_derives_capabilities() -> None:
    from eibrain.body.health.degradation_manager import DegradationManager
    from eibrain.body.health.organ_health import OrganHealth, SubfunctionHealth

    manager = DegradationManager()
    organ_states = [
        OrganHealth(
            organ="ear",
            health="degraded",
            subfunctions={
                "capture": SubfunctionHealth(name="capture", health="healthy"),
                "vad": SubfunctionHealth(name="vad", health="healthy"),
                "asr": SubfunctionHealth(name="asr", health="unavailable"),
            },
        ),
        OrganHealth(
            organ="mouth",
            health="healthy",
            subfunctions={
                "tts_plan": SubfunctionHealth(name="tts_plan", health="healthy"),
                "tts_playback": SubfunctionHealth(name="tts_playback", health="healthy"),
            },
        ),
    ]

    result = manager.evaluate(organ_states)

    assert result.capabilities.can_hear_voice is True
    assert result.capabilities.can_transcribe_speech is False
    assert result.capabilities.can_speak is True
    assert result.degradation_mode == "low_confidence_body"

