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


def test_degradation_manager_does_not_treat_noop_identity_or_tracking_as_real_capability() -> None:
    from eibrain.body.health.degradation_manager import DegradationManager
    from eibrain.body.health.organ_health import OrganHealth, SubfunctionHealth

    manager = DegradationManager()
    organ_states = [
        OrganHealth(
            organ="eye",
            health="healthy",
            subfunctions={
                "camera": SubfunctionHealth(name="camera", health="healthy", details={"driver": "command"}),
                "detection": SubfunctionHealth(name="detection", health="healthy", details={"driver": "command"}),
                "identity": SubfunctionHealth(name="identity", health="healthy", details={"driver": "noop"}),
            },
        ),
        OrganHealth(
            organ="neck",
            health="healthy",
            subfunctions={
                "motor": SubfunctionHealth(name="motor", health="healthy", details={"driver": "command"}),
                "tracking": SubfunctionHealth(name="tracking", health="healthy", details={"driver": "noop"}),
            },
        ),
    ]

    result = manager.evaluate(organ_states)

    assert result.capabilities.can_see_people is True
    assert result.capabilities.can_identify_person is False
    assert result.capabilities.can_orient_head is True



def test_degradation_manager_keeps_transcribe_capability_on_silence() -> None:
    from eibrain.body.health.degradation_manager import DegradationManager
    from eibrain.body.health.organ_health import OrganHealth, SubfunctionHealth

    manager = DegradationManager()
    organ_states = [
        OrganHealth(
            organ="ear",
            health="degraded",
            subfunctions={
                "capture": SubfunctionHealth(name="capture", health="healthy", details={"driver": "command"}),
                "vad": SubfunctionHealth(name="vad", health="healthy", details={"driver": "command"}),
                "asr": SubfunctionHealth(name="asr", health="degraded", details={"driver": "command", "status": "silence"}),
            },
        ),
        OrganHealth(
            organ="mouth",
            health="healthy",
            subfunctions={
                "tts_plan": SubfunctionHealth(name="tts_plan", health="healthy", details={"driver": "command"}),
                "tts_playback": SubfunctionHealth(name="tts_playback", health="healthy", details={"driver": "command"}),
            },
        ),
    ]

    result = manager.evaluate(organ_states)

    assert result.capabilities.can_transcribe_speech is True
    assert result.degradation_mode == "normal"
