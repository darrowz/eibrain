from __future__ import annotations


def test_fallback_policy_allows_normal_autorun_when_core_capabilities_are_online() -> None:
    from eibrain.body.health import CapabilityMatrix, FallbackPolicy

    policy = FallbackPolicy.from_capabilities(
        CapabilityMatrix(
            can_hear_voice=True,
            can_transcribe_speech=True,
            can_see_people=True,
            can_identify_person=True,
            can_speak=True,
            can_orient_head=True,
        )
    )

    assert policy.mode == "normal"
    assert policy.can_autorun is True
    assert policy.requires_confirmation is False
    assert policy.disabled_actions == ()
    assert policy.to_dict()["safe_actions"] == [
        "dialogue.listen",
        "dialogue.respond",
        "speech.play",
        "speech.stop",
        "head.move",
        "vision.track",
        "identity.recognize",
    ]


def test_fallback_policy_mutes_speech_when_tts_output_is_unavailable() -> None:
    from eibrain.body.health import CapabilityMatrix, FallbackPolicy

    policy = FallbackPolicy.from_capabilities(
        CapabilityMatrix(
            can_hear_voice=True,
            can_transcribe_speech=True,
            can_speak=False,
            can_orient_head=True,
        ),
        degradation_mode="mute_companion",
    )

    assert policy.mode == "mute_companion"
    assert policy.can_autorun is False
    assert policy.requires_confirmation is True
    assert "speech.play" in policy.disabled_actions
    assert "speech.stop" in policy.safe_actions
    assert "dialogue.respond" not in policy.safe_actions


def test_fallback_policy_blocks_final_dialogue_when_asr_confidence_is_low() -> None:
    from eibrain.body.health import CapabilityMatrix, FallbackPolicy

    policy = FallbackPolicy.from_capabilities(
        CapabilityMatrix(
            can_hear_voice=True,
            can_transcribe_speech=False,
            can_speak=True,
        ),
        degradation_mode="low_confidence_body",
    )

    assert policy.mode == "low_confidence_body"
    assert policy.can_autorun is False
    assert policy.requires_confirmation is True
    assert "dialogue.finalize" in policy.disabled_actions
    assert "dialogue.listen" in policy.safe_actions
    assert policy.operator_message


def test_fallback_policy_fixed_gaze_keeps_voice_but_blocks_head_motion() -> None:
    from eibrain.body.health import CapabilityMatrix, FallbackPolicy

    policy = FallbackPolicy.from_capabilities(
        CapabilityMatrix(
            can_hear_voice=True,
            can_transcribe_speech=True,
            can_speak=True,
            can_orient_head=False,
        ),
        degradation_mode="fixed_gaze",
    )

    assert policy.mode == "fixed_gaze"
    assert "head.move" in policy.disabled_actions
    assert "speech.play" in policy.safe_actions
    assert "speech.stop" in policy.safe_actions


def test_fallback_policy_keeps_voice_autorun_when_only_visual_capabilities_are_missing() -> None:
    from eibrain.body.health import CapabilityMatrix, FallbackPolicy

    policy = FallbackPolicy.from_capabilities(
        CapabilityMatrix(
            can_hear_voice=True,
            can_transcribe_speech=True,
            can_see_people=False,
            can_identify_person=False,
            can_speak=True,
            can_orient_head=True,
        ),
        degradation_mode="normal",
    )

    assert policy.mode == "normal"
    assert policy.reason == "visual_capabilities_partial"
    assert policy.can_autorun is True
    assert policy.requires_confirmation is False
    assert "dialogue.respond" in policy.safe_actions
    assert "vision.track" in policy.disabled_actions
