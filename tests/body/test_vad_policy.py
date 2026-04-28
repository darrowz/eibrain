from __future__ import annotations


def test_vad_endpoint_policy_starts_and_stops_on_silence() -> None:
    from eibrain.body.vad_policy import VadEndpointPolicy, VadFrame

    policy = VadEndpointPolicy(
        rms_threshold=0.1,
        frame_ms=100,
        min_voice_ms=200,
        end_silence_ms=300,
        min_capture_ms=500,
        max_capture_ms=2000,
    )

    decisions = [
        policy.observe(VadFrame(rms_level=0.02)),
        policy.observe(VadFrame(rms_level=0.12)),
        policy.observe(VadFrame(rms_level=0.13)),
        policy.observe(VadFrame(rms_level=0.01)),
        policy.observe(VadFrame(rms_level=0.01)),
        policy.observe(VadFrame(rms_level=0.01)),
    ]

    assert decisions[2].should_start is True
    assert decisions[-1].should_stop is True
    assert decisions[-1].reason == "endpoint_silence"


def test_vad_endpoint_policy_accepts_silero_probability() -> None:
    from eibrain.body.vad_policy import VadEndpointPolicy, VadFrame

    policy = VadEndpointPolicy(rms_threshold=0.9, speech_probability_threshold=0.6, frame_ms=100, min_voice_ms=100)

    decision = policy.observe(VadFrame(rms_level=0.01, speech_probability=0.72))

    assert decision.is_voice is True
    assert decision.should_start is True


def test_vad_endpoint_policy_force_decodes_short_wake_energy() -> None:
    from eibrain.body.vad_policy import VadEndpointPolicy, VadFrame

    policy = VadEndpointPolicy(
        rms_threshold=0.2,
        fallback_rms_threshold=0.075,
        frame_ms=100,
        min_voice_ms=200,
        max_capture_ms=300,
    )

    decision = None
    for _ in range(3):
        decision = policy.observe(VadFrame(rms_level=0.08))

    assert decision is not None
    assert decision.should_force_decode is True
    assert decision.reason == "max_capture_force_decode"
