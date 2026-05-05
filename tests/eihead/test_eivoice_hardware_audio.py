from eihead.devices.audio import (
    AcousticFrontendReadiness,
    AudioDeviceCandidate,
    PlaybackInterruptionPlan,
    build_aplay_command,
    build_arecord_command,
    evaluate_audio_frontend_readiness,
    select_preferred_input,
)


def test_select_preferred_input_prefers_u4k_over_spa3700() -> None:
    spa3700 = AudioDeviceCandidate(
        name="SPA3700 USB Audio",
        kind="input",
        device="hw:1,0",
        score=100,
        reason="appears first in ALSA list",
    )
    u4k = AudioDeviceCandidate(
        name="U4K capture",
        kind="input",
        device="hw:2,0",
        score=10,
        reason="confirmed usable field microphone",
    )

    selected = select_preferred_input([spa3700, u4k])

    assert selected.name == u4k.name
    assert selected.device == u4k.device
    assert selected.metadata["preferred_keyword"] == "U4K"
    assert selected.score > spa3700.score


def test_audio_device_helpers_are_exported_from_devices_package() -> None:
    from eihead.devices import AudioDeviceCandidate, select_preferred_input

    selected = select_preferred_input(
        [
            AudioDeviceCandidate(
                name="U4K capture",
                kind="input",
                device="hw:2,0",
                score=10,
                reason="confirmed usable field microphone",
            )
        ]
    )

    assert selected.device == "hw:2,0"


def test_select_preferred_input_degrades_spa3700_when_it_is_only_input() -> None:
    spa3700 = AudioDeviceCandidate(
        name="SPA3700 USB Audio",
        kind="input",
        device="hw:1,0",
        score=80,
        reason="listed input",
    )

    selected = select_preferred_input([spa3700])

    assert selected.name == "SPA3700 USB Audio"
    assert selected.score < 0
    assert "not confirmed usable" in selected.reason
    assert selected.metadata["degraded"] is True


def test_select_preferred_input_without_u4k_chooses_best_non_spa3700() -> None:
    candidates = [
        AudioDeviceCandidate("USB PnP Sound Device", "input", "hw:3,0", 25, "generic mic"),
        AudioDeviceCandidate("SPA3700 USB Audio", "input", "hw:1,0", 100, "listed input"),
        AudioDeviceCandidate("HD Webcam Microphone", "input", "hw:4,0", 40, "fallback mic"),
    ]

    selected = select_preferred_input(candidates)

    assert selected.name == "HD Webcam Microphone"
    assert "SPA3700" not in selected.name


def test_arecord_and_aplay_commands_are_16k_mono_pcm_without_execution() -> None:
    arecord = build_arecord_command("hw:2,0")
    aplay = build_aplay_command("hw:5,0")

    assert arecord == [
        "arecord",
        "-D",
        "hw:2,0",
        "-f",
        "S16_LE",
        "-r",
        "16000",
        "-c",
        "1",
        "--period-time",
        "60000",
    ]
    assert aplay == [
        "aplay",
        "-D",
        "hw:5,0",
        "-f",
        "S16_LE",
        "-r",
        "16000",
        "-c",
        "1",
    ]


def test_audio_frontend_readiness_degraded_without_loopback_or_aec() -> None:
    readiness = evaluate_audio_frontend_readiness(
        capture_device="hw:2,0",
        loopback_device=None,
        supports_aec=False,
        supports_ns=True,
        supports_vad=True,
    )

    assert isinstance(readiness, AcousticFrontendReadiness)
    assert readiness.capture is True
    assert readiness.healthy is False
    assert "loopback unavailable; speaker echo reference is degraded" in readiness.warnings
    assert "AEC unavailable; echo cancellation is degraded" in readiness.warnings
    assert readiness.to_dict()["healthy"] is False


def test_audio_frontend_readiness_capture_is_required() -> None:
    readiness = evaluate_audio_frontend_readiness(capture_device="")

    assert readiness.capture is False
    assert readiness.healthy is False
    assert "capture unavailable; microphone input is blocked" in readiness.warnings


def test_playback_interruption_plan_defaults_to_300ms_stop_command() -> None:
    plan = PlaybackInterruptionPlan(stop_command=["systemctl", "stop", "eivoice-playback"])

    assert plan.expected_max_ms == 300
    assert plan.reason == "barge-in requires playback stop before capture continues"
    assert plan.to_dict() == {
        "stop_command": ["systemctl", "stop", "eivoice-playback"],
        "reason": "barge-in requires playback stop before capture continues",
        "expected_max_ms": 300,
    }
