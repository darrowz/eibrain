from __future__ import annotations


def test_ear_organ_builds_audio_diagnostics() -> None:
    from eibrain.body.organs.ear.organ import EarOrgan
    from eibrain.infra.config import DriverConfig, OrganConfig, SubfunctionConfig

    class _Capture:
        device = "hw:3,0"
        sample_rate = 16000
        channels = 1

        def read_chunks(self, chunk_count: int):
            return [b"\x00\x00\x00\x08"] * chunk_count

    class _Recognizer:
        def transcribe(self, pcm_chunks, *, sample_rate: int, channels: int) -> str:
            return "ni hao honjia"

    config = OrganConfig(
        enabled=True,
        subfunctions={
            "capture": SubfunctionConfig(driver=DriverConfig(kind="command", command=["python"], extra={"device": "hw:3,0", "chunk_count": 2, "refresh_interval_s": 0.0})),
            "vad": SubfunctionConfig(driver=DriverConfig(kind="noop")),
            "asr": SubfunctionConfig(driver=DriverConfig(kind="command", command=["python"], extra={"model_dir": "/models"})),
        },
    )
    organ = EarOrgan(config=config)
    organ._capture = _Capture()
    organ._recognizer = _Recognizer()

    heartbeat = organ.heartbeat()

    assert heartbeat.health == "healthy"
    assert heartbeat.subfunctions["capture"].details["voice_activity"] is True
    assert heartbeat.subfunctions["asr"].details["transcript"] == "ni hao honjia"
    assert "heard speech" in heartbeat.subfunctions["asr"].details["speech_window_summary"]


def test_ear_organ_caches_audio_probe_results() -> None:
    from eibrain.body.organs.ear.organ import EarOrgan
    from eibrain.infra.config import DriverConfig, OrganConfig, SubfunctionConfig

    class _Capture:
        device = "hw:3,0"
        sample_rate = 16000
        channels = 1

        def __init__(self) -> None:
            self.calls = 0

        def read_chunks(self, chunk_count: int):
            self.calls += 1
            return [b"\x00\x00\x00\x08"] * chunk_count

    class _Recognizer:
        def transcribe(self, pcm_chunks, *, sample_rate: int, channels: int) -> str:
            return ""

    config = OrganConfig(
        enabled=True,
        subfunctions={
            "capture": SubfunctionConfig(driver=DriverConfig(kind="command", command=["python"], extra={"chunk_count": 1, "refresh_interval_s": 999.0})),
            "vad": SubfunctionConfig(driver=DriverConfig(kind="noop")),
            "asr": SubfunctionConfig(driver=DriverConfig(kind="command", command=["python"], extra={"model_dir": "/models"})),
        },
    )
    organ = EarOrgan(config=config)
    capture = _Capture()
    organ._capture = capture
    organ._recognizer = _Recognizer()

    organ.heartbeat()
    organ.heartbeat()

    assert capture.calls == 1


def test_ear_organ_passes_streaming_vad_endpoint_settings_to_capture() -> None:
    from eibrain.body.organs.ear.organ import EarOrgan
    from eibrain.infra.config import DriverConfig, OrganConfig, SubfunctionConfig

    config = OrganConfig(
        enabled=True,
        subfunctions={
            "capture": SubfunctionConfig(
                driver=DriverConfig(
                    kind="command",
                    command=["arecord"],
                    extra={
                        "device": "plughw:CARD=U4K,DEV=0",
                        "sample_rate": 48000,
                        "channels": 1,
                        "streaming_vad": True,
                        "vad_min_capture_ms": 1800,
                        "transcribe_vad_miss": True,
                        "vad_miss_rms_threshold": 0.012,
                        "vad_endpoint_policy": True,
                        "vad_backend": "adaptive_rms",
                        "vad_noise_ratio": 1.3,
                        "vad_silero_threshold": 0.42,
                    },
                )
            ),
            "vad": SubfunctionConfig(driver=DriverConfig(kind="noop")),
            "asr": SubfunctionConfig(driver=DriverConfig(kind="noop")),
        },
    )

    organ = EarOrgan(config=config)

    assert organ._capture is not None
    assert organ._capture.vad_min_capture_ms == 1800
    assert organ._capture.transcribe_vad_miss is True
    assert organ._capture.vad_miss_rms_threshold == 0.012
    assert organ._capture.vad_endpoint_policy is True
    assert organ._capture.vad_backend == "adaptive_rms"
    assert organ._capture.vad_noise_ratio == 1.3
    assert organ._capture.vad_silero_threshold == 0.42


def test_ear_organ_uses_faster_whisper_provider(monkeypatch) -> None:
    from eibrain.body.organs.ear.organ import EarOrgan
    from eibrain.infra.config import DriverConfig, OrganConfig, SubfunctionConfig

    class _Capture:
        device = "plughw:CARD=U4K,DEV=0"
        sample_rate = 48000
        channels = 2

        def read_window(self, duration_s: int):
            return [b"\x00\x00\x00\x08"] * duration_s

    calls = []

    def _fake_transcribe(**kwargs):
        calls.append(kwargs)
        return {"status": "ok", "details": {"text": "你好 honjia", "language": "zh"}}

    monkeypatch.setattr("eibrain.body.organs.ear.organ.transcribe_pcm_with_faster_whisper_subprocess", _fake_transcribe)

    config = OrganConfig(
        enabled=True,
        subfunctions={
            "capture": SubfunctionConfig(driver=DriverConfig(kind="command", command=["python"], extra={"device": "plughw:CARD=U4K,DEV=0", "chunk_count": 2, "refresh_interval_s": 0.0})),
            "vad": SubfunctionConfig(driver=DriverConfig(kind="noop")),
            "asr": SubfunctionConfig(
                driver=DriverConfig(
                    kind="command",
                    command=["python"],
                    extra={"provider": "faster_whisper", "model_name": "Systran/faster-whisper-tiny", "language": "zh", "vad_filter": False, "min_asr_dbfs": -120.0},
                )
            ),
        },
    )
    organ = EarOrgan(config=config)
    organ._capture = _Capture()
    organ._recognizer = None

    heartbeat = organ.heartbeat()

    assert heartbeat.subfunctions["asr"].details["transcript"] == "你好 honjia"
    assert heartbeat.subfunctions["asr"].health == "healthy"
    assert calls[0]["vad_filter"] is False


def test_ear_organ_reports_capture_failure_diagnostics() -> None:
    from eibrain.body.organs.ear.organ import EarOrgan
    from eibrain.infra.config import DriverConfig, OrganConfig, SubfunctionConfig

    class _Capture:
        device = "plughw:CARD=U4K,DEV=0"
        sample_rate = 48000
        channels = 2
        retry_count = 2
        last_returncode = 1
        last_stderr = "arecord: Device or resource busy"
        last_stdout_bytes = 0
        last_command = ["arecord", "-D", "plughw:CARD=U4K,DEV=0"]

        def read_window(self, duration_s: int):
            return []

    config = OrganConfig(
        enabled=True,
        subfunctions={
            "capture": SubfunctionConfig(driver=DriverConfig(kind="command", command=["python"], extra={"chunk_count": 2, "refresh_interval_s": 0.0})),
            "vad": SubfunctionConfig(driver=DriverConfig(kind="noop")),
            "asr": SubfunctionConfig(driver=DriverConfig(kind="command", command=["python"], extra={"provider": "faster_whisper"})),
        },
    )
    organ = EarOrgan(config=config)
    organ._capture = _Capture()
    organ._recognizer = None

    heartbeat = organ.heartbeat()

    details = heartbeat.subfunctions["capture"].details
    assert heartbeat.subfunctions["capture"].health == "degraded"
    assert details["capture_returncode"] == 1
    assert details["capture_stdout_bytes"] == 0
    assert details["capture_retry_count"] == 2
    assert details["error"] == "arecord: Device or resource busy"


def test_ear_organ_does_not_capture_when_asr_is_noop() -> None:
    from eibrain.body.organs.ear.organ import EarOrgan
    from eibrain.infra.config import DriverConfig, OrganConfig, SubfunctionConfig

    config = OrganConfig(
        enabled=True,
        subfunctions={
            "capture": SubfunctionConfig(driver=DriverConfig(kind="command", command=["python"])),
            "vad": SubfunctionConfig(driver=DriverConfig(kind="noop")),
            "asr": SubfunctionConfig(driver=DriverConfig(kind="noop")),
        },
    )
    organ = EarOrgan(config=config)

    heartbeat = organ.heartbeat()

    assert heartbeat.subfunctions["capture"].health == "healthy"
    assert "capture_command" not in heartbeat.subfunctions["capture"].details


def test_ear_organ_applies_transcript_replacements(monkeypatch) -> None:
    from eibrain.body.organs.ear.organ import EarOrgan
    from eibrain.infra.config import DriverConfig, OrganConfig, SubfunctionConfig

    class _Capture:
        device = "plughw:CARD=U4K,DEV=0"
        sample_rate = 48000
        channels = 2

        def read_window(self, duration_s: int):
            return [b"\x00\x00\x00\x08"] * duration_s

    def _fake_transcribe(**kwargs):
        return {"status": "ok", "details": {"text": "鸿好鸿途。鸿好鸿途。", "language": "zh"}}

    monkeypatch.setattr("eibrain.body.organs.ear.organ.transcribe_pcm_with_faster_whisper_subprocess", _fake_transcribe)

    config = OrganConfig(
        enabled=True,
        subfunctions={
            "capture": SubfunctionConfig(driver=DriverConfig(kind="command", command=["python"], extra={"chunk_count": 2, "refresh_interval_s": 0.0})),
            "vad": SubfunctionConfig(driver=DriverConfig(kind="noop")),
            "asr": SubfunctionConfig(
                driver=DriverConfig(
                    kind="command",
                    command=["python"],
                    extra={
                        "provider": "faster_whisper",
                            "transcript_replacements": {
                                "鸿好鸿途": "你好鸿途",
                                "您好鸿途": "你好鸿途",
                            },
                            "min_asr_dbfs": -120.0,
                        },
                )
            ),
        },
    )
    organ = EarOrgan(config=config)
    organ._capture = _Capture()
    organ._recognizer = None

    heartbeat = organ.heartbeat()

    assert heartbeat.subfunctions["asr"].details["transcript"] == "你好鸿途。"
