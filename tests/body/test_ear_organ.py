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
