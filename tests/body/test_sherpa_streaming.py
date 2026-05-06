from __future__ import annotations

from pathlib import Path


def test_sherpa_streaming_recognizer_transcribes_chunks() -> None:
    from eibrain.body.sherpa_streaming import SherpaOnnxStreamingRecognizer

    built_sample_rates: list[int] = []

    class _Result:
        text = "ni hao"

    class _Stream:
        def __init__(self) -> None:
            self.accepted: list[list[float]] = []
            self.result = _Result()

        def accept_waveform(self, sample_rate: int, waveform: list[float]) -> None:
            self.accepted.append(waveform)

        def input_finished(self) -> None:
            return None

    class _Recognizer:
        def __init__(self) -> None:
            self.stream = _Stream()
            self.decoded = 0

        def create_stream(self) -> _Stream:
            return self.stream

        def is_ready(self, stream: _Stream) -> bool:
            return self.decoded == 0

        def decode_stream(self, stream: _Stream) -> None:
            self.decoded += 1

    def _factory(sample_rate: int):
        built_sample_rates.append(sample_rate)
        return _Recognizer()

    recognizer = SherpaOnnxStreamingRecognizer(model_dir="/models", recognizer_factory=_factory)

    text = recognizer.transcribe([b"\x00\x00\x10\x00"], sample_rate=16000, channels=1)

    assert text == "ni hao"
    assert built_sample_rates == [16000]


def test_sherpa_streaming_recognizer_handles_string_get_result() -> None:
    from eibrain.body.sherpa_streaming import SherpaOnnxStreamingRecognizer

    class _Stream:
        def accept_waveform(self, sample_rate: int, waveform: list[float]) -> None:
            return None

        def input_finished(self) -> None:
            return None

    class _Recognizer:
        def create_stream(self) -> _Stream:
            return _Stream()

        def is_ready(self, stream: _Stream) -> bool:
            return False

        def decode_stream(self, stream: _Stream) -> None:
            return None

        def get_result(self, stream: _Stream) -> str:
            return "你好鸿途"

    recognizer = SherpaOnnxStreamingRecognizer(model_dir="/models", recognizer_factory=lambda sample_rate: _Recognizer())

    text = recognizer.transcribe([b"\x00\x00\x10\x00"], sample_rate=16000, channels=1)

    assert text == "你好鸿途"


def test_sherpa_streaming_recognizer_downmixes_and_resamples_to_16k() -> None:
    from eibrain.body.sherpa_streaming import SherpaOnnxStreamingRecognizer

    accepted_lengths: list[int] = []

    class _Result:
        text = "resampled"

    class _Stream:
        def __init__(self) -> None:
            self.result = _Result()

        def accept_waveform(self, sample_rate: int, waveform: list[float]) -> None:
            accepted_lengths.append(len(waveform))

        def input_finished(self) -> None:
            return None

    class _Recognizer:
        def create_stream(self) -> _Stream:
            return _Stream()

        def is_ready(self, stream: _Stream) -> bool:
            return False

        def decode_stream(self, stream: _Stream) -> None:
            return None

    recognizer = SherpaOnnxStreamingRecognizer(model_dir="/models", recognizer_factory=lambda sample_rate: _Recognizer())

    recognizer.transcribe([b"\x00\x00\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00"], sample_rate=48000, channels=2)

    assert accepted_lengths == [1, 12800]


def test_body_runtime_builds_default_ear_processor_from_config() -> None:
    from apps.body_runtime.app import BodyRuntimeApp

    config_path = Path.cwd() / ".tmp-test-ear-config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "body:",
                "  organs:",
                "    ear:",
                "      capture:",
                "        driver:",
                "          kind: noop",
                "          device: default",
                "      vad:",
                "        driver:",
                "          kind: noop",
                "      asr:",
                "        driver:",
                "          kind: noop",
                "          provider: sherpa_onnx",
                "          model_dir: /models/default",
            ]
        ),
        encoding="utf-8",
    )
    runtime = BodyRuntimeApp.from_config_path(config_path)

    runtime._make_capture = lambda capture_cfg: ("capture", capture_cfg.driver.extra["device"])  # type: ignore[method-assign]
    runtime._make_recognizer = lambda asr_cfg: ("recognizer", asr_cfg.driver.extra["model_dir"])  # type: ignore[method-assign]

    try:
        processor = runtime.build_default_ear_processor()
        assert processor.capture == ("capture", "default")
        assert processor.recognizer == ("recognizer", "/models/default")
    finally:
        config_path.unlink(missing_ok=True)


def test_body_runtime_passes_streaming_vad_endpoint_settings_to_capture() -> None:
    from apps.body_runtime.app import BodyRuntimeApp
    from eibrain.infra.config import DriverConfig, SubfunctionConfig

    runtime = BodyRuntimeApp()
    capture_cfg = SubfunctionConfig(
        driver=DriverConfig(
            kind="command",
            extra={
                "device": "plughw:CARD=U4K,DEV=0",
                "sample_rate": 48000,
                "channels": 1,
                "streaming_vad": True,
                "vad_min_capture_ms": 1800,
                "transcribe_vad_miss": True,
                "vad_miss_rms_threshold": 0.012,
                "vad_endpoint_policy": True,
            },
        )
    )

    capture = runtime._make_capture(capture_cfg)

    assert capture.vad_min_capture_ms == 1800
    assert capture.transcribe_vad_miss is True
    assert capture.vad_miss_rms_threshold == 0.012
    assert capture.vad_endpoint_policy is True


def test_pcm_to_float_samples_uses_loudest_channel_for_stereo() -> None:
    from eibrain.body.sherpa_streaming import _pcm_to_float_samples

    quiet = 100
    loud = 3000
    pcm = b"".join(
        quiet.to_bytes(2, "little", signed=True) + loud.to_bytes(2, "little", signed=True)
        for _ in range(4)
    )

    samples = _pcm_to_float_samples(pcm, channels=2)

    assert samples == [loud / 32768.0] * 4
