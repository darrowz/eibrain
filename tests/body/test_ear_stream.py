from __future__ import annotations


def test_arecord_stream_capture_builds_expected_command() -> None:
    from eibrain.body.ear_stream import ArecordStreamCapture

    capture = ArecordStreamCapture(
        device="hw:3,0",
        sample_rate=16000,
        channels=1,
    )

    assert capture.build_command() == [
        "arecord",
        "-D",
        "hw:3,0",
        "-f",
        "S16_LE",
        "-r",
        "16000",
        "-c",
        "1",
        "-t",
        "raw",
    ]


def test_arecord_stream_capture_retries_empty_capture(monkeypatch) -> None:
    from eibrain.body.ear_stream import ArecordStreamCapture

    class _Completed:
        def __init__(self, *, returncode: int, stdout: bytes, stderr: bytes = b"") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    calls = []
    results = [
        _Completed(returncode=1, stdout=b"", stderr=b"device busy"),
        _Completed(returncode=0, stdout=b"abcd"),
    ]

    def _run(command, **kwargs):
        calls.append((command, kwargs))
        return results.pop(0)

    monkeypatch.setattr("eibrain.body.ear_stream.subprocess.run", _run)
    monkeypatch.setattr("eibrain.body.ear_stream.time.sleep", lambda _delay: None)

    capture = ArecordStreamCapture(
        device="plughw:CARD=U4K,DEV=0",
        sample_rate=48000,
        channels=2,
        retry_count=1,
    )

    chunks = capture.read_window(1, chunk_bytes=2)

    assert chunks == [b"ab", b"cd"]
    assert len(calls) == 2
    assert capture.last_returncode == 0
    assert capture.last_stderr == ""
    assert capture.last_stdout_bytes == 4
    assert capture.last_command == capture.build_command() + ["-d", "1"]


def test_arecord_stream_capture_streaming_vad_stops_after_voice(monkeypatch) -> None:
    from eibrain.body.ear_stream import ArecordStreamCapture

    frame_bytes = 1600 * 2
    silence = (0).to_bytes(2, "little", signed=True) * 1600
    voice = (2000).to_bytes(2, "little", signed=True) * 1600
    frames = [silence, silence, voice, voice, silence, silence, silence, silence, voice, voice]

    class _Stdout:
        def __init__(self) -> None:
            self.frames = list(frames)

        def read(self, size: int) -> bytes:
            assert size == frame_bytes
            if not self.frames:
                return b""
            return self.frames.pop(0)

    class _Stderr:
        def read(self) -> bytes:
            return b""

    class _Process:
        def __init__(self, command, **kwargs) -> None:
            self.command = command
            self.stdout = _Stdout()
            self.stderr = _Stderr()
            self.returncode = None
            self.terminated = False

        def poll(self):
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def wait(self, timeout: float | None = None):
            self.returncode = self.returncode if self.returncode is not None else 0
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    monkeypatch.setattr("eibrain.body.ear_stream.subprocess.Popen", _Process)

    capture = ArecordStreamCapture(
        device="hw:3,0",
        sample_rate=16000,
        channels=1,
        streaming_vad=True,
        vad_frame_ms=100,
        vad_rms_threshold=0.03,
        vad_min_voice_ms=200,
        vad_end_silence_ms=300,
        vad_pre_roll_ms=200,
    )

    chunks = capture.read_window(2, chunk_bytes=frame_bytes)

    assert len(chunks) == 6
    assert capture.last_vad_triggered is True
    assert capture.last_vad_voice_frame_count == 2
    assert capture.last_vad_frame_count == 7
    assert capture.last_stdout_bytes == frame_bytes * 6
    assert capture.last_command == capture.build_command()


def test_ear_stream_processor_emits_audio_transcript_observation() -> None:
    from eibrain.body.ear_stream import EarStreamProcessor

    class _Capture:
        sample_rate = 16000
        channels = 1

        def read_chunks(self, chunk_count: int):
            assert chunk_count == 2
            return [b"a", b"b"]

    class _Recognizer:
        def transcribe(self, pcm_chunks: list[bytes], *, sample_rate: int, channels: int) -> str:
            assert pcm_chunks == [b"a", b"b"]
            assert sample_rate == 16000
            assert channels == 1
            return "ni hao eibrain"

    processor = EarStreamProcessor(capture=_Capture(), recognizer=_Recognizer())

    observation = processor.transcribe_window(
        chunk_count=2,
        session_id="session-1",
        actor_id="user-1",
    )

    assert observation.text == "ni hao eibrain"
    assert observation.session_id == "session-1"


def test_ear_stream_processor_prefers_vad_window_capture() -> None:
    from eibrain.body.ear_stream import EarStreamProcessor

    class _Capture:
        sample_rate = 48000
        channels = 2

        def read_window(self, duration_s: int):
            assert duration_s == 3
            return [b"voice"]

        def read_chunks(self, chunk_count: int):  # pragma: no cover - should not be used when VAD is available
            raise AssertionError("read_chunks bypasses streaming VAD")

    class _Recognizer:
        def transcribe(self, pcm_chunks: list[bytes], *, sample_rate: int, channels: int) -> str:
            assert pcm_chunks == [b"voice"]
            assert sample_rate == 48000
            assert channels == 2
            return "你好鸿途"

    processor = EarStreamProcessor(capture=_Capture(), recognizer=_Recognizer())

    observation = processor.transcribe_window(
        chunk_count=3,
        session_id="session-2",
        actor_id="user-2",
    )

    assert observation.text == "你好鸿途"
