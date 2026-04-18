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
