from __future__ import annotations

from eibrain.body.realtime_audio import PcmRingBuffer, RealtimeWakeDetector


class _Recognizer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = []

    def transcribe_audio_chunks(self, chunks, *, sample_rate: int, channels: int):
        self.calls.append(
            {
                "chunks": list(chunks),
                "sample_rate": sample_rate,
                "channels": channels,
            }
        )
        return self.text


def test_pcm_ring_buffer_keeps_recent_audio_window() -> None:
    ring = PcmRingBuffer(max_duration_ms=240, sample_rate=16000, channels=1)

    ring.append(b"a", duration_ms=80, captured_at_s=1.00)
    ring.append(b"b", duration_ms=80, captured_at_s=1.08)
    ring.append(b"c", duration_ms=80, captured_at_s=1.16)
    ring.append(b"d", duration_ms=80, captured_at_s=1.24)

    snapshot = ring.snapshot(duration_ms=160)

    assert snapshot.chunks == [b"c", b"d"]
    assert snapshot.duration_ms == 160
    assert snapshot.chunk_count == 2
    assert snapshot.sequence == 4
    assert ring.stats()["buffer_ms"] == 240


def test_realtime_wake_detector_emits_transcript_from_recent_ring_buffer() -> None:
    ring = PcmRingBuffer(max_duration_ms=2000, sample_rate=16000, channels=1)
    ring.append(b"wake-audio", duration_ms=480, captured_at_s=1.0)
    recognizer = _Recognizer("你好红图记住我喜欢短回答")
    detector = RealtimeWakeDetector(
        ring_buffer=ring,
        recognizer=recognizer,
        wake_words=("鸿途",),
        transcript_replacements={"红图": "鸿途"},
        session_id="sid",
        actor_id="actor",
        lookback_ms=1200,
        min_buffer_ms=200,
    )

    result = detector.poll_once()

    assert result is not None
    assert result.text == "你好鸿途记住我喜欢短回答"
    assert result.source == "ear.realtime_wake"
    assert detector.next_transcript(timeout_s=0.0) is result
    assert detector.snapshot()["wake_detector"]["emitted_count"] == 1
    assert recognizer.calls[0]["chunks"] == [b"wake-audio"]


def test_realtime_wake_detector_does_not_redecode_same_buffer_snapshot() -> None:
    ring = PcmRingBuffer(max_duration_ms=2000, sample_rate=16000, channels=1)
    ring.append(b"wake-audio", duration_ms=480, captured_at_s=1.0)
    recognizer = _Recognizer("你好鸿途")
    detector = RealtimeWakeDetector(
        ring_buffer=ring,
        recognizer=recognizer,
        wake_words=("鸿途",),
        min_buffer_ms=200,
    )

    assert detector.poll_once() is not None
    assert detector.poll_once() is None
    assert len(recognizer.calls) == 1


def test_realtime_wake_detector_skips_quiet_ring_buffer_before_asr_decode() -> None:
    ring = PcmRingBuffer(max_duration_ms=2000, sample_rate=16000, channels=1)
    ring.append(b"\x00\x00" * 1600, duration_ms=200, captured_at_s=1.0)
    recognizer = _Recognizer("你好鸿途")
    detector = RealtimeWakeDetector(
        ring_buffer=ring,
        recognizer=recognizer,
        wake_words=("鸿途",),
        min_buffer_ms=100,
        min_rms_level=0.01,
    )

    assert detector.poll_once() is None
    assert recognizer.calls == []
    assert detector.snapshot()["wake_detector"]["last_audio_stats"]["rms_level"] == 0.0
