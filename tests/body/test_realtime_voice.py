from __future__ import annotations


def test_realtime_voice_session_tracks_streaming_latency() -> None:
    from eibrain.body.realtime_voice import RealtimeVoiceSession

    ticks = iter([0.0, 0.1, 0.25, 0.8, 1.1, 1.3, 2.0])
    session = RealtimeVoiceSession(session_id="s1", actor_id="darrow", clock=lambda: next(ticks))

    session.start_listening()
    session.note_audio()
    session.update_partial_transcript("你好")
    session.finalize_transcript("你好鸿途")
    session.append_reply_delta("我在。")
    session.start_speaking()
    session.complete()

    snapshot = session.snapshot()
    assert snapshot["phase"] == "completed"
    assert snapshot["transcript_final"] == "你好鸿途"
    assert snapshot["reply_text"] == "我在。"
    assert snapshot["latency_ms"] == {
        "audio_detect": 100.0,
        "first_partial_asr": 250.0,
        "final_asr": 800.0,
        "first_reply_token": 1100.0,
        "first_speech": 1300.0,
        "total": 2000.0,
    }


def test_realtime_voice_session_records_barge_in() -> None:
    from eibrain.body.realtime_voice import RealtimeVoiceSession

    ticks = iter([0.0, 0.2, 0.4])
    session = RealtimeVoiceSession(session_id="s1", actor_id="darrow", clock=lambda: next(ticks))

    session.start_listening()
    session.start_speaking()
    session.interrupt(reason="user_started_speaking")

    assert session.phase == "barge_in"
    assert session.interrupted is True
    assert session.interrupt_reason == "user_started_speaking"
