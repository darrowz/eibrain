from __future__ import annotations

import json

import pytest


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


def test_realtime_voice_session_exposes_round_token_and_json_events() -> None:
    from eibrain.body.realtime_voice import RealtimeVoiceSession

    ticks = iter([0.0, 0.05, 0.1, 0.2, 0.4, 0.7])
    session = RealtimeVoiceSession(
        session_id="s1",
        actor_id="darrow",
        clock=lambda: next(ticks),
        round_id="round-voice-1",
        cancellation_token="tok-voice-1",
    )

    session.start_listening()
    session.update_partial_transcript("你好")
    session.update_microfeedback("我在听。")
    session.append_reply_delta("我理解了。")
    session.start_speaking()
    session.complete()

    snapshot = session.snapshot()
    lanes = [event["lane"] for event in snapshot["events"]]
    event_types = [event["event_type"] for event in snapshot["events"]]

    assert snapshot["round_id"] == "round-voice-1"
    assert snapshot["roundId"] == "round-voice-1"
    assert snapshot["cancellation_token"] == "tok-voice-1"
    assert snapshot["cancellationToken"] == "tok-voice-1"
    assert lanes == ["listening", "listening", "fast_think", "slow_thinking", "speaking", "complete"]
    assert event_types == ["listening_started", "asr_partial", "microfeedback", "agent_think", "tts_started", "complete"]
    assert snapshot["events"][1]["roundId"] == "round-voice-1"
    assert snapshot["events"][2]["payload"] == {"text": "我在听。"}
    assert json.loads(json.dumps(snapshot, ensure_ascii=False))["events"][3]["lane"] == "slow_thinking"


def test_realtime_voice_session_generates_stable_nonempty_round_token() -> None:
    from eibrain.body.realtime_voice import RealtimeVoiceSession

    session = RealtimeVoiceSession(session_id="s1", actor_id="darrow")

    assert session.round_id
    assert session.cancellation_token
    assert session.snapshot()["round_id"] == session.round_id
    assert session.snapshot()["cancellation_token"] == session.cancellation_token


def test_realtime_voice_interrupt_records_complete_cancellation_chain() -> None:
    from eibrain.body.realtime_voice import RealtimeVoiceSession

    ticks = iter([0.0, 0.1, 0.2])
    session = RealtimeVoiceSession(
        session_id="s1",
        actor_id="darrow",
        clock=lambda: next(ticks),
        round_id="round-voice-1",
        cancellation_token="tok-voice-1",
    )

    session.start_listening()
    session.start_speaking()
    session.interrupt(reason="user_started_speaking")

    snapshot = session.snapshot()
    chain_targets = [item["target"] for item in snapshot["cancellation_chain"]]
    interrupt_events = [event for event in snapshot["events"] if event["lane"] == "interrupt"]

    assert session.generation_cancelled is True
    assert session.tts_stopped is True
    assert session.action_plan_cancelled is True
    assert chain_targets == ["generation", "tts", "action_plan"]
    assert [event["event_type"] for event in interrupt_events] == [
        "interrupt",
        "generation_cancelled",
        "tts_stopped",
        "action_plan_cancelled",
    ]
    assert all(item["round_id"] == "round-voice-1" for item in snapshot["cancellation_chain"])
    assert all(item["cancellation_token"] == "tok-voice-1" for item in snapshot["cancellation_chain"])


def test_realtime_voice_rejects_stale_round_token_after_interrupt() -> None:
    from eibrain.body.realtime_voice import RealtimeVoiceSession

    ticks = iter([0.0, 0.1, 0.2])
    session = RealtimeVoiceSession(
        session_id="s1",
        actor_id="darrow",
        clock=lambda: next(ticks),
        round_id="round-voice-1",
        cancellation_token="tok-voice-1",
    )

    session.start_listening()
    assert session.is_current(round_id="round-voice-1", cancellation_token="tok-voice-1") is True

    session.interrupt(reason="user_started_speaking")
    event_count = len(session.events)

    assert session.is_current(round_id="round-voice-1", cancellation_token="tok-voice-1") is False
    with pytest.raises(RuntimeError, match="not current"):
        session.update_partial_transcript(
            "late asr",
            round_id="round-voice-1",
            cancellation_token="tok-voice-1",
        )
    assert len(session.events) == event_count


def test_realtime_voice_microfeedback_latency_is_additive() -> None:
    from eibrain.body.realtime_voice import RealtimeVoiceSession

    ticks = iter([0.0, 0.12])
    session = RealtimeVoiceSession(session_id="s1", actor_id="darrow", clock=lambda: next(ticks))

    session.start_listening()
    session.update_microfeedback("我先想一下。")

    snapshot = session.snapshot()
    assert snapshot["microfeedback"] == "我先想一下。"
    assert snapshot["latency_ms"] == {"first_microfeedback": 120.0}
