from __future__ import annotations

from eibrain.cognition.realtime import VoiceRuntimeBridge


def _clock():
    values = iter(
        [
            5000.0,
            5000.05,
            5000.1,
            5000.15,
            5000.2,
            5000.25,
            5000.3,
            5000.35,
            5000.4,
            5000.45,
            5000.5,
            5000.55,
        ]
    )

    def clock() -> float:
        return next(values)

    return clock


def test_partial_asr_routes_to_fast_think_without_stable_decision() -> None:
    bridge = VoiceRuntimeBridge(clock=_clock())

    result = bridge.handle_event(
        {
            "type": "ASR_PARTIAL",
            "roundId": "voice-round-1",
            "payload": {"text": "帮我记一下"},
        }
    )

    assert result["roundId"]
    assert result["conversationState"] == "listening"
    assert result["lane"] == "fast_think"
    assert result["blackboardPatch"]["asrPartial"] == "帮我记一下"
    assert result["actions"] == []
    assert result["speechPlan"] is None
    assert result["interrupt"] is None
    assert result["memoryHints"][0]["kind"] == "prefetch"
    assert result["memoryHints"][0]["stable"] is False
    assert result["blackboardPatch"]["fastHypothesis"]["stable"] is False
    assert "decision" not in result["blackboardPatch"]["fastHypothesis"]


def test_final_asr_routes_to_slow_reasoning_with_reasoner_input() -> None:
    bridge = VoiceRuntimeBridge(clock=_clock())
    bridge.handle_event({"type": "ASR_PARTIAL", "text": "请打开"})

    result = bridge.handle_event(
        {
            "eventType": "ASR_FINAL",
            "payload": {"finalText": "请打开客厅灯"},
        }
    )

    assert result["conversationState"] == "reasoning"
    assert result["lane"] == "slow_reasoning"
    assert result["blackboardPatch"]["asrFinal"] == "请打开客厅灯"
    assert result["blackboardPatch"]["slowReasonerInput"]["finalText"] == "请打开客厅灯"
    assert result["blackboardPatch"]["slowReasonerInput"]["roundId"] == result["roundId"]
    assert result["speechPlan"] is None


def test_tts_sentence_and_chunk_route_to_speaking_plan() -> None:
    bridge = VoiceRuntimeBridge(clock=_clock())

    sentence = bridge.handle_event(
        {
            "type": "TTS_SENTENCE_START",
            "roundId": "speech-round",
            "payload": {"sentenceId": "s1", "text": "好的，我来处理。"},
        }
    )
    chunk = bridge.handle_event(
        {
            "type": "TTS_CHUNK",
            "roundId": "speech-round",
            "payload": {"sentenceId": "s1", "chunkIndex": 2, "audioMs": 120},
        }
    )

    assert sentence["lane"] == "speaking"
    assert sentence["conversationState"] == "speaking"
    assert sentence["speechPlan"]["phase"] == "sentence_start"
    assert sentence["speechPlan"]["sentenceId"] == "s1"
    assert sentence["speechPlan"]["text"] == "好的，我来处理。"
    assert chunk["lane"] == "speaking"
    assert chunk["speechPlan"]["phase"] == "chunk"
    assert chunk["speechPlan"]["chunkIndex"] == 2
    assert chunk["speechPlan"]["metadata"]["audioMs"] == 120


def test_interrupt_cancels_old_round_and_starts_new_round() -> None:
    bridge = VoiceRuntimeBridge(clock=_clock())
    partial = bridge.handle_event({"type": "ASR_PARTIAL", "text": "请打开客厅灯"})

    result = bridge.handle_event(
        {
            "type": "CLIENT_INTERRUPT",
            "payload": {"reason": "barge_in"},
        }
    )

    assert result["lane"] == "interrupt"
    assert result["conversationState"] == "interrupted"
    assert result["interrupt"]["applied"] is True
    assert result["interrupt"]["cancelOldRound"] is True
    assert result["interrupt"]["oldRound"]["roundId"] == partial["roundId"]
    assert result["interrupt"]["oldRound"]["stale"] is True
    assert result["interrupt"]["newRound"]["roundId"] != partial["roundId"]
    assert result["roundId"] == result["interrupt"]["newRound"]["roundId"]


def test_activity_event_proposes_interruptible_proactive_interaction() -> None:
    bridge = VoiceRuntimeBridge(clock=_clock())

    result = bridge.handle_event(
        {
            "type": "ACTIVITY",
            "payload": {
                "idleSeconds": 180,
                "emotion": {"mood": "sad"},
                "memoryCandidates": [{"id": "mem-1", "text": "喝水提醒", "importance": 0.9}],
            },
        }
    )

    assert result["lane"] == "proactive_activity"
    assert result["conversationState"] == "proactive"
    assert result["actions"][0]["type"] == "proactive_activity"
    assert result["actions"][0]["interruptible"] is True
    assert result["actions"][0]["proposal"]["should_emit"] is True


def test_queue_health_degraded_emits_warnings_and_density_recommendation() -> None:
    bridge = VoiceRuntimeBridge(clock=_clock())

    result = bridge.handle_event(
        {
            "eventType": "QUEUE_HEALTH",
            "payload": {"status": "degraded", "queue": "tts", "latencyMs": 850},
        }
    )

    assert result["lane"] == "runtime_health"
    assert result["conversationState"] == "degraded"
    assert result["blackboardPatch"]["warnings"]
    assert "降低 verbal density" in result["blackboardPatch"]["warnings"][0]["recommendation"]
    assert result["actions"][0]["type"] == "runtime_backpressure"
    assert result["actions"][0]["pauseProactiveInteraction"] is True
