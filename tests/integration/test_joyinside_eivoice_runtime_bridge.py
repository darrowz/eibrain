from __future__ import annotations


def test_joyinside_asr_event_flows_into_realtime_voice_bridge() -> None:
    from eibrain.cognition.realtime import VoiceRuntimeBridge
    from eibrain.protocol.joyinside_voice import normalize_voice_event

    bridge = VoiceRuntimeBridge(clock=_clock())
    normalized = normalize_voice_event(
        {
            "uid": "darrow",
            "mid": "msg-1",
            "contentType": "ASR",
            "content": {
                "eventType": "ASR",
                "text": "你好鸿途",
                "textType": "IS_FINAL",
            },
        },
        direction="downstream",
    )

    result = bridge.handle_event(normalized)

    assert normalized["eiprotocolName"] == "ei.voice.asr.final"
    assert result["lane"] == "slow_reasoning"
    assert result["blackboardPatch"]["asrFinal"] == "你好鸿途"
    assert result["trace"]["eventName"] == normalized["eiprotocolName"]
    assert result["trace"]["roundId"] == result["roundId"]
    assert result["trace"]["cancellationToken"] == result["blackboardPatch"]["slowReasonerInput"]["cancellationToken"]
    assert result["trace"]["source"] == "voice_runtime_bridge"


def test_joyinside_tts_chunk_flows_into_speaking_lane() -> None:
    from eibrain.cognition.realtime import VoiceRuntimeBridge
    from eibrain.protocol.joyinside_voice import normalize_voice_event

    bridge = VoiceRuntimeBridge(clock=_clock())
    active = bridge.handle_event({"type": "ASR_FINAL", "payload": {"text": "介绍一下你自己"}})
    normalized = normalize_voice_event(
        {
            "uid": "darrow",
            "mid": "msg-tts-1",
            "contentType": "TTS",
            "content": {
                "eventType": "TTS",
                "audioBase64": "AQID",
                "index": 4,
            },
        },
        direction="downstream",
    )

    result = bridge.handle_event(normalized)

    assert normalized["eiprotocolName"] == "ei.voice.tts.chunk"
    assert result["lane"] == "speaking"
    assert result["speechPlan"]["phase"] == "chunk"
    assert result["speechPlan"]["chunkIndex"] == 4
    assert result["speechPlan"]["metadata"]["audioBase64"] == "AQID"
    assert result["trace"]["eventName"] == normalized["eiprotocolName"]
    assert result["trace"]["roundId"] == result["roundId"]
    assert result["trace"]["cancellationToken"] == active["trace"]["cancellationToken"]


def test_joyinside_interrupt_event_cancels_active_voice_round() -> None:
    from eibrain.cognition.realtime import VoiceRuntimeBridge
    from eibrain.protocol.joyinside_voice import interrupt, normalize_voice_event

    bridge = VoiceRuntimeBridge(clock=_clock())
    partial = bridge.handle_event({"type": "ASR_PARTIAL", "payload": {"text": "介绍一下"}})
    normalized = normalize_voice_event(interrupt(uid="darrow", mid="msg-2", reason="barge_in"), direction="upstream")

    result = bridge.handle_event(normalized)

    assert normalized["eiprotocolName"] == "ei.dialogue.interrupt.requested"
    assert result["lane"] == "interrupt"
    assert result["interrupt"]["cancelOldRound"] is True
    assert result["interrupt"]["oldRound"]["roundId"] == partial["roundId"]
    assert result["interrupt"]["cancellationChain"] == [
        "tts_playback",
        "llm_generation",
        "action_plan",
        "memory_prefetch",
    ]
    assert result["trace"]["eventName"] == normalized["eiprotocolName"]
    assert result["trace"]["roundId"] == result["interrupt"]["newRound"]["roundId"]
    assert result["trace"]["cancellationToken"] == result["interrupt"]["newRound"]["cancellationToken"]


def test_joyinside_activity_event_preserves_payload_for_proactive_lane() -> None:
    from eibrain.cognition.realtime import VoiceRuntimeBridge
    from eibrain.protocol.joyinside_voice import normalize_voice_event

    bridge = VoiceRuntimeBridge(clock=_clock())
    normalized = normalize_voice_event(
        {
            "uid": "darrow",
            "mid": "msg-activity-1",
            "contentType": "ACTIVITY",
            "content": {
                "eventType": "ACTIVITY",
                "idleSeconds": 180,
                "emotion": {"mood": "sad"},
                "memoryCandidates": [{"id": "mem-1", "text": "喝水提醒", "importance": 0.9}],
            },
        },
        direction="downstream",
    )

    result = bridge.handle_event(normalized)

    assert normalized["payload"]["idleSeconds"] == 180
    assert result["lane"] == "proactive_activity"
    assert result["actions"][0]["proposal"]["should_emit"] is True
    assert result["memoryHints"][0]["candidate"]["id"] == "mem-1"


def test_joyinside_queue_health_preserves_payload_for_runtime_health_lane() -> None:
    from eibrain.cognition.realtime import VoiceRuntimeBridge
    from eibrain.protocol.joyinside_voice import normalize_voice_event

    bridge = VoiceRuntimeBridge(clock=_clock())
    normalized = normalize_voice_event(
        {
            "uid": "darrow",
            "mid": "msg-health-1",
            "contentType": "QUEUE_HEALTH",
            "content": {
                "eventType": "QUEUE_HEALTH",
                "status": "degraded",
                "queue": "ws_send_queue",
                "latencyMs": 850,
            },
        },
        direction="downstream",
    )

    result = bridge.handle_event(normalized)

    assert normalized["payload"]["status"] == "degraded"
    assert result["lane"] == "runtime_health"
    assert result["conversationState"] == "degraded"
    assert result["blackboardPatch"]["warnings"][0]["queue"] == "ws_send_queue"
    assert result["actions"][0]["pauseProactiveInteraction"] is True
    assert result["trace"]["eventName"] == normalized["eiprotocolName"]
    assert result["trace"]["lane"] == "runtime_health"
    assert result["trace"]["source"] == "voice_runtime_bridge"


def test_joyinside_stale_tts_chunk_is_marked_stale() -> None:
    from eibrain.cognition.realtime import VoiceRuntimeBridge
    from eibrain.protocol.joyinside_voice import normalize_voice_event

    bridge = VoiceRuntimeBridge(clock=_clock())
    active = bridge.handle_event({"type": "ASR_FINAL", "payload": {"text": "介绍一下你自己"}})
    bridge.handle_event({"type": "CLIENT_INTERRUPT", "payload": {"reason": "barge_in"}})
    normalized = normalize_voice_event(
        {
            "uid": "darrow",
            "mid": "msg-tts-stale-1",
            "contentType": "TTS",
            "content": {
                "eventType": "TTS",
                "audioBase64": "AQID",
                "index": 1,
                "roundId": active["roundId"],
                "cancellationToken": active["trace"]["cancellationToken"],
            },
        },
        direction="downstream",
    )

    result = bridge.handle_event(normalized)

    assert result["actions"] == []
    assert result["speechPlan"] is None
    assert result["blackboardPatch"]["ttsPlayback"]["stale"] is True
    assert result["trace"]["eventName"] == normalized["eiprotocolName"]
    assert result["trace"]["roundId"] == active["roundId"]
    assert result["trace"]["cancellationToken"] == active["trace"]["cancellationToken"]


def test_joyinside_untagged_tts_after_interrupt_is_stale_not_rebound_to_new_round() -> None:
    from eibrain.cognition.realtime import VoiceRuntimeBridge
    from eibrain.protocol.joyinside_voice import normalize_voice_event

    bridge = VoiceRuntimeBridge(clock=_clock())
    bridge.handle_event({"type": "ASR_FINAL", "payload": {"text": "介绍一下你自己"}})
    interrupt = bridge.handle_event({"type": "CLIENT_INTERRUPT", "payload": {"reason": "barge_in"}})
    normalized = normalize_voice_event(
        {
            "uid": "darrow",
            "mid": "msg-tts-untagged-stale",
            "contentType": "TTS",
            "content": {
                "eventType": "TTS",
                "audioBase64": "AQID",
                "index": 2,
            },
        },
        direction="downstream",
    )

    result = bridge.handle_event(normalized)

    assert result["lane"] == "speaking"
    assert result["actions"] == []
    assert result["speechPlan"] is None
    assert result["blackboardPatch"]["ttsPlayback"]["stale"] is True
    assert result["blackboardPatch"]["ttsPlayback"]["staleReason"] == "untagged_tts_without_active_final"
    assert result["trace"]["roundId"] == interrupt["interrupt"]["newRound"]["roundId"]
    assert result["trace"]["cancellationToken"] == interrupt["interrupt"]["newRound"]["cancellationToken"]


def test_eivoice_runtime_status_feeds_monitoring_panel() -> None:
    from eihead.eivoice_runtime import AudioFrame, EiVoiceRuntimeCore
    from eihead.monitoring.eivoice_runtime import build_eivoice_runtime_panel

    runtime = EiVoiceRuntimeCore()
    runtime.state_machine.wake_detected()
    for sequence in range(27):
        runtime.ws_send_queue.push(AudioFrame(payload=b"x", sequence=sequence))

    panel = build_eivoice_runtime_panel(runtime.status())

    assert panel["state"] == "conversation"
    assert panel["queues"]["ws_send_queue"]["capacity"] == 25
    assert panel["queues"]["ws_send_queue"]["policy"] == "drop_oldest"
    assert panel["droppedTotal"] == 2
    assert panel["health"] == "degraded"
    assert panel["conversationState"] == "conversation"
    assert panel["wakeword"]["capacity_ms"] == 1500
    assert any("audio frontend readiness is missing" in warning for warning in panel["warnings"])


def _clock():
    values = iter(
        [
            7000.0,
            7000.01,
            7000.02,
            7000.03,
            7000.04,
            7000.05,
            7000.06,
            7000.07,
            7000.08,
            7000.09,
            7000.1,
            7000.11,
        ]
    )

    def clock() -> float:
        return next(values)

    return clock
