from __future__ import annotations

import json

import pytest


def test_summarize_voice_chain_reports_counts_stats_thresholds_leaks_and_bottleneck() -> None:
    from apps.body_runtime.voice_chain_benchmark import summarize_voice_chain

    turns = [
        {
            "wakeToListenMs": 100.0,
            "asrFinalMs": 600.0,
            "firstTokenMs": 300.0,
            "firstAudioMs": 1400.0,
            "interruptStopMs": 200.0,
            "roundLeak": False,
        },
        {
            "wakeToListenMs": 200.0,
            "asrFinalMs": 900.0,
            "firstTokenMs": 500.0,
            "firstAudioMs": 2600.0,
            "interruptStopMs": 250.0,
            "roundLeak": True,
        },
        {
            "wakeToListenMs": 300.0,
            "asrFinalMs": 1200.0,
            "firstTokenMs": 800.0,
            "firstAudioMs": 3100.0,
            "interruptStopMs": 450.0,
            "roundLeak": False,
        },
    ]

    summary = summarize_voice_chain(
        turns,
        thresholds={
            "asrFinalMs": 1000.0,
            "firstTokenMs": 900.0,
            "firstAudioMs": 2500.0,
            "interruptStopMs": 500.0,
        },
    )

    assert summary["turnCount"] == 3
    assert summary["roundLeakCount"] == 1
    assert summary["roundLeakRate"] == pytest.approx(1 / 3)
    assert summary["metrics"]["wakeToListenMs"] == {
        "count": 3,
        "avg": 200.0,
        "p95": 300.0,
        "threshold": None,
        "pass": None,
    }
    assert summary["metrics"]["firstAudioMs"] == {
        "count": 3,
        "avg": pytest.approx(2366.6666666666665),
        "p95": 3100.0,
        "threshold": 2500.0,
        "pass": False,
    }
    assert summary["metrics"]["firstTokenMs"]["pass"] is True
    assert summary["metrics"]["interruptStopMs"]["pass"] is True
    assert summary["bottleneck"] == {
        "field": "firstAudioMs",
        "label": "first_audio",
        "p95": 3100.0,
        "threshold": 2500.0,
        "ratio": pytest.approx(1.24),
    }


def test_summarize_voice_chain_uses_nearest_rank_p95() -> None:
    from apps.body_runtime.voice_chain_benchmark import summarize_voice_chain

    turns = [{"asrFinalMs": float(value)} for value in range(1, 21)]

    summary = summarize_voice_chain(turns, thresholds={"asrFinalMs": 100.0})

    assert summary["metrics"]["asrFinalMs"]["count"] == 20
    assert summary["metrics"]["asrFinalMs"]["avg"] == 10.5
    assert summary["metrics"]["asrFinalMs"]["p95"] == 19.0


def test_summarize_voice_chain_ignores_missing_and_non_numeric_fields() -> None:
    from apps.body_runtime.voice_chain_benchmark import summarize_voice_chain

    turns = [
        {"asrFinalMs": 500.0, "firstTokenMs": None, "firstAudioMs": "slow"},
        {"firstTokenMs": 420.0, "interruptStopMs": True},
        {"wakeToListenMs": 50},
    ]

    summary = summarize_voice_chain(turns)

    assert summary["metrics"]["asrFinalMs"]["count"] == 1
    assert summary["metrics"]["firstTokenMs"]["count"] == 1
    assert summary["metrics"]["wakeToListenMs"]["count"] == 1
    assert "firstAudioMs" not in summary["metrics"]
    assert "interruptStopMs" not in summary["metrics"]


def test_summarize_voice_chain_uses_default_thresholds_and_is_json_serializable() -> None:
    from apps.body_runtime.voice_chain_benchmark import DEFAULT_THRESHOLDS, summarize_voice_chain

    assert DEFAULT_THRESHOLDS == {
        "asrFinalMs": 800.0,
        "firstTokenMs": 700.0,
        "firstAudioMs": 2000.0,
        "interruptStopMs": 300.0,
    }

    summary = summarize_voice_chain(
        [
            {"asrFinalMs": 799.0, "firstTokenMs": 701.0},
            {"asrFinalMs": 800.0, "firstTokenMs": 650.0},
        ]
    )

    assert summary["metrics"]["asrFinalMs"]["pass"] is True
    assert summary["metrics"]["firstTokenMs"]["pass"] is False
    assert json.loads(json.dumps(summary, sort_keys=True)) == summary


def test_summarize_voice_chain_overlays_custom_thresholds_on_defaults() -> None:
    from apps.body_runtime.voice_chain_benchmark import summarize_voice_chain

    summary = summarize_voice_chain(
        [
            {
                "asrFinalMs": 900.0,
                "firstTokenMs": 650.0,
                "firstAudioMs": 1900.0,
                "interruptStopMs": 250.0,
                "interrupted": True,
                "streamingReady": True,
            }
        ],
        thresholds={"asrFinalMs": 1000.0},
    )

    assert summary["thresholds"] == {
        "asrFinalMs": 1000.0,
        "firstTokenMs": 700.0,
        "firstAudioMs": 2000.0,
        "interruptStopMs": 300.0,
    }
    assert summary["metrics"]["asrFinalMs"]["threshold"] == 1000.0
    assert summary["metrics"]["firstTokenMs"]["threshold"] == 700.0
    assert summary["metrics"]["firstAudioMs"]["threshold"] == 2000.0
    assert summary["metrics"]["interruptStopMs"]["threshold"] == 300.0
    assert summary["readinessSummary"]["honjiaReady"] is True


def test_summarize_voice_chain_reports_round_stage_and_readiness_summary() -> None:
    from apps.body_runtime.voice_chain_benchmark import summarize_voice_chain

    summary = summarize_voice_chain(
        [
            {
                "roundId": "round-1",
                "status": "reply_ready",
                "asrFinalMs": 500.0,
                "firstTokenMs": 260.0,
                "firstAudioMs": 1200.0,
                "stageLatencyMs": {
                    "listen_asr": 500.0,
                    "llm_first_token": 260.0,
                    "tts_first_audio": 440.0,
                    "total": 1200.0,
                },
                "streaming": {
                    "asrPartial": True,
                    "asrFinal": True,
                    "llmDelta": True,
                    "ttsChunk": True,
                    "playback": True,
                },
                "roundLeak": False,
            },
            {
                "roundId": "round-2",
                "status": "interrupted",
                "asrFinalMs": 700.0,
                "firstTokenMs": 320.0,
                "firstAudioMs": 1600.0,
                "interruptStopMs": 180.0,
                "stageLatencyMs": {
                    "listen_asr": 700.0,
                    "llm_first_token": 320.0,
                    "tts_first_audio": 580.0,
                    "total": 1600.0,
                },
                "streaming": {
                    "asrPartial": True,
                    "asrFinal": True,
                    "llmDelta": True,
                    "ttsChunk": True,
                    "playback": True,
                },
                "interrupted": True,
                "roundLeak": False,
            },
            {
                "roundId": "round-3",
                "status": "stale_round_blocked",
                "asrFinalMs": 650.0,
                "firstTokenMs": 300.0,
                "firstAudioMs": 1500.0,
                "stageLatencyMs": {
                    "listen_asr": 650.0,
                    "llm_first_token": 300.0,
                    "tts_first_audio": 550.0,
                    "total": 1500.0,
                },
                "streaming": {
                    "asrPartial": True,
                    "asrFinal": True,
                    "llmDelta": True,
                    "ttsChunk": True,
                    "playback": True,
                },
                "roundLeak": True,
            },
        ],
        thresholds={"asrFinalMs": 800.0, "firstTokenMs": 700.0, "firstAudioMs": 2000.0, "interruptStopMs": 300.0},
    )

    assert summary["rounds"][0]["roundId"] == "round-1"
    assert summary["rounds"][0]["stageLatencyMs"]["listen_asr"] == 500.0
    assert summary["rounds"][1]["interrupted"] is True
    assert summary["rounds"][1]["interruptStopMs"] == 180.0
    assert summary["rounds"][2]["roundLeak"] is True
    assert summary["stageLatencyMetrics"]["listen_asr"]["avg"] == pytest.approx(616.6666666666666)
    assert summary["stageLatencyMetrics"]["listen_asr"]["p95"] == 700.0
    assert summary["stageLatencyMetrics"]["total"]["p95"] == 1600.0
    assert summary["roundLeak"] == {"count": 1, "rate": pytest.approx(1 / 3), "free": False, "pass": False}
    assert summary["interruptStop"]["requiredCount"] == 1
    assert summary["interruptStop"]["confirmedCount"] == 1
    assert summary["interruptStop"]["ready"] is True
    assert summary["streaming"]["ready"] is True
    assert summary["streaming"]["readyTurnCount"] == 3
    assert summary["readinessSummary"]["honjiaReady"] is False
    assert summary["readinessSummary"]["roundLeakFree"] is False
    assert summary["readinessSummary"]["interruptStopReady"] is True
    assert summary["readinessSummary"]["streamingReady"] is True
    assert "round leak" in summary["readinessSummary"]["readinessMessage"]
    assert json.loads(json.dumps(summary, sort_keys=True)) == summary


def test_summarize_voice_chain_treats_non_empty_streaming_payloads_as_present() -> None:
    from apps.body_runtime.voice_chain_benchmark import summarize_voice_chain

    summary = summarize_voice_chain(
        [
            {
                "roundId": "round-streaming-text",
                "asrFinalMs": 500.0,
                "firstTokenMs": 260.0,
                "firstAudioMs": 1200.0,
                "interruptStopMs": 120.0,
                "status": "interrupted",
                "streaming": {
                    "asrPartial": "ni hao",
                    "asrFinal": "ni hao",
                    "replyDelta": "hello",
                    "ttsChunk": "chunk-001",
                    "playback": "started",
                },
            }
        ]
    )

    assert summary["streaming"]["ready"] is True
    assert summary["rounds"][0]["streamingReady"] is True
    assert summary["readinessSummary"]["streamingReady"] is True


def test_summarize_voice_chain_reports_real_streaming_stage_metrics_and_playback_signal() -> None:
    from apps.body_runtime.voice_chain_benchmark import summarize_voice_chain

    summary = summarize_voice_chain(
        [
            {
                "roundId": "round-streaming-sample",
                "status": "reply_ready",
                "firstAsrPartialMs": 120.0,
                "asrFinalMs": 460.0,
                "firstLlmDeltaMs": 640.0,
                "firstTokenMs": 640.0,
                "firstTtsChunkMs": 910.0,
                "firstAudioMs": 1040.0,
                "streaming": {
                    "asrPartial": True,
                    "asrFinal": True,
                    "llmDelta": True,
                    "ttsChunk": True,
                    "playback": True,
                },
            }
        ]
    )

    assert summary["metrics"]["firstAsrPartialMs"]["avg"] == 120.0
    assert summary["metrics"]["asrFinalMs"]["avg"] == 460.0
    assert summary["metrics"]["firstLlmDeltaMs"]["avg"] == 640.0
    assert summary["metrics"]["firstTokenMs"]["avg"] == 640.0
    assert summary["metrics"]["firstTtsChunkMs"]["avg"] == 910.0
    assert summary["metrics"]["firstAudioMs"]["avg"] == 1040.0
    assert summary["rounds"][0]["streamingReady"] is True
    assert summary["streaming"]["ready"] is True


def test_summarize_voice_chain_requires_asr_final_and_playback_evidence_for_streaming_readiness() -> None:
    from apps.body_runtime.voice_chain_benchmark import summarize_voice_chain

    summary = summarize_voice_chain(
        [
            {
                "roundId": "round-missing-playback",
                "status": "reply_ready",
                "firstAsrPartialMs": 120.0,
                "asrFinalMs": 460.0,
                "firstLlmDeltaMs": 640.0,
                "firstTtsChunkMs": 910.0,
                "streaming": {
                    "asrPartial": True,
                    "llmDelta": True,
                    "ttsChunk": True,
                },
            }
        ]
    )

    assert summary["rounds"][0]["streamingReady"] is False
    assert summary["rounds"][0]["streamingMissingSignals"] == ["playback"]
    assert summary["streaming"]["ready"] is False


def test_summarize_joyinside_voice_readiness_is_ready_when_acceptance_signals_pass() -> None:
    from apps.body_runtime.voice_chain_benchmark import summarize_joyinside_voice_readiness

    readiness = summarize_joyinside_voice_readiness(
        [
            {
                "roundId": "round-1",
                "asrFinalMs": 520.0,
                "firstTokenMs": 280.0,
                "firstAudioMs": 1300.0,
                    "streaming": {
                        "asrPartial": True,
                        "asrFinal": True,
                        "llmDelta": True,
                        "ttsChunk": True,
                        "playback": True,
                    },
                "roundLeak": False,
            },
            {
                "roundId": "round-2",
                "status": "interrupted",
                "interrupted": True,
                "asrFinalMs": 610.0,
                "firstTokenMs": 310.0,
                "firstAudioMs": 1500.0,
                "interruptStopMs": 180.0,
                    "streaming": {
                        "asrPartial": True,
                        "asrFinal": True,
                        "llmDelta": True,
                        "ttsChunk": True,
                        "playback": True,
                    },
                "roundLeak": False,
            },
        ],
        thresholds={"asrFinalMs": 800.0, "firstTokenMs": 700.0, "firstAudioMs": 2000.0, "interruptStopMs": 300.0},
    )

    assert readiness["ready"] is True
    assert readiness["score"] == 100
    assert readiness["grade"] == "A"
    assert readiness["blocking_reasons"] == []
    assert readiness["next_actions"] == []


def test_summarize_joyinside_voice_readiness_reports_actionable_blocking_reasons() -> None:
    from apps.body_runtime.voice_chain_benchmark import summarize_joyinside_voice_readiness

    readiness = summarize_joyinside_voice_readiness(
        [
            {
                "roundId": "round-1",
                "asrFinalMs": 520.0,
                "firstTokenMs": 280.0,
                "firstAudioMs": 2500.0,
                "streaming": {"asrPartial": True, "llmDelta": False, "ttsChunk": True},
                "roundLeak": True,
            }
        ],
        thresholds={"asrFinalMs": 800.0, "firstTokenMs": 700.0, "firstAudioMs": 2000.0, "interruptStopMs": 300.0},
    )

    assert readiness["ready"] is False
    assert readiness["score"] == 0
    assert readiness["grade"] == "F"
    assert readiness["blocking_reasons"] == [
        "firstAudioMs_p95_exceeded",
        "interrupt_not_confirmed",
        "round_leak_detected",
        "streaming_signals_missing",
    ]
    assert readiness["next_actions"] == [
        "Reduce firstAudioMs p95 to <= 2000.0ms.",
        "Capture at least one interrupted turn with interruptStopMs <= 300.0ms.",
        "Fix stale round suppression until roundLeak count is 0.",
            "Emit ASR partial/final, LLM delta, TTS chunk, and playback streaming signals for every turn.",
    ]


def test_summarize_joyinside_voice_readiness_does_not_report_leak_without_turns() -> None:
    from apps.body_runtime.voice_chain_benchmark import summarize_joyinside_voice_readiness

    readiness = summarize_joyinside_voice_readiness([])

    assert readiness["ready"] is False
    assert "no_turns" in readiness["blocking_reasons"]
    assert "round_leak_detected" not in readiness["blocking_reasons"]
