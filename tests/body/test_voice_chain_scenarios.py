from __future__ import annotations

import json


def test_run_voice_chain_scenarios_reports_honjia_ready_summary() -> None:
    from apps.body_runtime.voice_chain_scenarios import run_voice_chain_scenarios

    report = run_voice_chain_scenarios()

    assert report["schema"] == "eibrain.voice_chain_scenarios.v1"
    assert report["scenarioCount"] >= 5
    assert report["turnCount"] >= 5
    assert report["honjiaReady"] is True
    assert report["streamingReady"] is True
    assert report["interruptStopReady"] is True
    assert report["readinessSummary"]["honjiaReady"] is True
    assert report["readinessSummary"]["streamingReady"] is True
    assert report["readinessSummary"]["interruptStopReady"] is True
    assert report["summary"]["roundLeakCount"] == 0
    assert {
        "firstAsrPartialMs",
        "asrFinalMs",
        "firstLlmDeltaMs",
        "firstTokenMs",
        "firstTtsChunkMs",
        "firstAudioMs",
        "interruptStopMs",
    } <= set(report["summary"]["metrics"])
    assert "listen_asr" in report["summary"]["stageLatencyMetrics"]
    assert report["summary"]["rounds"][0]["stageLatencyMs"]["listen_asr"] > 0
    assert report["summary"]["rounds"][0]["streamingReady"] is True
    assert {scenario["name"] for scenario in report["scenarios"]} >= {
        "short_chinese",
        "child_fuzzy",
        "playback_barge_in",
        "follow_up_turn",
        "network_jitter",
    }
    assert json.loads(json.dumps(report, sort_keys=True)) == report


def test_run_voice_chain_scenarios_marks_not_ready_when_threshold_fails() -> None:
    from apps.body_runtime.voice_chain_scenarios import run_voice_chain_scenarios

    report = run_voice_chain_scenarios(thresholds={"firstAudioMs": 500.0})

    assert report["honjiaReady"] is False
    assert report["summary"]["metrics"]["asrFinalMs"]["threshold"] == 800.0
    assert report["summary"]["metrics"]["firstTokenMs"]["threshold"] == 700.0
    assert report["summary"]["metrics"]["firstAudioMs"]["pass"] is False
    assert report["failedMetrics"] == ["firstAudioMs"]
    assert report["readinessSummary"]["honjiaReady"] is False
    assert report["readinessSummary"]["failedMetrics"] == ["firstAudioMs"]


def test_run_voice_chain_scenarios_does_not_treat_empty_explicit_input_as_ready() -> None:
    from apps.body_runtime.voice_chain_scenarios import run_voice_chain_scenarios

    report = run_voice_chain_scenarios(scenarios=[])

    assert report["scenarioCount"] == 0
    assert report["turnCount"] == 0
    assert report["roundLeakFree"] is True
    assert report["honjiaReady"] is False
    assert report["streamingReady"] is False
    assert report["interruptStopReady"] is False
    assert report["scenarios"] == []


def test_run_voice_chain_scenarios_accepts_custom_turns() -> None:
    from apps.body_runtime.voice_chain_scenarios import VoiceScenario, run_voice_chain_scenarios

    report = run_voice_chain_scenarios(
        scenarios=[
            VoiceScenario(
                name="custom_round_leak",
                description="stale round regression",
                turns=[
                    {
                        "roundId": "rnd-1",
                        "asrFinalMs": 100.0,
                        "firstTokenMs": 100.0,
                        "firstAudioMs": 300.0,
                        "roundLeak": True,
                    }
                ],
            )
        ]
    )

    assert report["scenarioCount"] == 1
    assert report["turnCount"] == 1
    assert report["roundLeakFree"] is False
    assert report["honjiaReady"] is False
    assert report["summary"]["roundLeakCount"] == 1
    assert report["readinessSummary"]["roundLeakCount"] == 1


def test_run_voice_chain_scenarios_requires_streaming_signals_for_readiness() -> None:
    from apps.body_runtime.voice_chain_scenarios import VoiceScenario, run_voice_chain_scenarios

    report = run_voice_chain_scenarios(
        scenarios=[
            VoiceScenario(
                name="non_streaming_turn",
                description="latency is good but streaming evidence is missing",
                turns=[
                    {
                        "roundId": "rnd-non-streaming-1",
                        "asrFinalMs": 100.0,
                        "firstTokenMs": 100.0,
                        "firstAudioMs": 300.0,
                        "interruptStopMs": 100.0,
                        "roundLeak": False,
                    }
                ],
            )
        ]
    )

    assert report["summary"]["streaming"]["ready"] is False
    assert report["streamingReady"] is False
    assert report["honjiaReady"] is False
    assert report["readinessSummary"]["streamingReady"] is False
    assert "streaming" in report["readinessSummary"]["readinessMessage"]


def test_run_voice_chain_scenarios_preserves_real_streaming_event_metrics_in_custom_turns() -> None:
    from apps.body_runtime.voice_chain_scenarios import VoiceScenario, run_voice_chain_scenarios

    report = run_voice_chain_scenarios(
        scenarios=[
            VoiceScenario(
                name="streaming_sample",
                description="turn assembled from provider-like event timings",
                turns=[
                    {
                        "roundId": "rnd-stream-001",
                        "firstAsrPartialMs": 90.0,
                        "asrFinalMs": 420.0,
                        "firstLlmDeltaMs": 610.0,
                        "firstTokenMs": 610.0,
                        "firstTtsChunkMs": 860.0,
                        "firstAudioMs": 980.0,
                        "streaming": {
                            "asrPartial": True,
                            "asrFinal": True,
                            "llmDelta": True,
                            "ttsChunk": True,
                            "playback": True,
                        },
                        "roundLeak": False,
                    }
                ],
            )
        ]
    )

    assert report["streamingReady"] is True
    assert report["summary"]["metrics"]["firstAsrPartialMs"]["avg"] == 90.0
    assert report["summary"]["metrics"]["firstLlmDeltaMs"]["avg"] == 610.0
    assert report["summary"]["metrics"]["firstTtsChunkMs"]["avg"] == 860.0
    assert report["summary"]["rounds"][0]["streamingReady"] is True
