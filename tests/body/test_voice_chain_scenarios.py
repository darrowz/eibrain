from __future__ import annotations

import json


def test_run_voice_chain_scenarios_reports_honjia_ready_summary() -> None:
    from apps.body_runtime.voice_chain_scenarios import run_voice_chain_scenarios

    report = run_voice_chain_scenarios()

    assert report["schema"] == "eibrain.voice_chain_scenarios.v1"
    assert report["scenarioCount"] >= 5
    assert report["turnCount"] >= 5
    assert report["honjiaReady"] is True
    assert report["summary"]["roundLeakCount"] == 0
    assert {"asrFinalMs", "firstTokenMs", "firstAudioMs", "interruptStopMs"} <= set(report["summary"]["metrics"])
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


def test_run_voice_chain_scenarios_does_not_treat_empty_explicit_input_as_ready() -> None:
    from apps.body_runtime.voice_chain_scenarios import run_voice_chain_scenarios

    report = run_voice_chain_scenarios(scenarios=[])

    assert report["scenarioCount"] == 0
    assert report["turnCount"] == 0
    assert report["roundLeakFree"] is True
    assert report["honjiaReady"] is False
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
