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
