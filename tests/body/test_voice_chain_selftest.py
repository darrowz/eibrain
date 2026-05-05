from __future__ import annotations

import json


def test_voice_chain_selftest_replays_full_event_path_without_hardware() -> None:
    from apps.body_runtime.voice_chain_selftest import run_voice_chain_selftest

    report = run_voice_chain_selftest()

    assert report["schema"] == "eibrain.voice_chain_selftest.v1"
    assert report["source"] == "offline_selftest"
    assert report["honjiaRequired"] is False
    assert report["codeReady"] is True
    assert report["roundLeakFree"] is True
    assert report["streamingReady"] is True
    assert report["interruptStopReady"] is True
    assert report["failedMetrics"] == []
    assert report["benchmark"]["turnCount"] == 2
    assert report["benchmark"]["rounds"][0]["stageLatencyMs"]["listen_asr"] > 0
    assert report["benchmark"]["rounds"][1]["interrupted"] is True
    assert report["benchmark"]["interruptStop"]["ready"] is True
    assert report["benchmark"]["streaming"]["ready"] is True
    assert report["benchmark"]["readinessSummary"]["codeReady"] is True
    assert report["benchmark"]["readinessSummary"]["streamingReady"] is True
    assert report["benchmark"]["readinessSummary"]["interruptStopReady"] is True
    assert report["normalTurn"]["snapshot"]["closed_loop"] is True
    assert report["interruptTurn"]["snapshot"]["interrupted"] is True
    assert report["interruptTurn"]["snapshot"]["tts_stopped"] is True
    assert report["readiness"]["source"] == "offline_selftest"
    assert report["readiness"]["live"] is False
    assert report["readiness"]["honjiaReady"] is False
    assert report["readiness"]["codeReady"] is True
    assert report["readiness"]["streamingReady"] is True
    assert report["readiness"]["interruptStopReady"] is True
    assert "offline protocol selftest passed" in report["readiness"]["readinessMessage"]

    event_names = set(report["eventNames"])
    assert "ei.voice.audio.frame" in event_names
    assert "ei.voice.asr.partial" in event_names
    assert "ei.voice.asr.final" in event_names
    assert "ei.dialogue.agent.delta" in event_names
    assert "ei.voice.tts.sentence_start" in event_names
    assert "ei.voice.tts.chunk" in event_names
    assert "ei.voice.playback.started" in event_names
    assert "ei.voice.playback.stopped" in event_names
    assert "ei.voice.barge_in.detected" in event_names
    assert report["operationCounts"]["interrupt"] == 1
    assert report["operationCounts"]["complete_playback"] == 1


def test_voice_chain_selftest_main_prints_json(capsys) -> None:
    from apps.body_runtime.voice_chain_selftest import main

    exit_code = main([])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["schema"] == "eibrain.voice_chain_selftest.v1"
    assert payload["codeReady"] is True
