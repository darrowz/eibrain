from __future__ import annotations

import importlib.util
import json
from pathlib import Path


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


def test_voice_chain_selftest_exposes_check_summary() -> None:
    from apps.body_runtime.voice_chain_selftest import run_voice_chain_selftest

    report = run_voice_chain_selftest()

    assert report["checks"]["code_ready"]["ok"] is True
    assert report["checks"]["streaming_ready"]["ok"] is True
    assert report["checks"]["interrupt_stop_ready"]["ok"] is True
    assert report["checks"]["round_leak_free"]["ok"] is True


def test_honjia_readiness_report_runs_offline_without_live_dependencies() -> None:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "honjia_readiness_report.py"
    spec = importlib.util.spec_from_file_location("honjia_readiness_report", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    report = module.build_report(live=False)

    assert report["schema"] == "eibrain.honjia_readiness_report.v1"
    assert report["offline_capable"] is True
    assert report["voice_chain_selftest"]["codeReady"] is True
    assert report["vision_soak"]["collection"]["source"] == "synthetic"
    assert report["checks"]["monitor_active"]["ok"] is None


def test_honjia_readiness_report_loads_dotenv_without_overwriting_existing_env(tmp_path, monkeypatch) -> None:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "honjia_readiness_report.py"
    spec = importlib.util.spec_from_file_location("honjia_readiness_report", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "EIVOICE_MINIMAX_API_KEY=from-file",
                "EIVOICE_DASHSCOPE_API_KEY='dash-file'",
                "IGNORED_WITHOUT_EQUALS",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("EIVOICE_MINIMAX_API_KEY", "existing")
    monkeypatch.delenv("EIVOICE_DASHSCOPE_API_KEY", raising=False)

    result = module._load_env_file(env_file)

    assert result["loaded"] == 1
    assert result["status"] == "loaded"
    assert result["path"] == str(env_file)
    assert module.os.environ["EIVOICE_MINIMAX_API_KEY"] == "existing"
    assert module.os.environ["EIVOICE_DASHSCOPE_API_KEY"] == "dash-file"
