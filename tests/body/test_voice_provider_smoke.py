from __future__ import annotations

import json


def test_minimax_dry_run_redacts_api_key_and_reports_configured(monkeypatch) -> None:
    from apps.body_runtime.voice_provider_smoke import build_readiness

    secret = "minimax-secret-token"
    monkeypatch.setenv("EIVOICE_MINIMAX_API_KEY", secret)
    monkeypatch.setenv("EIVOICE_MINIMAX_BASE_URL", "wss://example.invalid/minimax")
    monkeypatch.setenv("EIVOICE_MINIMAX_MODEL", "speech-2.8-turbo")
    monkeypatch.setenv("EIVOICE_MINIMAX_VOICE_ID", "female-shaonv")

    readiness = build_readiness("minimax-tts", dry_run=True)
    encoded = json.dumps(readiness)

    assert readiness["provider"] == "minimax-tts"
    assert readiness["configured"] is True
    assert readiness["missing_fields"] == []
    assert readiness["base_url_present"] is True
    assert readiness["model_present"] is True
    assert readiness["voice_id_present"] is True
    assert readiness["dry_run"] is True
    assert readiness["live_supported"] is False
    assert secret not in encoded
    assert readiness["diagnostics"]["api_key"] == "m***n"


def test_minimax_legacy_env_fallback_counts_as_configured(monkeypatch) -> None:
    from apps.body_runtime.voice_provider_smoke import build_readiness

    monkeypatch.setenv("MINIMAX_API_KEY", "legacy-minimax-key")
    monkeypatch.setenv("MINIMAX_BASE_URL", "wss://legacy.invalid/minimax")
    monkeypatch.setenv("MINIMAX_MODEL", "speech-2.8-turbo")
    monkeypatch.setenv("MINIMAX_VOICE_ID", "female-shaonv")

    readiness = build_readiness("minimax-tts", dry_run=True)

    assert readiness["configured"] is True
    assert readiness["missing_fields"] == []


def test_minimax_api_key_only_counts_as_configured_with_runtime_defaults(monkeypatch) -> None:
    from apps.body_runtime.voice_provider_smoke import build_readiness

    monkeypatch.setenv("MINIMAX_API_KEY", "legacy-minimax-key")
    monkeypatch.delenv("MINIMAX_MODEL", raising=False)
    monkeypatch.delenv("EIVOICE_MINIMAX_MODEL", raising=False)
    monkeypatch.delenv("MINIMAX_VOICE_ID", raising=False)
    monkeypatch.delenv("EIVOICE_MINIMAX_VOICE_ID", raising=False)

    readiness = build_readiness("minimax-tts", dry_run=True)

    assert readiness["configured"] is True
    assert readiness["missing_fields"] == []
    assert readiness["model_present"] is False
    assert readiness["voice_id_present"] is False
    assert readiness["defaults_used"] == ["model", "voice_id"]


def test_dashscope_without_key_reports_missing_api_key(monkeypatch) -> None:
    from apps.body_runtime.voice_provider_smoke import build_readiness

    monkeypatch.setenv("EIVOICE_DASHSCOPE_BASE_URL", "wss://example.invalid/dashscope")
    monkeypatch.setenv("EIVOICE_DASHSCOPE_MODEL", "paraformer-realtime-v2")
    monkeypatch.delenv("EIVOICE_DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    readiness = build_readiness("dashscope-asr", dry_run=True)

    assert readiness["provider"] == "dashscope-asr"
    assert readiness["configured"] is False
    assert "api_key" in readiness["missing_fields"]
    assert readiness["base_url_present"] is True
    assert readiness["model_present"] is True
    assert readiness["voice_id_present"] is False


def test_all_providers_cli_aggregates_json_without_secret_leakage(monkeypatch, capsys) -> None:
    from apps.body_runtime.voice_provider_smoke import main

    secret = "aggregate-secret-token"
    monkeypatch.setenv("EIVOICE_MINIMAX_API_KEY", secret)
    monkeypatch.setenv("EIVOICE_MINIMAX_MODEL", "speech-2.8-turbo")
    monkeypatch.setenv("EIVOICE_MINIMAX_VOICE_ID", "female-shaonv")
    monkeypatch.setenv("EIVOICE_DASHSCOPE_MODEL", "paraformer-realtime-v2")
    monkeypatch.delenv("EIVOICE_DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    exit_code = main(["--provider", "all", "--dry-run"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["schema"] == "eibrain.voice_provider_smoke.v1"
    assert [item["provider"] for item in payload["providers"]] == ["minimax-tts", "dashscope-asr"]
    assert payload["dry_run"] is True
    assert secret not in captured.out
