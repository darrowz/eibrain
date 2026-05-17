from __future__ import annotations

import os


def test_load_config_reads_unified_yaml_and_expands_env(tmp_path) -> None:
    from eibrain.infra.config import load_config

    config_path = tmp_path / "eibrain.yaml"
    os.environ["EIBRAIN_TEST_API_KEY"] = "secret-key"
    config_path.write_text(
        "\n".join(
            [
                "system:",
                "  project_name: eibrain",
                "  environment: integration",
                "body:",
                "  node_id: honjia",
                "  organs:",
                "    ear:",
                "      enabled: true",
                "      capture:",
                "        driver:",
                "          kind: noop",
                "    mouth:",
                "      enabled: true",
                "      tts_playback:",
                "        driver:",
                "          kind: noop",
                "cognition:",
                "  node_id: honxin",
                "  llm:",
                "    provider: echo",
                "    api_key: ${EIBRAIN_TEST_API_KEY}",
                "memory:",
                "  openclaw:",
                "    provider: in_memory",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.system.project_name == "eibrain"
    assert config.body.node_id == "honjia"
    assert config.cognition.llm.api_key == "secret-key"
    assert config.memory.openclaw.provider == "in_memory"


def test_load_config_reads_eimemory_rpc_fields(tmp_path) -> None:
    from eibrain.infra.config import load_config

    config_path = tmp_path / "eibrain.yaml"
    config_path.write_text(
        "\n".join(
            [
                "memory:",
                "  openclaw:",
                "    provider: eimemory_rpc",
                "    endpoint: http://127.0.0.1:8091/",
                "    timeout_s: 1.5",
                "    agent_id: honxin",
                "    workspace_id: honjia-prod",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.memory.openclaw.provider == "eimemory_rpc"
    assert config.memory.openclaw.endpoint == "http://127.0.0.1:8091/"
    assert config.memory.openclaw.agent_id == "honxin"
    assert config.memory.openclaw.workspace_id == "honjia-prod"


def test_load_config_supports_env_defaults(tmp_path) -> None:
    from eibrain.infra.config import load_config

    config_path = tmp_path / "eibrain.yaml"
    config_path.write_text(
        "\n".join(
            [
                "cognition:",
                "  llm:",
                "    provider: anthropic_compatible",
                "    model: ${ANTHROPIC_MODEL:-MiniMax-M2.7-highspeed}",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.cognition.llm.model == "MiniMax-M2.7-highspeed"


def test_load_config_reads_openclaw_hontu_llm_command(tmp_path) -> None:
    from eibrain.infra.config import load_config

    config_path = tmp_path / "eibrain.yaml"
    config_path.write_text(
        "\n".join(
            [
                "cognition:",
                "  llm:",
                "    provider: openclaw_hontu",
                "    command:",
                "      - ssh",
                "      - honxin",
                "      - /home/darrow/n/bin/openclaw",
                "    agent_id: main",
                "    session_id: eibrain-honjia-voice",
                "    timeout_s: 45",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.cognition.llm.provider == "openclaw_hontu"
    assert config.cognition.llm.command == ["ssh", "honxin", "/home/darrow/n/bin/openclaw"]
    assert config.cognition.llm.agent_id == "main"
    assert config.cognition.llm.session_id == "eibrain-honjia-voice"
    assert config.cognition.llm.timeout_s == 45.0


def test_load_config_normalizes_cli_command_string(tmp_path) -> None:
    from eibrain.infra.config import load_config

    config_path = tmp_path / "eibrain.yaml"
    config_path.write_text(
        "\n".join(
            [
                "vision:",
                "  provider: minimax_cli",
                "  cli:",
                "    enabled: true",
                "    command: /home/darrow/.npm-global/bin/mmx",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.vision.cli.command == ["/home/darrow/.npm-global/bin/mmx"]


def test_load_config_reads_monitoring_settings(tmp_path) -> None:
    from eibrain.infra.config import load_config

    config_path = tmp_path / "eibrain.yaml"
    config_path.write_text(
        "\n".join(
            [
                "monitoring:",
                "  enabled: true",
                "  host: 0.0.0.0",
                "  port: 18080",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.monitoring.enabled is True
    assert config.monitoring.port == 18080


def test_load_config_normalizes_driver_health_command_string(tmp_path) -> None:
    from eibrain.infra.config import load_config

    config_path = tmp_path / "eibrain.yaml"
    config_path.write_text(
        "\n".join(
            [
                "body:",
                "  organs:",
                "    ear:",
                "      capture:",
                "        driver:",
                "          kind: command",
                "          command: python",
                "          health_command: python",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.body.organs["ear"].subfunctions["capture"].driver.extra["health_command"] == ["python"]


def test_honjia_config_uses_eimemory_endpoint_and_scope(monkeypatch) -> None:
    from pathlib import Path

    from eibrain.infra.config import load_config

    monkeypatch.setenv("EIMEMORY_ENDPOINT", "http://honxin:8091/")
    monkeypatch.delenv("EIBRAIN_MEMORY_ENDPOINT", raising=False)
    config_path = Path(__file__).resolve().parents[2] / "config" / "eibrain.honjia.yaml"

    config = load_config(config_path)

    assert config.memory.openclaw.provider == "eimemory_rpc"
    assert config.memory.openclaw.endpoint == "http://honxin:8091/"
    assert config.memory.openclaw.agent_id == "honxin"
    assert config.memory.openclaw.workspace_id == "honjia"
    assert config.cognition.llm.provider == "openclaw_hontu"
    assert config.cognition.llm.command[-4:] == [
        "honxin",
        "env",
        "PATH=/home/darrow/n/bin:/usr/local/bin:/usr/bin:/bin",
        "/home/darrow/n/bin/openclaw",
    ]
    assert config.cognition.llm.agent_id == "main"
    assert config.cognition.llm.session_id == "eibrain-honjia-voice"


def test_honjia_config_normalizes_field_wake_word_confusions(monkeypatch) -> None:
    from pathlib import Path

    from eibrain.infra.config import load_config

    monkeypatch.setenv("EIMEMORY_ENDPOINT", "http://honxin:8091/")
    config_path = Path(__file__).resolve().parents[2] / "config" / "eibrain.honjia.yaml"

    config = load_config(config_path)
    replacements = (
        config.body.organs["ear"]
        .subfunctions["asr"]
        .driver.extra["transcript_replacements"]
    )

    for phrase in ("你好洪福", "你好胡服", "你好红烛", "你好红姑", "你好黄图", "你好黄渤"):
        assert replacements[phrase] == "你好鸿途"
    assert "黄渤" not in replacements


def test_primary_config_uses_reachable_eimemory_endpoint(monkeypatch) -> None:
    from pathlib import Path

    from eibrain.infra.config import load_config

    monkeypatch.delenv("EIMEMORY_ENDPOINT", raising=False)
    config_path = Path(__file__).resolve().parents[2] / "config" / "eibrain.yaml"

    config = load_config(config_path)

    assert config.memory.openclaw.provider == "eimemory_rpc"
    assert config.memory.openclaw.endpoint == "http://100.66.161.64:8091/"

