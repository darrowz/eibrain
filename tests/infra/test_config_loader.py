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
