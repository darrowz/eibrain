from __future__ import annotations


def test_load_config_supports_unified_deployment_and_streaming_model_choices(tmp_path) -> None:
    from eibrain.infra.config import load_config

    config_path = tmp_path / "eibrain.yaml"
    config_path.write_text(
        "\n".join(
            [
                "system:",
                "  project_name: eibrain",
                "deployment:",
                "  root_dir: /home/tester/eibrain",
                "  allow_override: true",
                "body:",
                "  node_id: honjia",
                "  foundation:",
                "    ear: camera_mic",
                "    eye: camera",
                "    mouth: speaker",
                "    neck: gimbal",
                "  organs:",
                "    ear:",
                "      asr:",
                "        driver:",
                "          kind: noop",
                "          provider: sherpa_onnx",
                "          mode: streaming",
                "          model_dir: /models/sherpa-streaming-zh",
                "    eye:",
                "      detection:",
                "        driver:",
                "          kind: noop",
                "          provider: hailo8",
                "          device: /dev/hailo0",
                "cognition:",
                "  node_id: honxin",
                "  llm:",
                "    provider: minimax",
                "    model: MiniMax-M2.7-highspeed",
                "  vision_llm:",
                "    provider: minimax",
                "    model: coding-plan-vlm",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.deployment.root_dir == "/home/tester/eibrain"
    assert config.deployment.body_runtime_dir == "/home/tester/eibrain"
    assert config.deployment.cognitive_runtime_dir == "/home/tester/eibrain"
    assert config.body.foundation["ear"] == "camera_mic"
    assert config.body.organs["ear"].subfunctions["asr"].driver.extra["provider"] == "sherpa_onnx"
    assert config.body.organs["ear"].subfunctions["asr"].driver.extra["mode"] == "streaming"
    assert config.cognition.llm.provider == "minimax"
    assert config.cognition.llm.model == "MiniMax-M2.7-highspeed"
    assert config.cognition.vision_llm.model == "coding-plan-vlm"
