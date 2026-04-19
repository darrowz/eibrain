from __future__ import annotations


def test_cognitive_runtime_uses_yaml_configured_services(tmp_path) -> None:
    from apps.cognitive_runtime.app import CognitiveRuntimeApp
    from eibrain.protocol.observations import AudioTranscriptFinal

    config_path = tmp_path / "eibrain.yaml"
    config_path.write_text(
        "\n".join(
            [
                "system:",
                "  project_name: eibrain",
                "cognition:",
                "  node_id: honxin",
                "  llm:",
                "    provider: echo",
                "memory:",
                "  openclaw:",
                "    provider: in_memory",
            ]
        ),
        encoding="utf-8",
    )

    runtime = CognitiveRuntimeApp.from_config_path(config_path)
    actions = runtime.handle_observation(
        AudioTranscriptFinal(
            ts=1.0,
            source="ear.asr",
            text="hello eibrain",
            session_id="session-1",
            actor_id="user-1",
        )
    )

    assert len(actions) == 1
    assert actions[0].kind == "play_speech_action"
    assert "hello" in actions[0].text


def test_cognitive_runtime_can_handle_visual_focus_with_mcp_adapter() -> None:
    from apps.cognitive_runtime.app import CognitiveRuntimeApp

    class _VisionAdapter:
        def understand_image(self, *, prompt: str, image_url: str):
            from eibrain.vision.minimax_mcp import VisionUnderstandingResult

            return VisionUnderstandingResult(
                summary="a person is centered in frame",
                primary_subject="person",
                confidence=0.9,
            )

    runtime = CognitiveRuntimeApp(vision_adapter=_VisionAdapter())
    actions = runtime.handle_visual_frame(
        image_url="https://example.com/frame.jpg",
        actor_id="user-1",
        target_x=0.8,
    )

    assert len(actions) == 1
    assert actions[0].kind == "move_head_action"
    assert actions[0].target_x == 0.8


def test_cognitive_runtime_prefers_minimax_cli_provider(tmp_path) -> None:
    from apps.cognitive_runtime.app import CognitiveRuntimeApp

    config_path = tmp_path / "eibrain.yaml"
    config_path.write_text(
        "\n".join(
            [
                "vision:",
                "  provider: minimax_cli",
                "  cli:",
                "    enabled: true",
                "    api_key: secret",
                "    base_url: https://api.minimaxi.com",
            ]
        ),
        encoding="utf-8",
    )

    runtime = CognitiveRuntimeApp.from_config_path(config_path)

    assert runtime.vision is not None
    assert runtime.vision.__class__.__name__ == "MiniMaxCLIAdapter"


def test_cognitive_runtime_disables_minimax_cli_without_credentials(tmp_path) -> None:
    from apps.cognitive_runtime.app import CognitiveRuntimeApp

    config_path = tmp_path / "eibrain.yaml"
    config_path.write_text(
        "\n".join(
            [
                "vision:",
                "  provider: minimax_cli",
                "  cli:",
                "    enabled: true",
                "    command: mmx",
                "    api_key: ''",
                "    base_url: https://api.minimaxi.com",
            ]
        ),
        encoding="utf-8",
    )

    runtime = CognitiveRuntimeApp.from_config_path(config_path)

    assert runtime.vision is None


def test_cognitive_runtime_disables_minimax_mcp_without_credentials(tmp_path) -> None:
    from apps.cognitive_runtime.app import CognitiveRuntimeApp

    config_path = tmp_path / "eibrain.yaml"
    config_path.write_text(
        "\n".join(
            [
                "vision:",
                "  provider: minimax_mcp",
                "  mcp:",
                "    enabled: true",
                "    command:",
                "      - uvx",
                "      - minimax-coding-plan-mcp",
                "    api_key: ''",
            ]
        ),
        encoding="utf-8",
    )

    runtime = CognitiveRuntimeApp.from_config_path(config_path)

    assert runtime.vision is None
