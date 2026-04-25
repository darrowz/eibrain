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


def test_cognitive_runtime_builds_eimemory_rpc_adapter_from_config(tmp_path) -> None:
    from apps.cognitive_runtime.app import CognitiveRuntimeApp

    config_path = tmp_path / "eibrain.yaml"
    config_path.write_text(
        "\n".join(
            [
                "memory:",
                "  openclaw:",
                "    provider: eimemory_rpc",
                "    endpoint: http://127.0.0.1:8091/",
            ]
        ),
        encoding="utf-8",
    )

    runtime = CognitiveRuntimeApp.from_config_path(config_path)

    assert runtime.memory.__class__.__name__ == "EIMemoryRPCAdapter"



def test_cognitive_runtime_uses_eimemory_rpc_memory_in_handle_observation(monkeypatch, tmp_path) -> None:
    from apps.cognitive_runtime.app import CognitiveRuntimeApp
    from eibrain.memory.contracts import MemoryResult
    from eibrain.protocol.observations import AudioTranscriptFinal

    config_path = tmp_path / "eibrain.yaml"
    config_path.write_text(
        "\n".join(
            [
                "cognition:",
                "  llm:",
                "    provider: echo",
                "memory:",
                "  openclaw:",
                "    provider: eimemory_rpc",
                "    endpoint: http://127.0.0.1:8091/",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "eibrain.memory.adapters.eimemory_rpc.EIMemoryRPCAdapter.retrieve_context",
        lambda self, query: MemoryResult(summary="Prefer concise replies."),
    )

    runtime = CognitiveRuntimeApp.from_config_path(config_path)
    actions = runtime.handle_observation(
        AudioTranscriptFinal(ts=1.0, source="ear.asr", text="hello", session_id="s1", actor_id="user-1")
    )

    assert actions
    assert runtime.memory.__class__.__name__ == "EIMemoryRPCAdapter"


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


def test_cognitive_runtime_records_audio_episode_for_memory() -> None:
    from apps.cognitive_runtime.app import CognitiveRuntimeApp
    from eibrain.memory.contracts import MemoryResult
    from eibrain.protocol.observations import AudioTranscriptFinal

    class _MemoryAdapter:
        def __init__(self) -> None:
            self.remembered: list[dict[str, object]] = []

        def retrieve_context(self, query):
            return MemoryResult(summary="Prefer concise spoken replies.")

        def remember_episode(self, **kwargs):
            self.remembered.append(dict(kwargs))

    runtime = CognitiveRuntimeApp()
    runtime.memory = _MemoryAdapter()
    runtime.handle_observation(
        AudioTranscriptFinal(ts=1.0, source="ear.asr", text="hello", session_id="s1", actor_id="user-1")
    )

    assert runtime.memory.remembered
    payload = runtime.memory.remembered[0]
    assert payload["session_id"] == "s1"
    assert payload["actor_id"] == "user-1"
    assert payload["source"] == "eibrain.audio_dialogue"
    assert payload["modality"] == "audio_text"
    assert payload["organ"] == "ear"


def test_cognitive_runtime_records_visual_episode_for_memory() -> None:
    from apps.cognitive_runtime.app import CognitiveRuntimeApp
    from eibrain.memory.contracts import MemoryResult

    class _VisionAdapter:
        def understand_image(self, *, prompt: str, image_url: str):
            from eibrain.vision.minimax_mcp import VisionUnderstandingResult

            return VisionUnderstandingResult(
                summary="a person is centered in frame",
                primary_subject="person",
                confidence=0.9,
            )

    class _MemoryAdapter:
        def __init__(self) -> None:
            self.remembered: list[dict[str, object]] = []
            self.queries: list[object] = []

        def retrieve_context(self, query):
            self.queries.append(query)
            return MemoryResult(summary="Track the current speaker.")

        def remember_episode(self, **kwargs):
            self.remembered.append(dict(kwargs))

    runtime = CognitiveRuntimeApp(vision_adapter=_VisionAdapter())
    runtime.memory = _MemoryAdapter()
    runtime.handle_visual_frame(
        image_url="https://example.com/frame.jpg",
        actor_id="user-1",
        target_x=0.8,
    )

    assert runtime.memory.queries
    assert runtime.memory.queries[0].session_id == "vision:user-1"
    assert runtime.memory.queries[0].actor_id == "user-1"
    assert runtime.memory.remembered
    payload = runtime.memory.remembered[0]
    assert payload["session_id"] == "vision:user-1"
    assert payload["actor_id"] == "user-1"
    assert payload["source"] == "eibrain.visual_frame"
    assert payload["modality"] == "vision"
    assert payload["organ"] == "eye"


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
