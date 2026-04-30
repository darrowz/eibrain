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
        AudioTranscriptFinal(ts=1.0, source="ear.asr", text="记住我喜欢简短回复", session_id="s1", actor_id="user-1")
    )

    assert runtime.memory.remembered
    payload = runtime.memory.remembered[0]
    assert payload["session_id"] == "s1"
    assert payload["actor_id"] == "user-1"
    assert payload["source"] == "eibrain.audio_dialogue"
    assert payload["modality"] == "audio_text"
    assert payload["organ"] == "ear"
    assert payload["outcome"]["status"] == "planned"
    assert payload["outcome"]["action_count"] == 1
    snapshot = runtime.snapshot()
    assert snapshot["last_attention"]["should_reply"] is True
    assert snapshot["last_policy_decision"]["decision_type"] == "reply"


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
    assert payload["outcome"]["status"] == "planned"
    assert "modality" in runtime.memory.queries[0].task_context
    assert runtime.memory.queries[0].task_context["task_type"] == "brain.orient"
    assert runtime.memory.queries[0].task_context["modality"] == "vision"


def test_cognitive_runtime_observes_outcome_when_memory_supports_it() -> None:
    from apps.cognitive_runtime.app import CognitiveRuntimeApp
    from eibrain.memory.contracts import MemoryResult
    from eibrain.protocol.observations import AudioTranscriptFinal

    class _MemoryAdapter:
        def __init__(self) -> None:
            self.observed: list[dict[str, object]] = []

        def retrieve_context(self, query):
            return MemoryResult(summary="Prefer concise spoken replies.")

        def remember_episode(self, **kwargs):
            pass

        def observe_outcome(self, **kwargs):
            self.observed.append(dict(kwargs))

        def get_active_policy(self, **kwargs):
            return {"response_policy": {"reply_style": "concise"}}

    runtime = CognitiveRuntimeApp()
    runtime.memory = _MemoryAdapter()
    runtime.handle_observation(
        AudioTranscriptFinal(ts=1.0, source="ear.asr", text="hello", session_id="s1", actor_id="user-1")
    )
    assert runtime.memory.observed == []

    runtime.handle_observation(
        AudioTranscriptFinal(ts=2.0, source="ear.asr", text="记住我喜欢简短回复", session_id="s1", actor_id="user-1")
    )

    assert runtime.memory.observed
    assert runtime.memory.observed[0]["signal_type"] == "cognitive_turn"
    assert runtime.memory.observed[0]["payload"]["decision"] == "reply"
    assert runtime.memory.observed[0]["payload"]["trace_reason"] == "explicit_remember"


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



def test_cognitive_runtime_records_skill_trace_after_audio_turn() -> None:
    from apps.cognitive_runtime.app import CognitiveRuntimeApp
    from eibrain.memory.contracts import MemoryResult
    from eibrain.protocol.observations import AudioTranscriptFinal

    class _MemoryAdapter:
        def __init__(self):
            self.skill_traces = []
            self.last_writeback_status = {}

        def retrieve_context(self, query):
            return MemoryResult(summary="")

        def remember_episode(self, **kwargs):
            return None

        def observe_outcome(self, **kwargs):
            return None

        def record_skill_trace(self, payload, **kwargs):
            self.skill_traces.append((payload, kwargs))

    runtime = CognitiveRuntimeApp()
    memory = _MemoryAdapter()
    runtime.memory = memory

    runtime.handle_observation(AudioTranscriptFinal(ts=1.0, source="test", session_id="s1", actor_id="user-1", text="hello eibrain"))
    assert memory.skill_traces == []

    runtime.handle_observation(AudioTranscriptFinal(ts=2.0, source="test", session_id="s1", actor_id="user-1", text="记住我喜欢用 pytest -q"))

    assert memory.skill_traces
    payload, kwargs = memory.skill_traces[0]
    assert payload["trace_id"] == "s1"
    assert payload["task_type"] == "brain.respond"
    assert payload["input_summary"] == "记住我喜欢用 pytest -q"
    assert "reply.default" in payload["selected_skills"]
    assert payload["outcome"] == "planned"
    assert payload["meta"]["write_policy_version"] == "meaningful_event_v1"
    assert payload["meta"]["trace_reason"] == "explicit_remember"
    assert payload["meta"]["write_filter"]["allowed"] is True
    assert kwargs == {"session_id": "s1", "actor_id": "user-1"}


def test_cognitive_runtime_records_skill_trace_after_visual_turn() -> None:
    from apps.cognitive_runtime.app import CognitiveRuntimeApp
    from eibrain.memory.contracts import MemoryResult

    class _VisionUnderstanding:
        primary_subject = "person"
        summary = "person standing near desk"

    class _VisionAdapter:
        def understand_image(self, **kwargs):
            return _VisionUnderstanding()

    class _MemoryAdapter:
        def __init__(self):
            self.skill_traces = []
            self.last_writeback_status = {}

        def retrieve_context(self, query):
            return MemoryResult(summary="")

        def remember_episode(self, **kwargs):
            return None

        def observe_outcome(self, **kwargs):
            return None

        def record_skill_trace(self, payload, **kwargs):
            self.skill_traces.append((payload, kwargs))

    runtime = CognitiveRuntimeApp(vision_adapter=_VisionAdapter())
    memory = _MemoryAdapter()
    runtime.memory = memory

    runtime.handle_visual_frame(image_url="https://example.com/frame.jpg", actor_id="user-1", target_x=0.7)
    runtime.handle_visual_frame(image_url="https://example.com/frame.jpg", actor_id="user-1", target_x=0.7)

    assert len(memory.skill_traces) == 1
    payload, kwargs = memory.skill_traces[0]
    assert payload["trace_id"] == "vision:user-1"
    assert payload["task_type"] == "brain.orient"
    assert payload["input_summary"] == "person standing near desk"
    assert "orient.default" in payload["selected_skills"]
    assert payload["meta"]["write_policy_version"] == "meaningful_event_v1"
    assert payload["meta"]["trace_reason"] == "visual_new_scene"
    assert payload["meta"]["write_filter"]["dedupe_key"] == "person:right"
    assert kwargs == {"session_id": "vision:user-1", "actor_id": "user-1"}

    runtime.handle_visual_frame(image_url="https://example.com/frame.jpg", actor_id="user-1", target_x=0.2)
    assert len(memory.skill_traces) == 2
    assert memory.skill_traces[1][0]["meta"]["trace_reason"] == "visual_state_change"
