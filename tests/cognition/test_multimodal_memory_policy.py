from __future__ import annotations

from eibrain.cognition.policy.multimodal_memory import MultimodalMemoryPolicy


def test_audio_response_recall_prefers_hongtu_subject_memory() -> None:
    context = MultimodalMemoryPolicy().build_recall_context(
        task_type="brain.respond",
        modality="audio_text",
        organ="ear",
        phase="engaged",
        salience_score=0.8,
        body_capabilities={"can_hear_voice": True},
        query="介绍下你自己",
    )

    assert context["recall_profile"] == "precision"
    assert "eimemory.knowledge.claims" in context["blocked_sources"]
    assert "eibrain.audio_dialogue" in context["allowed_sources"]
    assert "audio_text" in context["preferred_modalities"]
    assert "ear" in context["organs"]
    assert context["source_system"] == "eibrain"
    assert context["channel_id"] == "voice.honjia"
    assert context["agent_id"] == "eibrain.voice"
    assert context["memory_contract_version"] == "multimodal-memory.v1"
    assert context["recall_filters"]["channel_ids"] == ["voice.honjia", "global.profile", "global.summary"]


def test_visual_orient_recall_prefers_visual_identity_memory() -> None:
    context = MultimodalMemoryPolicy().build_recall_context(
        task_type="brain.orient",
        modality="vision",
        organ="eye",
        phase="attending",
        salience_score=0.7,
        body_capabilities={"can_see_people": True},
        query="person face",
        visual_context={"target_x": 0.42},
    )

    assert context["allowed_sources"] == [
        "eibrain.visual_world",
        "eibrain.visual_frame",
        "eibrain.identity",
        "eibrain.policy",
    ]
    assert "world_observation" in context["allowed_memory_types"]
    assert context["preferred_modalities"] == ["vision", "multimodal"]
    assert context["visual_context"] == {"target_x": 0.42}
    assert context["recall_filters"]["source_systems"] == ["eibrain", "openclaw"]


def test_diagnostic_recall_uses_policy_sources() -> None:
    context = MultimodalMemoryPolicy().build_recall_context(
        task_type="brain.respond",
        modality="audio_text",
        organ="ear",
        phase="engaged",
        salience_score=0.9,
        body_capabilities={},
        query="ASR 为什么识别不准",
    )

    assert "eibrain.policy" in context["allowed_sources"]
    assert "policy" in context["allowed_memory_types"]
    assert set(context["organs"]) >= {"ear", "eye", "mouth", "neck"}


def test_writeback_outcome_tags_subject_modality_and_organ() -> None:
    outcome = MultimodalMemoryPolicy().writeback_outcome(
        modality="audio_text",
        organ="ear",
        success=True,
        status="planned",
        action_count=1,
        reply_present=True,
        learning_decision="keep_policy",
    )

    assert outcome["subject"] == "hongtu"
    assert outcome["modality"] == "audio_text"
    assert outcome["organ"] == "ear"
    assert outcome["source_system"] == "eibrain"
    assert outcome["channel_id"] == "voice.honjia"
    assert outcome["agent_id"] == "eibrain.voice"
    assert outcome["memory_contract_version"] == "multimodal-memory.v1"


def test_cognitive_runtime_builds_world_observation_payload_from_visual_state() -> None:
    from apps.cognitive_runtime.app import CognitiveRuntimeApp

    runtime = CognitiveRuntimeApp()
    payload = runtime.build_world_observation_payload(
        {
            "source": "state",
            "frame_path": "/tmp/eibrain-vision/latest.jpg",
            "objects": [
                {"label": "person", "confidence": 0.91, "bbox": {"x_min": 0.1}},
                {"label": "cup", "score": 0.74},
            ],
            "identity_candidates": [{"candidate_id": "unknown-face-1"}],
        },
        session_id="vision:user-1",
        actor_id="user-1",
    )

    assert payload["content"]["objects"] == [
        {"label": "person", "confidence": 0.91, "bbox": {"x_min": 0.1}},
        {"label": "cup", "confidence": 0.74},
    ]
    assert payload["content"]["source"] == "state"
    assert payload["content"]["confidence"] == 0.91
    assert payload["meta"]["source"] == "eibrain.visual_world"
    assert payload["meta"]["dedupe_key"].startswith("world_observation:")
    assert payload["meta"]["session_id"] == "vision:user-1"
    assert payload["meta"]["actor_id"] == "user-1"
    assert "world_observation" in payload["tags"]
    assert "person" in payload["tags"]


def test_cognitive_runtime_writes_world_observation_from_visual_state() -> None:
    from apps.cognitive_runtime.app import CognitiveRuntimeApp

    class _Memory:
        last_writeback_status = {"status": "idle"}

        def __init__(self) -> None:
            self.captured = {}

        def retrieve_context(self, query):  # pragma: no cover - not used by this test
            raise AssertionError("retrieve_context should not be called")

        def remember_episode(self, **kwargs):  # pragma: no cover - not used by this test
            raise AssertionError("remember_episode should not be called")

        def remember_world_observation(self, **kwargs) -> None:
            self.captured = kwargs
            self.last_writeback_status = {"status": "ok", "source": "eibrain.visual_world"}

    runtime = CognitiveRuntimeApp()
    memory = _Memory()
    runtime.memory = memory

    payload = runtime.remember_world_observation_from_state(
        {
            "backend": "gstreamer_hailo",
            "objects": [{"label": "cup", "score": 0.8}],
        },
        session_id="vision:desk",
    )

    assert payload["summary"] == "Observed cup"
    assert memory.captured["session_id"] == "vision:desk"
    assert memory.captured["summary"] == "Observed cup"
    assert memory.captured["content"]["objects"][0]["label"] == "cup"
    assert runtime.last_memory_diagnostics["last_writeback"]["status"] == "ok"
