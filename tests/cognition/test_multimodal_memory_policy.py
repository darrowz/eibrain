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


def test_subject_voice_recall_blocks_knowledge_news_and_records_decision_trace() -> None:
    context = MultimodalMemoryPolicy().build_recall_context(
        task_type="brain.respond",
        modality="audio_text",
        organ="ear",
        phase="engaged",
        salience_score=0.74,
        body_capabilities={"can_hear_voice": True, "can_speak": True},
        query="你还记得我刚才说喜欢什么吗",
        trace_id="trace-voice-1",
        source_event_id="asr-final-1",
    )

    assert context["recall_profile"] == "subject_dialogue"
    assert context["allowed_sources"] == [
        "eibrain.identity",
        "eibrain.preference",
        "eibrain.dialogue",
        "eibrain.audio_dialogue",
        "openclaw.agent_end",
    ]
    assert set(context["blocked_sources"]) >= {
        "eimemory.knowledge.claims",
        "eimemory.knowledge_base",
        "eimemory.news",
        "eimemory.paper",
        "eimemory.research",
    }
    assert context["privacy"] == {
        "scope": "subject_conversation",
        "sensitivity": "personal",
        "allowed_use": "embodied_response",
    }
    assert context["writeback_eligibility"] == {
        "eligible": True,
        "requires_explicit_memory_request": True,
        "default_memory_type": "conversation",
    }
    assert context["decision_trace"]["decision"] == "voice_subject_dialogue_recall"
    assert "avoid knowledge/news/paper sources" in context["decision_trace"]["why"]


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


def test_visual_orient_recall_uses_visual_profile_and_low_sensitivity_trace() -> None:
    context = MultimodalMemoryPolicy().build_recall_context(
        task_type="brain.orient",
        modality="vision",
        organ="eye",
        phase="attending",
        salience_score=0.69,
        body_capabilities={"can_see_people": True},
        query="look at the person near the desk",
        visual_context={"target_label": "person"},
    )

    assert context["recall_profile"] == "visual_grounding"
    assert "eimemory.news" in context["blocked_sources"]
    assert context["privacy"]["scope"] == "situational_awareness"
    assert context["privacy"]["sensitivity"] == "environmental"
    assert context["writeback_eligibility"] == {
        "eligible": True,
        "requires_explicit_memory_request": False,
        "default_memory_type": "world_observation",
    }
    assert context["decision_trace"]["decision"] == "vision_world_grounding_recall"


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


def test_action_feedback_recall_excludes_identity_memory_sources() -> None:
    context = MultimodalMemoryPolicy().build_recall_context(
        task_type="head.execute",
        modality="multimodal_action",
        organ="neck",
        phase="acting",
        salience_score=0.86,
        body_capabilities={"can_move_head": True},
        query="上次转头失败，下次应该怎么调整",
        trace_id="trace-neck-1",
        source_event_id="evt-outcome-1",
    )

    assert "eibrain.head_feedback" in context["allowed_sources"]
    assert "eibrain.outcome_feedback" in context["allowed_sources"]
    assert "eibrain.identity" in context["blocked_sources"]
    assert "identity" not in context["allowed_memory_types"]
    assert "procedural_adjustment_candidate" in context["allowed_memory_types"]
    assert context["trace_id"] == "trace-neck-1"
    assert context["source_event_id"] == "evt-outcome-1"
    assert context["recall_filters"]["memory_types"] == context["allowed_memory_types"]
    assert context["recall_profile"] == "action_outcome_repair"
    assert context["privacy"]["sensitivity"] == "operational"
    assert context["writeback_eligibility"]["default_memory_type"] == "action_outcome"
    assert context["decision_trace"]["decision"] == "action_outcome_feedback_recall"


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


def test_dialogue_without_explicit_memory_is_working_candidate_not_identity() -> None:
    candidate = MultimodalMemoryPolicy().classify_writeback_candidate(
        event_type="dialogue",
        summary="user:hello | reply:hi",
        modality="audio_text",
        organ="ear",
        success=True,
        status="planned",
        action_count=1,
        reply_present=True,
        trace_id="s1",
        source_event_id="asr-1",
    )

    assert candidate["candidate_types"] == ["working"]
    assert candidate["memory_type"] == "conversation"
    assert candidate["source"] == "eibrain.audio_dialogue"
    assert candidate["retention"] == "short_lived"
    assert candidate["meta"]["trace_id"] == "s1"
    assert candidate["meta"]["source_event_id"] == "asr-1"
    assert candidate["meta"]["identity_memory"] is False
    assert candidate["meta"]["persona_memory"] is False
    assert "eibrain.identity" not in candidate["source"]
    assert candidate["writeback"]["eligible"] is False
    assert candidate["meta"]["dedupe_key"].startswith("conversation:audio_text:ear:")
    assert candidate["meta"]["privacy"]["sensitivity"] == "personal"
    assert candidate["meta"]["decision_trace"]["decision"] == "writeback_conversation_working_only"


def test_explicit_preference_dialogue_is_semantic_candidate_not_identity() -> None:
    candidate = MultimodalMemoryPolicy().classify_writeback_candidate(
        event_type="dialogue",
        summary="user:记住我喜欢简短回复 | reply:好的",
        modality="audio_text",
        organ="ear",
        explicit_memory_request=True,
        success=True,
        status="planned",
        action_count=1,
        reply_present=True,
        trace_id="s2",
        source_event_id="asr-2",
    )

    assert candidate["candidate_types"] == ["semantic"]
    assert candidate["memory_type"] == "semantic_candidate"
    assert candidate["source"] == "eibrain.semantic_candidate"
    assert candidate["promotion_status"] == "candidate"
    assert candidate["meta"]["identity_memory"] is False
    assert candidate["meta"]["durable_identity_allowed"] is False
    assert "semantic_candidate" in candidate["tags"]
    assert candidate["writeback"] == {
        "eligible": True,
        "reason": "explicit_memory_request",
        "target_memory_type": "semantic_candidate",
    }
    assert candidate["meta"]["privacy"]["scope"] == "subject_conversation"
    assert candidate["meta"]["sensitivity"] == "personal"
    assert candidate["meta"]["decision_trace"]["why"] == "explicit user memory request, but not durable identity"


def test_dialogue_summary_is_durable_episode_candidate() -> None:
    candidate = MultimodalMemoryPolicy().classify_writeback_candidate(
        event_type="dialogue_summary",
        summary="用户偏好简短回复，并希望以后保持这个风格。",
        modality="audio_text",
        organ="ear",
        success=True,
        status="planned",
        action_count=1,
        reply_present=True,
        trace_id="summary-1",
        source_event_id="turn-summary-1",
    )

    assert candidate["candidate_types"] == ["episodic"]
    assert candidate["memory_type"] == "conversation"
    assert candidate["retention"] == "episode"
    assert candidate["promotion_status"] == "not_promoted"
    assert candidate["writeback"] == {
        "eligible": True,
        "reason": "dialogue_summary",
        "target_memory_type": "conversation",
    }
    assert candidate["meta"]["decision_trace"]["decision"] == "writeback_dialogue_summary_episode"


def test_visual_observation_is_episodic_candidate_with_source_trace_metadata() -> None:
    candidate = MultimodalMemoryPolicy().classify_writeback_candidate(
        event_type="vision_observation",
        summary="person standing near desk",
        modality="vision",
        organ="eye",
        source="eibrain.visual_frame",
        visual_context={"target_x": 0.7, "summary": "person standing near desk"},
        trace_id="vision:user-1",
        source_event_id="frame-1",
    )

    assert candidate["candidate_types"] == ["episodic"]
    assert candidate["memory_type"] == "world_observation"
    assert candidate["source"] == "eibrain.visual_frame"
    assert candidate["meta"]["trace_id"] == "vision:user-1"
    assert candidate["content"]["visual_context"] == {"target_x": 0.7, "summary": "person standing near desk"}
    assert candidate["meta"]["identity_memory"] is False
    assert "vision" in candidate["tags"]
    assert candidate["writeback"]["eligible"] is True
    assert candidate["meta"]["dedupe_key"].startswith("world_observation:vision:eye:")
    assert candidate["meta"]["privacy"]["sensitivity"] == "environmental"
    assert candidate["meta"]["decision_trace"]["decision"] == "writeback_visual_world_observation"


def test_visual_frame_is_trace_only_and_not_durable_memory() -> None:
    candidate = MultimodalMemoryPolicy().classify_writeback_candidate(
        event_type="visual_frame",
        summary="raw frame 42 with transient detections",
        modality="vision",
        organ="eye",
        source="eibrain.visual_frame",
        trace_id="vision-frame-42",
        source_event_id="frame-42",
    )

    assert candidate["candidate_types"] == ["working"]
    assert candidate["memory_type"] == "working_event"
    assert candidate["retention"] == "short_lived"
    assert candidate["writeback"] == {
        "eligible": False,
        "reason": "high_frequency_visual_frame",
        "target_memory_type": "working_event",
    }
    assert candidate["meta"]["decision_trace"]["decision"] == "writeback_visual_frame_trace_only"


def test_action_outcome_feedback_is_procedural_and_training_candidate() -> None:
    candidate = MultimodalMemoryPolicy().classify_writeback_candidate(
        event_type="action_outcome",
        summary="MoveHeadAction failed with oscillation",
        modality="multimodal_action",
        organ="neck",
        success=False,
        status="failed",
        action_count=1,
        user_feedback="tracking looked unstable",
        suggested_adjustment="increase yaw deadband before moving",
        trace_id="trace-neck-2",
        source_event_id="outcome-2",
    )

    assert candidate["candidate_types"] == ["procedural", "training"]
    assert candidate["memory_type"] == "procedural_adjustment_candidate"
    assert candidate["source"] == "eibrain.procedural_feedback"
    assert candidate["retention"] == "adjustment_candidate"
    assert candidate["promotion_status"] == "candidate"
    assert candidate["training_candidate"] is True
    assert candidate["meta"]["trace_id"] == "trace-neck-2"
    assert candidate["content"]["suggested_adjustment"] == "increase yaw deadband before moving"
    assert candidate["meta"]["identity_memory"] is False
    assert candidate["writeback"] == {
        "eligible": True,
        "reason": "procedural_adjustment",
        "target_memory_type": "procedural_adjustment_candidate",
    }
    assert candidate["meta"]["dedupe_key"].startswith("procedural_adjustment_candidate:multimodal_action:neck:")
    assert candidate["meta"]["privacy"]["sensitivity"] == "operational"
    assert candidate["meta"]["decision_trace"]["decision"] == "writeback_procedural_training_candidate"


def test_user_feedback_without_adjustment_is_training_candidate_not_identity() -> None:
    candidate = MultimodalMemoryPolicy().classify_writeback_candidate(
        event_type="user_feedback",
        summary="reply was too long",
        modality="audio_text",
        organ="ear",
        user_feedback="reply was too long",
        trace_id="feedback-1",
        source_event_id="thumbs-down-1",
    )

    assert candidate["candidate_types"] == ["training"]
    assert candidate["memory_type"] == "training_candidate"
    assert candidate["source"] == "eibrain.training_candidate"
    assert candidate["retention"] == "training_candidate"
    assert candidate["training_candidate"] is True
    assert candidate["meta"]["identity_memory"] is False
    assert candidate["content"]["user_feedback"] == "reply was too long"


def test_cognitive_runtime_audio_writeback_attaches_candidate_metadata() -> None:
    from apps.cognitive_runtime.app import CognitiveRuntimeApp
    from eibrain.memory.contracts import MemoryResult
    from eibrain.protocol.observations import AudioTranscriptFinal

    class _MemoryAdapter:
        last_writeback_status = {"status": "idle"}

        def __init__(self) -> None:
            self.remembered: list[dict[str, object]] = []

        def retrieve_context(self, query):
            return MemoryResult(summary="")

        def remember_episode(self, **kwargs):
            self.remembered.append(dict(kwargs))
            self.last_writeback_status = {"status": "ok", "source": kwargs.get("source")}

    runtime = CognitiveRuntimeApp()
    runtime.memory = _MemoryAdapter()
    runtime.handle_observation(
        AudioTranscriptFinal(ts=1.0, source="ear.asr", text="记住我喜欢简短回复", session_id="s1", actor_id="user-1")
    )

    payload = runtime.memory.remembered[0]
    assert payload["memory_type"] == "semantic_candidate"
    assert payload["source"] == "eibrain.audio_dialogue"
    assert payload["meta"]["candidate_types"] == ["semantic"]
    assert payload["meta"]["identity_memory"] is False
    assert payload["meta"]["trace_id"] == "s1"
    assert payload["content"]["event_type"] == "dialogue"
    assert "semantic_candidate" in payload["tags"]


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


def test_cognitive_runtime_skips_duplicate_world_observation() -> None:
    from apps.cognitive_runtime.app import CognitiveRuntimeApp

    class _Memory:
        last_writeback_status = {"status": "idle"}

        def __init__(self) -> None:
            self.captured: list[dict[str, object]] = []

        def remember_world_observation(self, **kwargs) -> None:
            self.captured.append(kwargs)
            self.last_writeback_status = {"status": "ok", "source": "eibrain.visual_world"}

    runtime = CognitiveRuntimeApp()
    memory = _Memory()
    runtime.memory = memory
    visual_state = {
        "backend": "gstreamer_hailo",
        "objects": [{"label": "cup", "score": 0.8, "bbox": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.2, "y_max": 0.2}}],
    }

    runtime.remember_world_observation_from_state(visual_state, session_id="vision:desk")
    runtime.remember_world_observation_from_state(visual_state, session_id="vision:desk")

    assert len(memory.captured) == 1
    assert runtime.last_memory_diagnostics["last_writeback"]["status"] == "skipped"
    assert runtime.last_memory_diagnostics["last_writeback"]["reason"] == "unchanged_world_observation"

