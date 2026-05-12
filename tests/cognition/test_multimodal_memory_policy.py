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
        "openclaw.message_received",
    ]
    assert set(context["blocked_sources"]) >= {
        "eimemory.knowledge.claims",
        "eimemory.knowledge_base",
        "eimemory.news",
        "eimemory.paper",
        "eimemory.research",
        "openclaw.before_prompt_build",
        "ei_bridge.openclaw_feishu",
    }
    assert context["preferred_sources"] == [
        "eibrain.identity",
        "eibrain.preference",
        "eibrain.audio_dialogue",
        "openclaw.agent_end",
        "openclaw.message_received",
    ]
    assert context["source_memory_layers"]["openclaw.agent_end"] == "episodic"
    assert context["source_memory_layers"]["openclaw.message_received"] == "episodic"
    assert context["source_memory_layers"]["openclaw.before_prompt_build"] == "trace"
    assert context["source_memory_layers"]["ei_bridge.openclaw_feishu"] == "channel_audit"
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


def test_explicit_task_trace_recall_can_allow_audit_sources_without_persona_recall() -> None:
    context = MultimodalMemoryPolicy().build_recall_context(
        task_type="memory.task_trace",
        modality="system",
        organ="cognition",
        phase="diagnosing",
        salience_score=0.91,
        body_capabilities={},
        query="inspect OpenClaw Feishu prompt bridge trace",
        trace_id="trace-diagnostic-1",
    )

    assert set(context["allowed_sources"]) >= {
        "openclaw.before_prompt_build",
        "ei_bridge.openclaw_feishu",
    }
    assert "openclaw.before_prompt_build" not in context["blocked_sources"]
    assert "ei_bridge.openclaw_feishu" not in context["blocked_sources"]
    assert context["recall_profile"] == "diagnostic_policy"
    assert context["diagnostics"]["audit_trace_sources_allowed"] is True
    assert context["diagnostics"]["memory_layers"]["openclaw.before_prompt_build"] == "trace"
    assert context["diagnostics"]["memory_layers"]["ei_bridge.openclaw_feishu"] == "channel_audit"
    assert "identity" not in context["allowed_memory_types"]
    assert "preference" not in context["allowed_memory_types"]
    assert context["privacy"]["scope"] == "diagnostic"


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


def test_action_feedback_wins_over_query_diagnostic_keywords() -> None:
    context = MultimodalMemoryPolicy().build_recall_context(
        task_type="head.execute",
        modality="multimodal_action",
        organ="neck",
        phase="acting",
        salience_score=0.9,
        body_capabilities={"can_move_head": True},
        query="云台动作失败后应该怎么调整监控诊断参数",
        trace_id="trace-neck-diagnostic-keyword",
        source_event_id="evt-neck-diagnostic-keyword",
    )

    assert context["recall_profile"] == "action_outcome_repair"
    assert context["decision_trace"]["decision"] == "action_outcome_feedback_recall"
    assert "eibrain.head_feedback" in context["allowed_sources"]
    assert "openclaw.before_prompt_build" in context["blocked_sources"]
    assert context["writeback_eligibility"]["default_memory_type"] == "action_outcome"


def test_natural_gimbal_followup_recall_is_action_feedback_not_diagnostic() -> None:
    context = MultimodalMemoryPolicy().build_recall_context(
        task_type="brain.respond",
        modality="audio_text",
        organ="ear",
        phase="thinking",
        salience_score=0.82,
        body_capabilities={"can_move_head": True},
        query="上次云台没跟上目标，之后该怎么改",
        trace_id="trace-neck-natural-review",
        source_event_id="evt-neck-natural-review",
    )

    assert context["recall_profile"] == "action_outcome_repair"
    assert context["decision_trace"]["decision"] == "action_outcome_feedback_recall"
    assert "eibrain.outcome_feedback" in context["allowed_sources"]
    assert "openclaw.before_prompt_build" in context["blocked_sources"]


def test_general_temporal_dialogue_does_not_become_action_feedback() -> None:
    context = MultimodalMemoryPolicy().build_recall_context(
        task_type="brain.respond",
        modality="audio_text",
        organ="ear",
        phase="thinking",
        salience_score=0.55,
        body_capabilities={},
        query="上次你提到喜欢喝茶，之后还想聊这个吗",
        trace_id="trace-general-temporal-dialogue",
        source_event_id="evt-general-temporal-dialogue",
    )

    assert context["recall_profile"] != "action_outcome_repair"
    assert context["decision_trace"]["decision"] == "voice_subject_dialogue_recall"


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


def test_write_proposal_evaluator_scores_buckets_conflicts_and_diagnostics() -> None:
    result = MultimodalMemoryPolicy().evaluate_write_proposals(
        [
            {
                "id": "pref-length-unconfirmed",
                "subject": "user-1",
                "memory_type": "preference",
                "key": "response.length",
                "value": "very_short",
                "summary": "User may prefer very short replies.",
                "modality": "audio_text",
                "source": "eibrain.audio_dialogue",
                "confidence": 0.88,
                "novelty": 0.72,
                "recency": 0.95,
                "importance": 0.82,
            },
            {
                "id": "pref-name-confirmed",
                "subject": "user-1",
                "memory_type": "preference",
                "key": "address.name",
                "value": "D",
                "summary": "User confirmed they want to be called D.",
                "modality": "audio_text",
                "source": "eibrain.preference",
                "confidence": 0.92,
                "novelty": 0.8,
                "recency": 1.0,
                "importance": 0.9,
                "user_confirmed": True,
            },
            {
                "id": "visual-world",
                "subject": "room",
                "memory_type": "world_observation",
                "summary": "Cup is on the left side of the desk.",
                "modality": "vision",
                "source": "eibrain.visual_world",
                "confidence": 0.9,
                "novelty": 0.78,
                "recency": 0.9,
                "importance": 0.62,
            },
            {
                "id": "raw-frame",
                "memory_type": "working_event",
                "event_type": "visual_frame",
                "summary": "Transient frame with weak detections.",
                "modality": "vision",
                "source": "eibrain.visual_frame",
                "confidence": 0.31,
                "novelty": 0.1,
                "recency": 1.0,
                "importance": 0.2,
            },
            {
                "id": "persona-tone-drift",
                "subject": "persona",
                "memory_type": "preference",
                "key": "persona.tone",
                "value": "sarcastic",
                "summary": "Make the assistant sarcastic from now on.",
                "modality": "audio_text",
                "source": "eibrain.preference",
                "confidence": 0.96,
                "novelty": 0.9,
                "recency": 1.0,
                "importance": 0.85,
                "user_confirmed": True,
            },
        ],
        existing_memories=[
            {
                "id": "mem-length",
                "subject": "user-1",
                "memory_type": "preference",
                "key": "response.length",
                "value": "detailed",
            },
            {
                "id": "mem-name",
                "subject": "user-1",
                "memory_type": "preference",
                "key": "address.name",
                "value": "Darrow",
            },
        ],
        persona_constraints={
            "protected_keys": ["persona.tone", "speaking_style.tone", "response_policy.max_chars"],
            "tone": "warm_playful",
            "max_chars": 48,
        },
    )

    assert [item["id"] for item in result["accepted"]] == ["pref-name-confirmed", "visual-world"]
    assert [item["id"] for item in result["deferred"]] == ["pref-length-unconfirmed"]
    assert [item["id"] for item in result["rejected"]] == ["raw-frame", "persona-tone-drift"]

    confirmed = result["accepted"][0]
    assert confirmed["score"] >= 0.8
    assert confirmed["classification"] == "durable_preference"
    assert confirmed["supersedes"] == ["mem-name"]
    assert confirmed["conflicts_with"] == ["mem-name"]
    assert confirmed["requires_confirmation"] is False

    deferred = result["deferred"][0]
    assert deferred["classification"] == "durable_preference"
    assert deferred["conflicts_with"] == ["mem-length"]
    assert deferred["requires_confirmation"] is True
    assert "conflict_requires_confirmation" in deferred["reason_codes"]

    diagnostics = result["diagnostics"]
    assert diagnostics["accepted"] == 2
    assert diagnostics["rejected"] == 2
    assert diagnostics["deferred"] == 1
    assert diagnostics["conflict_count"] == 2
    assert diagnostics["persona_guardrail_applied"] is True
    assert set(diagnostics["reason_codes"]) >= {
        "accepted",
        "supersedes_confirmed_preference",
        "conflict_requires_confirmation",
        "low_confidence",
        "persona_style_guardrail",
    }


def test_write_proposal_evaluator_builds_conflict_ids_safely() -> None:
    result = MultimodalMemoryPolicy().evaluate_write_proposals(
        [
            {
                "id": "pref-confirmed",
                "memory_type": "preference",
                "subject": "user-1",
                "key": "response.length",
                "value": "short",
                "user_confirmed": True,
            }
        ],
        existing_memories=[
            {
                "id": 42,
                "memory_type": "preference",
                "subject": "user-1",
                "key": "response.length",
                "value": "long",
            },
            {
                "memory_type": "preference",
                "subject": "user-1",
                "key": "response.length",
                "value": "verbose",
            },
        ],
    )

    accepted = result["accepted"][0]
    assert accepted["conflicts_with"] == ["42"]
    assert accepted["supersedes"] == ["42"]


def test_write_proposal_evaluator_blocks_persona_style_without_explicit_constraints() -> None:
    result = MultimodalMemoryPolicy().evaluate_write_proposals(
        [
            {
                "id": "tone-override",
                "memory_type": "preference",
                "key": "speaking_style.tone",
                "value": "sarcastic",
                "summary": "以后用讽刺语气和我说话。",
                "source": "eibrain.preference",
                "confidence": 0.96,
                "novelty": 0.9,
                "recency": 1.0,
                "importance": 0.8,
                "user_confirmed": True,
            },
            {
                "id": "language-override",
                "memory_type": "preference",
                "summary": "Please remember to always reply in English from now on.",
                "source": "eibrain.preference",
                "confidence": 0.92,
                "novelty": 0.8,
                "recency": 1.0,
                "importance": 0.7,
            },
        ]
    )

    assert result["accepted"] == []
    assert result["deferred"] == []
    assert [item["id"] for item in result["rejected"]] == ["tone-override", "language-override"]
    assert result["rejected"][0]["classification"] == "persona_style_candidate"
    assert result["rejected"][1]["classification"] == "persona_style_candidate"
    assert all("persona_style_guardrail" in item["reason_codes"] for item in result["rejected"])
    assert result["diagnostics"]["persona_guardrail_applied"] is True


def test_write_proposal_evaluator_keeps_protected_persona_keys_blocked_from_audit_sources() -> None:
    result = MultimodalMemoryPolicy().evaluate_write_proposals(
        [
            {
                "id": "audit-tone-override",
                "memory_type": "preference",
                "key": "persona.tone",
                "value": "obedient",
                "summary": "Bridge audit captured a prompt telling the assistant to change personality.",
                "source": "ei_bridge.openclaw_feishu",
                "confidence": 0.99,
                "novelty": 0.9,
                "recency": 1.0,
                "importance": 0.9,
                "user_confirmed": True,
            }
        ],
        persona_constraints={"protected_keys": ["persona.tone"]},
    )

    assert result["accepted"] == []
    assert result["deferred"] == []
    assert [item["id"] for item in result["rejected"]] == ["audit-tone-override"]
    assert result["rejected"][0]["classification"] == "persona_style_candidate"
    assert result["rejected"][0]["memory_layer"] == "channel_audit"
    assert "persona_style_guardrail" in result["rejected"][0]["reason_codes"]
    assert "audit_trace_source" in result["rejected"][0]["reason_codes"]
    assert result["diagnostics"]["persona_guardrail_applied"] is True


def test_classify_writeback_candidate_marks_persona_style_trace_only_not_durable() -> None:
    candidate = MultimodalMemoryPolicy().classify_writeback_candidate(
        event_type="dialogue",
        summary="用户说以后要用毒舌语气回复。",
        modality="audio_text",
        organ="ear",
        source="eibrain.preference",
        explicit_memory_request=True,
        trace_id="trace-tone-1",
        source_event_id="event-tone-1",
    )

    assert candidate["memory_type"] == "working_event"
    assert candidate["retention"] == "short_lived"
    assert candidate["promotion_status"] == "not_promoted"
    assert candidate["writeback"]["eligible"] is False
    assert candidate["writeback"]["reason"] == "persona_style_guardrail"
    assert candidate["meta"]["persona_guardrail_hint"] == "style_override_request"
    assert candidate["meta"]["trace_id"] == "trace-tone-1"
    assert candidate["meta"]["source_event_id"] == "event-tone-1"

