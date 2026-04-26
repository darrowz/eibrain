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

    assert context["allowed_sources"] == ["eibrain.visual_frame", "eibrain.identity", "eibrain.policy"]
    assert context["preferred_modalities"] == ["vision", "multimodal"]
    assert context["visual_context"] == {"target_x": 0.42}


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
