from __future__ import annotations

from eibrain.memory.subject import (
    DEFAULT_HONGTU_USER_ID,
    HONGTU_AGENT_ID,
    HONGTU_SUBJECT_ID,
    HONGTU_WORKSPACE_ID,
    canonical_user_id,
    classify_memory_layer,
    normalize_hongtu_scope,
    subject_context,
)


def test_normalize_hongtu_scope_defaults_to_single_subject() -> None:
    scope = normalize_hongtu_scope({"agent_id": "honxin", "workspace_id": "honjia"})

    assert scope == {
        "tenant_id": "default",
        "agent_id": HONGTU_AGENT_ID,
        "workspace_id": HONGTU_WORKSPACE_ID,
        "user_id": DEFAULT_HONGTU_USER_ID,
    }


def test_subject_context_preserves_channel_aliases() -> None:
    open_id = "ou_644f810515d8ae7789de6a932d4de854"

    ctx = subject_context(
        channel_id="feishu",
        actor_id=open_id,
        session_id="sess-1",
        source="openclaw.message_received",
    )

    assert ctx["subject_id"] == HONGTU_SUBJECT_ID
    assert ctx["channel_id"] == "feishu"
    assert ctx["canonical_user_id"] == DEFAULT_HONGTU_USER_ID
    assert open_id in ctx["user_aliases"]


def test_classify_memory_layer_keeps_audits_out_of_persona() -> None:
    assert classify_memory_layer("openclaw.before_prompt_build", "conversation") == "trace"
    assert classify_memory_layer("ei_bridge.openclaw_feishu", "audit") == "channel_audit"
    assert classify_memory_layer("openclaw.agent_end", "conversation") == "episodic"
    assert classify_memory_layer("eibrain.identity", "profile") == "identity_core"
    assert classify_memory_layer("eibrain.preference", "preference") == "preference"
    assert classify_memory_layer("eibrain.outcome_feedback", "action_outcome") == "operational_feedback"
    assert classify_memory_layer("eibrain.training_candidate", "training_candidate") == "training"
    assert classify_memory_layer("eibrain.policy", "policy") == "policy"
    assert classify_memory_layer("eibrain.working_event", "working_event") == "trace"


def test_unknown_actor_defaults_to_darrow_but_remains_alias() -> None:
    actor_id = "unknown-channel-user"

    assert canonical_user_id(actor_id) == DEFAULT_HONGTU_USER_ID
    assert actor_id in subject_context(channel_id="voice", actor_id=actor_id)["user_aliases"]
