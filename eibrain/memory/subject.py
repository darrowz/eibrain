"""Hongtu subject identity helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

HONGTU_SUBJECT_ID = "hongtu"
HONGTU_AGENT_ID = "hongtu"
HONGTU_WORKSPACE_ID = "embodied"
DEFAULT_HONGTU_USER_ID = "darrow"

_DEFAULT_TENANT_ID = "default"
_DEFAULT_CHANNEL_ID = "voice"
_FEISHU_OPEN_ID = "ou_644f810515d8ae7789de6a932d4de854"

_KNOWN_USER_ALIASES = {
    DEFAULT_HONGTU_USER_ID: DEFAULT_HONGTU_USER_ID,
    _FEISHU_OPEN_ID: DEFAULT_HONGTU_USER_ID,
}
_DEFAULT_USER_ALIASES = (DEFAULT_HONGTU_USER_ID, _FEISHU_OPEN_ID)

_SOURCE_MEMORY_LAYERS = {
    "openclaw.before_prompt_build": "trace",
    "ei_bridge.openclaw_feishu": "channel_audit",
    "openclaw.agent_end": "episodic",
    "openclaw.message_received": "episodic",
    "eibrain.identity": "identity_core",
    "eibrain.preference": "preference",
    "eibrain.dialogue": "episodic",
    "eibrain.audio_dialogue": "episodic",
    "eibrain.visual_world": "episodic",
    "eibrain.visual_frame": "trace",
    "eibrain.head_feedback": "operational_feedback",
    "eibrain.action_trace": "trace",
    "eibrain.outcome_feedback": "operational_feedback",
    "eibrain.procedural_feedback": "operational_feedback",
    "eibrain.training_candidate": "training",
    "eibrain.policy": "policy",
    "eibrain.system": "diagnostic",
    "eibrain.skill_trace": "trace",
    "eibrain.memory_trace": "trace",
    "eibrain.working_event": "trace",
}


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _unique_non_empty(values: tuple[str, ...] | list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        clean = _clean(value)
        if clean and clean not in seen:
            unique.append(clean)
            seen.add(clean)
    return unique


def canonical_user_id(actor_id: str | None = None) -> str:
    """Return Hongtu's canonical user id for any known or channel-specific actor."""

    actor = _clean(actor_id)
    return _KNOWN_USER_ALIASES.get(actor, DEFAULT_HONGTU_USER_ID)


def normalize_hongtu_scope(scope: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Collapse legacy/channel scopes into Hongtu's canonical memory scope."""

    return {
        "tenant_id": _DEFAULT_TENANT_ID,
        "agent_id": HONGTU_AGENT_ID,
        "workspace_id": HONGTU_WORKSPACE_ID,
        "user_id": canonical_user_id(_clean((scope or {}).get("user_id"))),
    }


def subject_context(
    *,
    channel_id: str | None = None,
    actor_id: str | None = None,
    session_id: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Build a serializable subject context while preserving channel actor aliases."""

    actor = _clean(actor_id)
    source_id = _clean(source)
    ctx: dict[str, Any] = {
        "subject_id": HONGTU_SUBJECT_ID,
        "channel_id": _clean(channel_id) or _DEFAULT_CHANNEL_ID,
        "canonical_user_id": canonical_user_id(actor),
        "user_aliases": _user_aliases(actor),
    }

    if actor:
        ctx["actor_id"] = actor
    if session := _clean(session_id):
        ctx["session_id"] = session
    if source_id:
        ctx["source"] = source_id
        ctx["memory_layer"] = classify_memory_layer(source_id)

    return ctx


def classify_memory_layer(source: str | None, memory_type: str | None = None) -> str:
    """Classify memory sources so traces/audits do not pollute persona recall."""

    source_id = _clean(source)
    if source_id in _SOURCE_MEMORY_LAYERS:
        return _SOURCE_MEMORY_LAYERS[source_id]
    if source_id.startswith("openclaw.before_"):
        return "trace"
    if source_id.startswith("ei_bridge."):
        return "channel_audit"
    if source_id.endswith(".identity"):
        return "identity_core"

    kind = _clean(memory_type)
    if kind in {"trace", "diagnostic", "debug", "working_event"}:
        return "trace"
    if kind in {"audit", "channel_audit"}:
        return "channel_audit"
    if kind in {"identity", "identity_core", "profile"}:
        return "identity_core"
    if kind == "preference":
        return "preference"
    if kind in {"action_outcome", "head_execution_feedback", "procedural_adjustment_candidate"}:
        return "operational_feedback"
    if kind == "training_candidate":
        return "training"
    if kind == "policy":
        return "policy"
    return "episodic"


def _user_aliases(actor_id: str) -> list[str]:
    aliases = list(_DEFAULT_USER_ALIASES)
    if actor_id:
        aliases.append(actor_id)
    return _unique_non_empty(aliases)


__all__ = [
    "DEFAULT_HONGTU_USER_ID",
    "HONGTU_AGENT_ID",
    "HONGTU_SUBJECT_ID",
    "HONGTU_WORKSPACE_ID",
    "canonical_user_id",
    "classify_memory_layer",
    "normalize_hongtu_scope",
    "subject_context",
]
