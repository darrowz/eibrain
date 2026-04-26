"""Multimodal memory policy for the Hongtu embodied subject."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MultimodalMemoryPolicy:
    """Build recall/writeback metadata without letting modalities pollute each other."""

    identity_sources: list[str] = field(
        default_factory=lambda: [
            "eibrain.identity",
            "eibrain.preference",
            "eibrain.dialogue",
            "eibrain.audio_dialogue",
            "openclaw.agent_end",
        ]
    )
    blocked_general_sources: list[str] = field(default_factory=lambda: ["eimemory.knowledge.claims"])

    def build_recall_context(
        self,
        *,
        task_type: str,
        modality: str,
        organ: str,
        phase: str,
        salience_score: float,
        body_capabilities: dict[str, bool],
        active_policy: dict[str, object] | None = None,
        query: str = "",
        visual_context: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        context: dict[str, Any] = {
            "task_type": task_type,
            "goal": "retrieve memory for embodied response",
            "modality": modality,
            "organ": organ,
            "phase": phase,
            "salience_score": round(float(salience_score), 3),
            "body_capabilities": dict(body_capabilities),
            "recall_profile": "precision",
        }
        if active_policy:
            context["active_policy"] = dict(active_policy)
        if visual_context:
            context["visual_context"] = dict(visual_context)

        lower_query = query.lower()
        if task_type == "brain.orient" or modality == "vision" or organ == "eye":
            context.update(
                {
                    "allowed_sources": [
                        "eibrain.visual_world",
                        "eibrain.visual_frame",
                        "eibrain.identity",
                        "eibrain.policy",
                    ],
                    "blocked_sources": [],
                    "allowed_memory_types": ["world_observation", "identity", "fact", "policy", "semantic"],
                    "preferred_modalities": ["vision", "multimodal"],
                    "organs": ["eye", "cognition"],
                    "source_weights": {
                        "eibrain.visual_world": 1.6,
                        "eibrain.visual_frame": 1.4,
                        "eibrain.identity": 1.3,
                    },
                }
            )
            return context

        if any(token in lower_query or token in query for token in ("asr", "tts", "语音", "麦克风", "摄像头", "视觉", "检测", "云台")):
            context.update(
                {
                    "allowed_sources": ["eibrain.audio_dialogue", "eibrain.visual_frame", "eibrain.policy", "eibrain.system"],
                    "blocked_sources": [],
                    "allowed_memory_types": ["conversation", "fact", "policy", "semantic"],
                    "preferred_modalities": ["audio_text", "vision", "multimodal", "system"],
                    "organs": ["ear", "eye", "mouth", "neck", "cognition"],
                    "source_weights": {"eibrain.policy": 1.6, "eibrain.audio_dialogue": 1.25},
                }
            )
            return context

        context.update(
            {
                "allowed_sources": list(self.identity_sources),
                "blocked_sources": list(self.blocked_general_sources),
                "allowed_memory_types": ["identity", "preference", "conversation", "semantic", "fact"],
                "preferred_modalities": ["audio_text", "multimodal", "text"],
                "organs": ["ear", "cognition"],
                "source_weights": {
                    "eibrain.identity": 1.8,
                    "eibrain.preference": 1.5,
                    "eibrain.audio_dialogue": 1.25,
                    "openclaw.agent_end": 1.15,
                },
            }
        )
        return context

    def writeback_outcome(
        self,
        *,
        modality: str,
        organ: str,
        success: bool,
        status: str,
        action_count: int,
        reply_present: bool,
        learning_decision: str,
        visual_context: dict[str, object] | None = None,
    ) -> dict[str, object]:
        outcome: dict[str, object] = {
            "success": success,
            "status": status,
            "action_count": action_count,
            "reply_present": reply_present,
            "learning_decision": learning_decision,
            "modality": modality,
            "organ": organ,
            "subject": "hongtu",
        }
        if visual_context:
            outcome["visual_context"] = dict(visual_context)
        return outcome
