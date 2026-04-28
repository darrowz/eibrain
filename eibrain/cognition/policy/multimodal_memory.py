"""Multimodal memory policy for the Hongtu embodied subject."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MultimodalMemoryPolicy:
    """Build recall/writeback metadata without letting modalities pollute each other."""

    source_system: str = "eibrain"
    channel_id: str = "voice.honjia"
    agent_id: str = "eibrain.voice"
    contract_version: str = "multimodal-memory.v1"
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
            "source_system": self.source_system,
            "channel_id": self.channel_id,
            "agent_id": self.agent_id,
            "channel_owner": "eibrain",
            "agent_owner": "eibrain",
            "memory_owner": "eimemory",
            "memory_contract_version": self.contract_version,
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
                    "recall_filters": self._recall_filters(
                        channel_id=self.channel_id,
                        agent_id=self.agent_id,
                        memory_types=["world_observation", "identity", "fact", "policy", "semantic"],
                    ),
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
                    "recall_filters": self._recall_filters(
                        channel_id=self.channel_id,
                        agent_id=self.agent_id,
                        memory_types=["conversation", "fact", "policy", "semantic"],
                    ),
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
                "recall_filters": self._recall_filters(
                    channel_id=self.channel_id,
                    agent_id=self.agent_id,
                    memory_types=["identity", "preference", "conversation", "semantic", "fact"],
                ),
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
            "source_system": self.source_system,
            "channel_id": self.channel_id,
            "agent_id": self.agent_id,
            "memory_contract_version": self.contract_version,
        }
        if visual_context:
            outcome["visual_context"] = dict(visual_context)
        return outcome

    def _recall_filters(self, *, channel_id: str, agent_id: str, memory_types: list[str]) -> dict[str, object]:
        return {
            "source_systems": [self.source_system, "openclaw"],
            "channel_ids": [channel_id, "global.profile", "global.summary"],
            "agent_ids": [agent_id, "openclaw.feishu"],
            "memory_types": list(memory_types),
        }
