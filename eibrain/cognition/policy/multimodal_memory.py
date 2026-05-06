"""Multimodal memory policy for the Hongtu embodied subject."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
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
    blocked_general_sources: list[str] = field(
        default_factory=lambda: [
            "eimemory.knowledge.claims",
            "eimemory.knowledge_base",
            "eimemory.news",
            "eimemory.paper",
            "eimemory.research",
        ]
    )

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
        trace_id: str = "",
        source_event_id: str = "",
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
        if trace_id:
            context["trace_id"] = trace_id
        if source_event_id:
            context["source_event_id"] = source_event_id

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
                    "blocked_sources": list(self.blocked_general_sources),
                    "allowed_memory_types": ["world_observation", "identity", "fact", "policy", "semantic"],
                    "preferred_modalities": ["vision", "multimodal"],
                    "organs": ["eye", "cognition"],
                    "recall_profile": "visual_grounding",
                    "privacy": self._privacy_context(scope="situational_awareness", sensitivity="environmental"),
                    "writeback_eligibility": {
                        "eligible": True,
                        "requires_explicit_memory_request": False,
                        "default_memory_type": "world_observation",
                    },
                    "decision_trace": {
                        "decision": "vision_world_grounding_recall",
                        "why": [
                            "prefer visual world and frame memory",
                            "avoid knowledge/news/paper sources",
                        ],
                    },
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

        if self._is_action_feedback_recall(task_type=task_type, modality=modality, organ=organ, query=lower_query):
            memory_types = [
                "head_execution_feedback",
                "action_outcome",
                "procedural_adjustment_candidate",
                "training_candidate",
                "policy",
                "semantic",
            ]
            context.update(
                {
                    "allowed_sources": [
                        "eibrain.head_feedback",
                        "eibrain.action_trace",
                        "eibrain.outcome_feedback",
                        "eibrain.procedural_feedback",
                        "eibrain.training_candidate",
                        "eibrain.policy",
                    ],
                    "blocked_sources": [*self.identity_sources, *self.blocked_general_sources],
                    "allowed_memory_types": memory_types,
                    "preferred_modalities": ["multimodal_action", "audio_text", "vision", "system"],
                    "organs": ["neck", "mouth", "eye", "ear", "cognition"],
                    "recall_profile": "action_outcome_repair",
                    "privacy": self._privacy_context(scope="operational_feedback", sensitivity="operational"),
                    "writeback_eligibility": {
                        "eligible": True,
                        "requires_explicit_memory_request": False,
                        "default_memory_type": "action_outcome",
                    },
                    "decision_trace": {
                        "decision": "action_outcome_feedback_recall",
                        "why": [
                            "retrieve action outcomes and procedural feedback",
                            "avoid identity/persona memory contamination",
                            "avoid knowledge/news/paper sources",
                        ],
                    },
                    "source_weights": {
                        "eibrain.head_feedback": 1.8,
                        "eibrain.procedural_feedback": 1.7,
                        "eibrain.outcome_feedback": 1.5,
                        "eibrain.policy": 1.2,
                    },
                    "recall_filters": self._recall_filters(
                        channel_id=self.channel_id,
                        agent_id=self.agent_id,
                        memory_types=memory_types,
                    ),
                }
            )
            return context

        if any(token in lower_query or token in query for token in ("asr", "tts", "语音", "麦克风", "摄像头", "视觉", "检测", "云台")):
            context.update(
                {
                    "allowed_sources": ["eibrain.audio_dialogue", "eibrain.visual_frame", "eibrain.policy", "eibrain.system"],
                    "blocked_sources": list(self.blocked_general_sources),
                    "allowed_memory_types": ["conversation", "fact", "policy", "semantic"],
                    "preferred_modalities": ["audio_text", "vision", "multimodal", "system"],
                    "organs": ["ear", "eye", "mouth", "neck", "cognition"],
                    "recall_profile": "diagnostic_policy",
                    "privacy": self._privacy_context(scope="diagnostic", sensitivity="operational"),
                    "writeback_eligibility": {
                        "eligible": False,
                        "requires_explicit_memory_request": True,
                        "default_memory_type": "training_candidate",
                    },
                    "decision_trace": {
                        "decision": "diagnostic_policy_recall",
                        "why": [
                            "retrieve policy and system diagnostics",
                            "avoid knowledge/news/paper sources",
                        ],
                    },
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
                "recall_profile": "subject_dialogue" if self._is_subject_memory_query(lower_query) else "precision",
                "privacy": self._privacy_context(scope="subject_conversation", sensitivity="personal"),
                "writeback_eligibility": {
                    "eligible": True,
                    "requires_explicit_memory_request": True,
                    "default_memory_type": "conversation",
                },
                "decision_trace": {
                    "decision": "voice_subject_dialogue_recall",
                    "why": [
                        "retrieve Hongtu subject dialogue memory",
                        "avoid knowledge/news/paper sources",
                    ],
                },
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
        success: bool | None,
        status: str,
        action_count: int,
        reply_present: bool,
        learning_decision: str,
        visual_context: dict[str, object] | None = None,
        trace_id: str = "",
        source_event_id: str = "",
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
        if trace_id:
            outcome["trace_id"] = trace_id
        if source_event_id:
            outcome["source_event_id"] = source_event_id
        return outcome

    def classify_writeback_candidate(
        self,
        *,
        event_type: str,
        summary: str,
        modality: str,
        organ: str,
        source: str = "",
        success: bool | None = None,
        status: str = "",
        action_count: int = 0,
        reply_present: bool | None = None,
        learning_decision: str = "keep_policy",
        user_feedback: str = "",
        suggested_adjustment: str = "",
        explicit_memory_request: bool = False,
        trace_id: str = "",
        source_event_id: str = "",
        visual_context: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Classify multimodal writeback without promoting it to identity memory."""

        event = self._clean_text(event_type).lower() or "event"
        cleaned_summary = self._clean_text(summary) or f"{event} event"
        feedback = self._clean_text(user_feedback)
        adjustment = self._clean_text(suggested_adjustment)
        status_lower = self._clean_text(status).lower()

        if adjustment:
            candidate_types = ["procedural", "training"]
            memory_type = "procedural_adjustment_candidate"
            resolved_source = source or "eibrain.procedural_feedback"
            retention = "adjustment_candidate"
            promotion_status = "candidate"
        elif event in {"dialogue_summary", "conversation_summary", "turn_summary"}:
            candidate_types = ["episodic"]
            memory_type = "conversation"
            resolved_source = source or "eibrain.audio_dialogue"
            retention = "episode"
            promotion_status = "not_promoted"
        elif event in {"frame", "visual_frame", "video_frame", "detection", "detections", "object_detected"}:
            candidate_types = ["working"]
            memory_type = "working_event"
            resolved_source = source or "eibrain.visual_frame"
            retention = "short_lived"
            promotion_status = "not_promoted"
        elif event in {"vision_observation", "visual_observation", "world_observation"} or modality == "vision":
            candidate_types = ["episodic"]
            memory_type = "world_observation"
            resolved_source = source or "eibrain.visual_world"
            retention = "episode"
            promotion_status = "not_promoted"
        elif explicit_memory_request or self._looks_like_semantic_candidate(cleaned_summary):
            candidate_types = ["semantic"]
            memory_type = "semantic_candidate"
            resolved_source = self._non_identity_write_source(source) or "eibrain.semantic_candidate"
            retention = "semantic_candidate"
            promotion_status = "candidate"
        elif feedback or event in {"user_feedback", "feedback", "dialogue_feedback"}:
            candidate_types = ["training"]
            memory_type = "training_candidate"
            resolved_source = source or "eibrain.training_candidate"
            retention = "training_candidate"
            promotion_status = "candidate"
        elif event == "dialogue":
            candidate_types = ["working"]
            memory_type = "conversation"
            resolved_source = source or "eibrain.audio_dialogue"
            retention = "short_lived"
            promotion_status = "not_promoted"
        elif self._is_action_outcome_event(event=event, status=status_lower, success=success, action_count=action_count):
            candidate_types = ["episodic"]
            if feedback or success is False:
                candidate_types.append("training")
            memory_type = "action_outcome"
            resolved_source = source or "eibrain.outcome_feedback"
            retention = "episode"
            promotion_status = "candidate" if "training" in candidate_types else "not_promoted"
        else:
            candidate_types = ["working"]
            memory_type = "working_event"
            resolved_source = source or ("eibrain.audio_dialogue" if modality == "audio_text" else "eibrain.working_event")
            retention = "short_lived"
            promotion_status = "not_promoted"

        training_candidate = "training" in candidate_types
        privacy = self._candidate_privacy(memory_type=memory_type, modality=modality, organ=organ)
        dedupe_key = self._dedupe_key(
            memory_type=memory_type,
            modality=modality,
            organ=organ,
            summary=cleaned_summary,
            source_event_id=source_event_id,
        )
        writeback = self._writeback_policy(
            event_type=event,
            memory_type=memory_type,
            explicit_memory_request=explicit_memory_request,
            training_candidate=training_candidate,
            adjustment=adjustment,
        )
        decision_trace = self._candidate_decision_trace(
            event_type=event,
            memory_type=memory_type,
            candidate_types=candidate_types,
            explicit_memory_request=explicit_memory_request,
            adjustment=adjustment,
        )
        content: dict[str, object] = {
            "event_type": event,
            "summary": cleaned_summary,
            "modality": modality,
            "organ": organ,
            "success": success,
            "status": status,
            "action_count": action_count,
        }
        if reply_present is not None:
            content["reply_present"] = reply_present
        if feedback:
            content["user_feedback"] = feedback
        if adjustment:
            content["suggested_adjustment"] = adjustment
        if visual_context:
            content["visual_context"] = dict(visual_context)

        meta = {
            "source_system": self.source_system,
            "channel_id": self.channel_id,
            "agent_id": self.agent_id,
            "memory_contract_version": self.contract_version,
            "event_type": event,
            "source": resolved_source,
            "trace_id": trace_id,
            "source_event_id": source_event_id,
            "candidate_types": list(candidate_types),
            "memory_kind": candidate_types[0],
            "retention": retention,
            "promotion_status": promotion_status,
            "training_candidate": training_candidate,
            "persona_memory": False,
            "identity_memory": False,
            "durable_identity_allowed": False,
            "dedupe_key": dedupe_key,
            "privacy": privacy,
            "sensitivity": privacy["sensitivity"],
            "decision_trace": decision_trace,
        }
        outcome = self.writeback_outcome(
            modality=modality,
            organ=organ,
            success=success,
            status=status or "unknown",
            action_count=action_count,
            reply_present=bool(reply_present),
            learning_decision=learning_decision,
            visual_context=visual_context,
            trace_id=trace_id,
            source_event_id=source_event_id,
        )
        return {
            "summary": cleaned_summary,
            "candidate_types": list(candidate_types),
            "memory_type": memory_type,
            "source": resolved_source,
            "modality": modality,
            "organ": organ,
            "retention": retention,
            "promotion_status": promotion_status,
            "training_candidate": training_candidate,
            "writeback": writeback,
            "content": content,
            "meta": meta,
            "outcome": outcome,
            "tags": self._candidate_tags(
                event_type=event,
                memory_type=memory_type,
                modality=modality,
                organ=organ,
                candidate_types=candidate_types,
                training_candidate=training_candidate,
            ),
        }

    def _recall_filters(self, *, channel_id: str, agent_id: str, memory_types: list[str]) -> dict[str, object]:
        return {
            "source_systems": [self.source_system, "openclaw"],
            "channel_ids": [channel_id, "global.profile", "global.summary"],
            "agent_ids": [agent_id, "openclaw.feishu"],
            "memory_types": list(memory_types),
        }

    def _is_action_feedback_recall(self, *, task_type: str, modality: str, organ: str, query: str) -> bool:
        if task_type in {"head.execute", "brain.act", "brain.feedback", "brain.outcome"}:
            return True
        if modality in {"multimodal_action", "action", "system"} or organ in {"neck", "mouth"}:
            return True
        return any(
            token in query
            for token in (
                "action",
                "outcome",
                "feedback",
                "failed",
                "failure",
                "adjust",
                "retry",
                "执行",
                "动作",
                "失败",
                "调整",
                "反馈",
            )
        )

    @staticmethod
    def _is_subject_memory_query(query: str) -> bool:
        return any(
            token in query
            for token in (
                "remember",
                "preference",
                "prefer",
                "like",
                "记得",
                "记住",
                "喜欢",
                "偏好",
                "刚才",
            )
        )

    @staticmethod
    def _is_action_outcome_event(*, event: str, status: str, success: bool | None, action_count: int) -> bool:
        if event in {"action", "action_outcome", "outcome", "execution_outcome", "head_feedback"}:
            return True
        if action_count > 0 or success is False:
            return True
        return status in {"failed", "failure", "error", "timeout", "aborted"}

    @staticmethod
    def _looks_like_semantic_candidate(summary: str) -> bool:
        lowered = summary.lower()
        return any(
            token in lowered
            for token in (
                "remember that",
                "remember this",
                "keep in mind",
                "note that",
                "记住",
                "记得",
                "喜欢",
                "偏好",
                "prefer",
            )
        )

    @staticmethod
    def _privacy_context(*, scope: str, sensitivity: str) -> dict[str, str]:
        return {
            "scope": scope,
            "sensitivity": sensitivity,
            "allowed_use": "embodied_response",
        }

    def _candidate_privacy(self, *, memory_type: str, modality: str, organ: str) -> dict[str, str]:
        if memory_type == "world_observation" or modality == "vision" or organ == "eye":
            return self._privacy_context(scope="situational_awareness", sensitivity="environmental")
        if memory_type in {"action_outcome", "procedural_adjustment_candidate", "training_candidate"} or modality in {
            "multimodal_action",
            "action",
            "system",
        }:
            return self._privacy_context(scope="operational_feedback", sensitivity="operational")
        return self._privacy_context(scope="subject_conversation", sensitivity="personal")

    @staticmethod
    def _dedupe_key(*, memory_type: str, modality: str, organ: str, summary: str, source_event_id: str) -> str:
        basis = source_event_id or summary
        digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]
        return f"{memory_type}:{modality}:{organ}:{digest}"

    @staticmethod
    def _writeback_policy(
        *,
        event_type: str,
        memory_type: str,
        explicit_memory_request: bool,
        training_candidate: bool,
        adjustment: str,
    ) -> dict[str, object]:
        if adjustment:
            return {
                "eligible": True,
                "reason": "procedural_adjustment",
                "target_memory_type": memory_type,
            }
        if explicit_memory_request or memory_type == "semantic_candidate":
            return {
                "eligible": True,
                "reason": "explicit_memory_request",
                "target_memory_type": memory_type,
            }
        if memory_type == "world_observation":
            return {
                "eligible": True,
                "reason": "visual_world_observation",
                "target_memory_type": memory_type,
            }
        if event_type in {"dialogue_summary", "conversation_summary", "turn_summary"}:
            return {
                "eligible": True,
                "reason": "dialogue_summary",
                "target_memory_type": memory_type,
            }
        if event_type in {"frame", "visual_frame", "video_frame", "detection", "detections", "object_detected"}:
            return {
                "eligible": False,
                "reason": "high_frequency_visual_frame",
                "target_memory_type": memory_type,
            }
        if training_candidate:
            return {
                "eligible": True,
                "reason": "training_feedback",
                "target_memory_type": memory_type,
            }
        return {
            "eligible": False,
            "reason": "working_memory_only",
            "target_memory_type": memory_type,
        }

    @staticmethod
    def _candidate_decision_trace(
        *,
        event_type: str,
        memory_type: str,
        candidate_types: list[str],
        explicit_memory_request: bool,
        adjustment: str,
    ) -> dict[str, object]:
        if adjustment:
            return {
                "decision": "writeback_procedural_training_candidate",
                "why": "action outcome includes a suggested procedural adjustment",
            }
        if explicit_memory_request or memory_type == "semantic_candidate":
            return {
                "decision": "writeback_semantic_candidate",
                "why": "explicit user memory request, but not durable identity",
            }
        if memory_type == "world_observation":
            return {
                "decision": "writeback_visual_world_observation",
                "why": "vision observation can be retained as situational episodic memory",
            }
        if event_type in {"dialogue_summary", "conversation_summary", "turn_summary"}:
            return {
                "decision": "writeback_dialogue_summary_episode",
                "why": "conversation summaries are durable episodic memory, unlike raw dialogue turns",
            }
        if event_type in {"frame", "visual_frame", "video_frame", "detection", "detections", "object_detected"}:
            return {
                "decision": "writeback_visual_frame_trace_only",
                "why": "raw high-frequency visual frames stay in trace/working memory, not durable memory",
            }
        if "training" in candidate_types:
            return {
                "decision": "writeback_training_candidate",
                "why": "feedback can improve future embodied behavior",
            }
        if memory_type == "conversation":
            return {
                "decision": "writeback_conversation_working_only",
                "why": "dialogue is working memory unless the user explicitly asks to remember it",
            }
        return {
            "decision": "writeback_working_event",
            "why": "event is not eligible for durable subject memory",
        }

    @staticmethod
    def _candidate_tags(
        *,
        event_type: str,
        memory_type: str,
        modality: str,
        organ: str,
        candidate_types: list[str],
        training_candidate: bool,
    ) -> list[str]:
        tags = [event_type, memory_type, modality, organ, *candidate_types]
        if training_candidate:
            tags.append("training_candidate")
        return MultimodalMemoryPolicy._unique_tags(tags)

    @staticmethod
    def _unique_tags(tags: list[str]) -> list[str]:
        unique: list[str] = []
        for tag in tags:
            cleaned = str(tag or "").strip()
            if cleaned and cleaned not in unique:
                unique.append(cleaned)
        return unique

    @staticmethod
    def _non_identity_write_source(source: str) -> str:
        cleaned = str(source or "").strip()
        if cleaned in {"eibrain.identity", "eibrain.preference"}:
            return ""
        return cleaned

    @staticmethod
    def _clean_text(value: object) -> str:
        return " ".join(str(value or "").strip().split())
