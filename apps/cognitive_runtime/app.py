"""Cognitive runtime assembly for deployable configurations."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from eibrain.cognition.attention.manager import AttentionManager
from eibrain.cognition.fusion.binder import ObservationBinder
from eibrain.cognition.dialogue.llm_router import LLMRouter
from eibrain.cognition.policy.engine import PolicyEngine
from eibrain.cognition.policy.multimodal_memory import MultimodalMemoryPolicy
from eibrain.cognition.planner.intent_planner import IntentPlanner
from eibrain.infra import TraceRecorder
from eibrain.infra.config import EIBrainConfig, load_config
from eibrain.learning.adaptation import AdaptationEngine
from eibrain.learning.evaluation import EvaluationEngine
from eibrain.learning.review import SelfReviewEngine
from eibrain.memory.adapters.factory import build_memory_adapter
from eibrain.memory.contracts import MemoryQuery
from eibrain.protocol.observations import AudioTranscriptFinal
from eibrain.skills.compiler import SkillCompiler
from eibrain.state.embodied import EmbodiedState
from eibrain.vision.minimax_cli import MiniMaxCLIAdapter
from eibrain.vision.minimax_mcp import MiniMaxMCPAdapter


class CognitiveRuntimeApp:
    def __init__(self, *, config: EIBrainConfig | None = None, vision_adapter=None) -> None:
        self.config = config or EIBrainConfig()
        self.memory = build_memory_adapter(self.config.memory.openclaw)
        self.binder = ObservationBinder()
        self.attention = AttentionManager()
        self.policy = PolicyEngine()
        self.memory_policy = MultimodalMemoryPolicy()
        self.planner = IntentPlanner(policy=self.policy, llm_router=LLMRouter(self.config.cognition.llm))
        self.compiler = SkillCompiler()
        self.review = SelfReviewEngine()
        self.evaluation = EvaluationEngine()
        self.adaptation = AdaptationEngine()
        self.vision = vision_adapter or self._build_vision_adapter()
        self.trace_recorder = TraceRecorder()
        self.last_review: dict[str, object] = {}
        self.last_learning_decision = "keep_policy"
        self.last_reply = ""
        self.last_attention: dict[str, object] = {}
        self.last_policy_decision: dict[str, object] = {}
        self.last_llm_status: dict[str, object] = {
            "provider": self.planner.llm_router.config.provider,
            "status": "idle",
            "error": "",
        }
        self.last_memory_diagnostics: dict[str, object] = {
            "last_query": "",
            "task_context": {},
            "last_recall": {},
            "last_writeback": {},
        }

    @classmethod
    def from_config_path(cls, path) -> "CognitiveRuntimeApp":
        return cls(config=load_config(path))

    @property
    def traces(self) -> list[dict[str, object]]:
        return self.trace_recorder.snapshot()

    def _build_vision_adapter(self):
        if (
            self.config.vision.provider == "minimax_cli"
            and self.config.vision.cli.enabled
            and self.config.vision.cli.api_key.strip()
            and self.config.vision.cli.command
        ):
            return MiniMaxCLIAdapter(self.config.vision.cli)
        if (
            self.config.vision.provider == "minimax_mcp"
            and self.config.vision.mcp.enabled
            and self.config.vision.mcp.api_key.strip()
            and self.config.vision.mcp.command
        ):
            return MiniMaxMCPAdapter(self.config.vision.mcp)
        return None

    def handle_observation(self, observation: AudioTranscriptFinal) -> list:
        state = EmbodiedState.create_default().with_observation(observation)
        moment = self.binder.bind(state)
        salience = self.attention.evaluate(moment)
        active_policy = self._get_active_policy(
            task_type="brain.respond",
            session_id=state.session.active_session_id,
            actor_id=state.world.current_speaker_id,
        )
        decision = self.policy.decide(
            state=state,
            moment=moment,
            salience=salience,
            active_policy=active_policy,
        )
        self._record_attention_policy(salience=salience, decision=decision)
        query_text = moment.query_text or state.world.last_transcript
        task_context = self._task_context(
            task_type="brain.respond",
            modality="audio_text",
            organ="ear",
            phase=state.engagement.phase,
            salience_score=salience.score,
            body_capabilities=moment.body_capabilities,
            active_policy=active_policy,
            query=query_text,
        )
        memory_result = self.memory.retrieve_context(
            MemoryQuery(
                query=query_text,
                session_id=state.session.active_session_id,
                actor_id=state.world.current_speaker_id,
                task_context=task_context,
            )
        )
        self._record_memory_diagnostics(query=query_text, task_context=task_context, memory_result=memory_result)
        intents = self.planner.plan(state=state, memory=memory_result, decision=decision)
        router = self.planner.llm_router
        self.last_llm_status = {
            "provider": router.last_provider,
            "status": router.last_status,
            "error": router.last_error,
            "text_preview": router.last_text[:80],
        }
        actions = self.compiler.compile(intents)
        self.last_reply = next((action.text for action in actions if hasattr(action, "text")), "")
        if decision.should_writeback:
            self.memory.remember_episode(
                session_id=state.session.active_session_id or "unknown-session",
                actor_id=state.world.current_speaker_id,
                summary=f"user:{state.world.last_transcript} | reply:{self.last_reply}",
                title="Audio dialogue turn",
                memory_type="conversation",
                source="eibrain.audio_dialogue",
                modality="audio_text",
                organ="ear",
                outcome={
                    "success": bool(actions),
                    "status": "planned",
                    "action_count": len(actions),
                    "reply_present": bool(self.last_reply),
                    **self.memory_policy.writeback_outcome(
                        modality="audio_text",
                        organ="ear",
                        success=bool(actions),
                        status="planned",
                        action_count=len(actions),
                        reply_present=bool(self.last_reply),
                        learning_decision=self.last_learning_decision,
                    ),
                },
            )
            self.last_memory_diagnostics["last_writeback"] = dict(getattr(self.memory, "last_writeback_status", {}))
        self._record_learning(
            event_type=observation.kind,
            transcript=state.world.last_transcript,
            reply=self.last_reply,
            outcome="planned",
        )
        self.trace_recorder.record(
            trace_id=state.session.active_session_id or "unknown-session",
            kind=observation.kind,
            payload={
                "transcript": state.world.last_transcript,
                "action_count": len(actions),
                "decision": decision.decision_type,
                "salience_score": salience.score,
            },
        )
        self._observe_outcome(
            signal_type="cognitive_turn",
            session_id=state.session.active_session_id,
            actor_id=state.world.current_speaker_id,
            payload={
                "modality": "audio_text",
                "decision": decision.decision_type,
                "action_count": len(actions),
                "salience_score": salience.score,
                "reply_present": bool(self.last_reply),
            },
        )
        return actions

    def describe_visual_frame(self, *, image_url: str):
        if self.vision is None:
            return None
        return self.vision.understand_image(
            prompt="Identify the main interaction target in this frame.",
            image_url=image_url,
        )

    def handle_visual_frame(self, *, image_url: str, actor_id: str | None = None, target_x: float | None = None) -> list:
        understanding = self.describe_visual_frame(image_url=image_url)
        if understanding is None:
            return []
        state = EmbodiedState.create_default().with_visual_focus(
            target_name=understanding.primary_subject or "target",
            actor_id=actor_id or "visual-target",
            summary=understanding.summary,
            target_x=target_x,
            ts=1.0,
        )
        visual_session_id = f"vision:{actor_id or 'visual-target'}"
        moment = self.binder.bind(state)
        salience = self.attention.evaluate(moment)
        active_policy = self._get_active_policy(
            task_type="brain.orient",
            session_id=visual_session_id,
            actor_id=actor_id or "visual-target",
        )
        decision = self.policy.decide(
            state=state,
            moment=moment,
            salience=salience,
            active_policy=active_policy,
        )
        self._record_attention_policy(salience=salience, decision=decision)
        intents = self.planner.plan(
            state=state,
            memory=self._retrieve_memory_for_visual(
                query=moment.query_text or understanding.summary,
                session_id=visual_session_id,
                actor_id=actor_id or "visual-target",
                phase=state.engagement.phase,
                salience_score=salience.score,
                body_capabilities=moment.body_capabilities,
                active_policy=active_policy,
                visual_context={"summary": understanding.summary, "target_x": target_x},
            ),
            decision=decision,
        )
        actions = self.compiler.compile(intents)
        if decision.should_writeback:
            self.memory.remember_episode(
                session_id=visual_session_id,
                actor_id=actor_id or "visual-target",
                summary=understanding.summary,
                title="Visual frame understanding",
                memory_type="fact",
                source="eibrain.visual_frame",
                modality="vision",
                organ="eye",
                outcome={
                    "success": bool(actions),
                    "status": "planned",
                    "action_count": len(actions),
                    "reply_present": False,
                    **self.memory_policy.writeback_outcome(
                        modality="vision",
                        organ="eye",
                        success=bool(actions),
                        status="planned",
                        action_count=len(actions),
                        reply_present=False,
                        learning_decision=self.last_learning_decision,
                        visual_context={"summary": understanding.summary, "target_x": target_x},
                    ),
                },
            )
            self.last_memory_diagnostics["last_writeback"] = dict(getattr(self.memory, "last_writeback_status", {}))
        self.trace_recorder.record(
            trace_id=f"vision:{actor_id or 'visual-target'}",
            kind="vision_frame_captured",
            payload={
                "summary": understanding.summary,
                "action_count": len(actions),
                "decision": decision.decision_type,
                "salience_score": salience.score,
            },
        )
        self._observe_outcome(
            signal_type="visual_turn",
            session_id=visual_session_id,
            actor_id=actor_id or "visual-target",
            payload={
                "modality": "vision",
                "decision": decision.decision_type,
                "action_count": len(actions),
                "salience_score": salience.score,
            },
        )
        return [action for action in actions if action.kind == "move_head_action"]

    def build_world_observation_payload(
        self,
        visual_state: dict[str, object],
        *,
        session_id: str,
        actor_id: str | None = None,
    ) -> dict[str, object]:
        objects = self._normalize_world_objects(visual_state.get("objects") or visual_state.get("detections") or [])
        confidence = max((float(item.get("confidence", 0.0)) for item in objects), default=0.0)
        content: dict[str, object] = {
            "source": str(visual_state.get("source") or visual_state.get("backend") or "visual_state"),
            "objects": objects,
            "confidence": confidence,
        }
        if visual_state.get("frame_path"):
            content["frame_path"] = str(visual_state["frame_path"])
        if visual_state.get("identity_candidates"):
            content["identity_candidates"] = list(visual_state.get("identity_candidates") or [])

        object_labels = [str(item.get("label", "")).strip() for item in objects if str(item.get("label", "")).strip()]
        summary = "Observed visual scene"
        if object_labels:
            summary = f"Observed {', '.join(object_labels)}"
        dedupe_key = self._world_observation_dedupe_key(content)
        tags = self._unique_tags(["world_observation", "vision", *object_labels])
        return {
            "session_id": session_id,
            "actor_id": actor_id,
            "summary": summary,
            "content": content,
            "meta": {
                "source": "eibrain.visual_world",
                "modality": "vision",
                "organ": "eye",
                "session_id": session_id,
                "actor_id": actor_id,
                "dedupe_key": dedupe_key,
                "confidence": confidence,
            },
            "tags": tags,
        }

    def remember_world_observation_from_state(
        self,
        visual_state: dict[str, object],
        *,
        session_id: str,
        actor_id: str | None = None,
    ) -> dict[str, object]:
        payload = self.build_world_observation_payload(
            visual_state,
            session_id=session_id,
            actor_id=actor_id,
        )
        writer = getattr(self.memory, "remember_world_observation", None)
        if not callable(writer):
            return payload
        writer(
            session_id=str(payload["session_id"]),
            actor_id=payload.get("actor_id") if isinstance(payload.get("actor_id"), str) else None,
            summary=str(payload["summary"]),
            content=dict(payload["content"]),
            meta=dict(payload["meta"]),
            tags=[str(tag) for tag in payload.get("tags", [])],
        )
        self.last_memory_diagnostics["last_writeback"] = dict(getattr(self.memory, "last_writeback_status", {}))
        return payload

    def _record_learning(self, *, event_type: str, transcript: str, reply: str, outcome: str) -> None:
        self.last_review = self.review.review_turn(
            event_type=event_type,
            transcript=transcript,
            reply=reply,
            outcome=outcome,
        )
        score = self.evaluation.score_review(self.last_review)
        self.last_learning_decision = self.adaptation.decide(score)

    def snapshot(self) -> dict[str, object]:
        return {
            "last_reply": self.last_reply,
            "learning_decision": self.last_learning_decision,
            "last_review": self.last_review,
            "last_llm_status": dict(self.last_llm_status),
            "last_attention": dict(self.last_attention),
            "last_policy_decision": dict(self.last_policy_decision),
            "memory_diagnostics": dict(self.last_memory_diagnostics),
        }

    def _retrieve_memory_for_visual(
        self,
        *,
        query: str,
        session_id: str,
        actor_id: str,
        phase: str,
        salience_score: float,
        body_capabilities: dict[str, bool],
        active_policy: dict[str, object],
        visual_context: dict[str, object],
    ):
        task_context = self._task_context(
            task_type="brain.orient",
            modality="vision",
            organ="eye",
            phase=phase,
            salience_score=salience_score,
            body_capabilities=body_capabilities,
            active_policy=active_policy,
            query=query,
            visual_context=visual_context,
        )
        result = self.memory.retrieve_context(
            MemoryQuery(
                query=query,
                session_id=session_id,
                actor_id=actor_id,
                task_context=task_context,
            )
        )
        self._record_memory_diagnostics(query=query, task_context=task_context, memory_result=result)
        return result

    def _record_memory_diagnostics(self, *, query: str, task_context: dict[str, Any], memory_result) -> None:
        self.last_memory_diagnostics = {
            "last_query": query,
            "task_context": dict(task_context),
            "last_recall": dict(getattr(memory_result, "recall_diagnostics", {}) or {}),
            "last_writeback": dict(getattr(self.memory, "last_writeback_status", {}) or {}),
        }

    def _task_context(
        self,
        *,
        task_type: str,
        modality: str,
        organ: str,
        phase: str,
        salience_score: float,
        body_capabilities: dict[str, bool],
        active_policy: dict[str, object],
        query: str = "",
        visual_context: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        return self.memory_policy.build_recall_context(
            task_type=task_type,
            modality=modality,
            organ=organ,
            phase=phase,
            salience_score=salience_score,
            body_capabilities=body_capabilities,
            active_policy=active_policy,
            query=query,
            visual_context=visual_context,
        )

    def _get_active_policy(
        self,
        *,
        task_type: str,
        session_id: str | None,
        actor_id: str | None,
    ) -> dict[str, object]:
        getter = getattr(self.memory, "get_active_policy", None)
        if getter is None:
            return {}
        try:
            return dict(getter(task_type=task_type, session_id=session_id, actor_id=actor_id) or {})
        except (OSError, ValueError, TypeError):
            return {}

    def _observe_outcome(
        self,
        *,
        signal_type: str,
        payload: dict[str, object],
        session_id: str | None,
        actor_id: str | None,
    ) -> None:
        observer = getattr(self.memory, "observe_outcome", None)
        if observer is None:
            return
        try:
            observer(signal_type=signal_type, payload=payload, session_id=session_id, actor_id=actor_id)
        except (OSError, ValueError, TypeError):
            return

    def _normalize_world_objects(self, objects: object) -> list[dict[str, object]]:
        if not isinstance(objects, list):
            return []
        normalized: list[dict[str, object]] = []
        for item in objects:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or item.get("name") or item.get("class") or "").strip()
            if not label:
                continue
            normalized_item = {
                key: value
                for key, value in item.items()
                if key not in {"score", "class", "name"} and value is not None
            }
            normalized_item["label"] = label
            normalized_item["confidence"] = float(item.get("confidence", item.get("score", 0.0)) or 0.0)
            normalized.append(normalized_item)
        return normalized

    def _world_observation_dedupe_key(self, content: dict[str, object]) -> str:
        fingerprint = json.dumps(content, sort_keys=True, ensure_ascii=True, default=str)
        digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
        return f"world_observation:{digest}"

    def _unique_tags(self, tags: list[str]) -> list[str]:
        unique: list[str] = []
        for tag in tags:
            cleaned = str(tag or "").strip()
            if cleaned and cleaned not in unique:
                unique.append(cleaned)
        return unique

    def _record_attention_policy(self, *, salience, decision) -> None:
        self.last_attention = {
            "score": round(float(salience.score), 3),
            "reason": salience.reason,
            "should_recall": salience.should_recall,
            "should_reply": salience.should_reply,
            "should_orient": salience.should_orient,
            "should_writeback": salience.should_writeback,
            "priority": salience.priority,
        }
        self.last_policy_decision = {
            "decision_type": decision.decision_type,
            "reason": decision.reason,
            "should_recall": decision.should_recall,
            "should_reply": decision.should_reply,
            "should_orient": decision.should_orient,
            "should_writeback": decision.should_writeback,
            "priority": decision.priority,
        }
