"""Cognitive runtime assembly for deployable configurations."""

from __future__ import annotations

from eibrain.cognition.dialogue.llm_router import LLMRouter
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
        self.planner = IntentPlanner(llm_router=LLMRouter(self.config.cognition.llm))
        self.compiler = SkillCompiler()
        self.review = SelfReviewEngine()
        self.evaluation = EvaluationEngine()
        self.adaptation = AdaptationEngine()
        self.vision = vision_adapter or self._build_vision_adapter()
        self.trace_recorder = TraceRecorder()
        self.last_review: dict[str, object] = {}
        self.last_learning_decision = "keep_policy"
        self.last_reply = ""
        self.last_llm_status: dict[str, object] = {
            "provider": self.planner.llm_router.config.provider,
            "status": "idle",
            "error": "",
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
        memory_result = self.memory.retrieve_context(
            MemoryQuery(
                query=state.world.last_transcript,
                session_id=state.session.active_session_id,
                actor_id=state.world.current_speaker_id,
            )
        )
        intents = self.planner.plan(state=state, memory=memory_result)
        router = self.planner.llm_router
        self.last_llm_status = {
            "provider": router.last_provider,
            "status": router.last_status,
            "error": router.last_error,
            "text_preview": router.last_text[:80],
        }
        actions = self.compiler.compile(intents)
        self.last_reply = next((action.text for action in actions if hasattr(action, "text")), "")
        self.memory.remember_episode(
            session_id=state.session.active_session_id or "unknown-session",
            summary=f"user:{state.world.last_transcript} | reply:{self.last_reply}",
        )
        self._record_learning(
            event_type=observation.kind,
            transcript=state.world.last_transcript,
            reply=self.last_reply,
            outcome="planned",
        )
        self.trace_recorder.record(
            trace_id=state.session.active_session_id or "unknown-session",
            kind=observation.kind,
            payload={"transcript": state.world.last_transcript, "action_count": len(actions)},
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
        intents = self.planner.plan(state=state, memory=self.memory.retrieve_context(MemoryQuery(query=understanding.summary)))
        actions = self.compiler.compile(intents)
        self.trace_recorder.record(
            trace_id=f"vision:{actor_id or 'visual-target'}",
            kind="vision_frame_captured",
            payload={"summary": understanding.summary, "action_count": len(actions)},
        )
        return [action for action in actions if action.kind == "move_head_action"]

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
        }
