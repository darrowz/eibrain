"""Minimal phase 1 intent planner."""

from eibrain.cognition.dialogue.dialogue_manager import DialogueManager
from eibrain.cognition.dialogue.llm_router import LLMRouter
from eibrain.cognition.dialogue.prompt_builder import PromptBuilder
from eibrain.cognition.policy.engine import PolicyEngine
from eibrain.memory.contracts import MemoryResult
from eibrain.protocol.intents import OrientIntent, PauseIntent, SpeakIntent
from eibrain.state.embodied import EmbodiedState


class IntentPlanner:
    def __init__(
        self,
        *,
        policy: PolicyEngine | None = None,
        prompt_builder: PromptBuilder | None = None,
        llm_router: LLMRouter | None = None,
        dialogue_manager: DialogueManager | None = None,
    ) -> None:
        self.policy = policy or PolicyEngine()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.llm_router = llm_router or LLMRouter()
        self.dialogue_manager = dialogue_manager or DialogueManager()

    def plan(self, state: EmbodiedState, memory: MemoryResult) -> list[SpeakIntent | OrientIntent | PauseIntent]:
        if state.engagement.phase == "interrupted":
            return [
                PauseIntent(
                    ts=state.session.active_session_id is not None and 1.0 or 0.0,
                    source="intent_planner",
                    session_id=state.session.active_session_id,
                    actor_id=state.world.current_speaker_id,
                    reason="user_interrupt",
                    priority=100,
                )
            ]

        intents: list[SpeakIntent | OrientIntent | PauseIntent] = []
        prompt = self.prompt_builder.build(state, memory)
        llm_text = self.llm_router.generate(prompt) if self.policy.should_reply(state) else ""
        reply_text = self.dialogue_manager.build_reply_text(state, memory, llm_text).strip()
        if reply_text:
            intents.append(
                SpeakIntent(
                    ts=state.session.active_session_id is not None and 1.0 or 0.0,
                    source="intent_planner",
                    session_id=state.session.active_session_id,
                    actor_id=state.world.current_speaker_id,
                    reason="phase1_reply",
                    priority=10,
                    text=reply_text,
                )
            )
        if state.world.focus_target_name:
            intents.append(
                OrientIntent(
                    ts=state.session.active_session_id is not None and 1.0 or 0.0,
                    source="intent_planner",
                    session_id=state.session.active_session_id,
                    actor_id=state.world.current_speaker_id,
                    target_id=state.world.current_speaker_id,
                    reason="visual_focus",
                    priority=5,
                    target_name=state.world.focus_target_name,
                    target_x=state.world.focus_target_x,
                )
            )
        return intents
