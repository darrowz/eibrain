"""Policy and engagement state machines."""

from eibrain.protocol.events import CognitiveDecision, Moment, SalienceDecision
from eibrain.state.embodied import EmbodiedState


class PolicyEngine:
    def should_reply(self, state: EmbodiedState, decision: CognitiveDecision | None = None) -> bool:
        if decision is not None:
            return decision.should_reply
        return bool(state.world.last_transcript.strip())

    def decide(
        self,
        *,
        state: EmbodiedState,
        moment: Moment,
        salience: SalienceDecision,
        active_policy: dict[str, object] | None = None,
    ) -> CognitiveDecision:
        if state.engagement.phase == "interrupted":
            return CognitiveDecision(
                decision_type="interrupt",
                reason="user_interrupt",
                should_recall=False,
                should_reply=False,
                should_orient=False,
                should_writeback=False,
                priority=100,
                active_policy=dict(active_policy or {}),
            )
        if salience.should_reply:
            return CognitiveDecision(
                decision_type="reply",
                reason=salience.reason,
                should_recall=salience.should_recall,
                should_reply=True,
                should_orient=salience.should_orient,
                should_writeback=True,
                priority=salience.priority,
                active_policy=dict(active_policy or {}),
            )
        if salience.should_orient:
            return CognitiveDecision(
                decision_type="orient",
                reason=salience.reason,
                should_recall=salience.should_recall,
                should_reply=False,
                should_orient=True,
                should_writeback=True,
                priority=salience.priority,
                active_policy=dict(active_policy or {}),
            )
        return CognitiveDecision(
            decision_type="ignore",
            reason=salience.reason,
            should_recall=salience.should_recall,
            should_reply=False,
            should_orient=False,
            should_writeback=False,
            priority=salience.priority,
            active_policy=dict(active_policy or {}),
        )


class EngagementStateMachine:
    _TRANSITIONS = {
        ("idle", "speech_started"): "listening",
        ("listening", "transcript_final"): "thinking",
        ("thinking", "reply_started"): "speaking",
        ("speaking", "user_interrupt"): "interrupted",
        ("interrupted", "listen_resumed"): "listening",
    }

    def next_phase(self, *, current: str, event: str) -> str:
        return self._TRANSITIONS.get((current, event), current)
