"""Attention scoring for embodied moments."""

from eibrain.protocol.events import Moment, SalienceDecision
from eibrain.state.embodied import EmbodiedState


class AttentionManager:
    def current_target(self, state: EmbodiedState) -> str | None:
        return state.world.current_speaker_id

    def evaluate(self, moment: Moment) -> SalienceDecision:
        score = 0.0
        reasons = []
        transcript = moment.transcript.strip()
        if transcript:
            score += 0.7
            reasons.append("speech")
        if moment.visual_target:
            score += 0.25
            reasons.append("visual_target")
        if moment.actor_id:
            score += 0.1
            reasons.append("actor")
        score = min(score, 1.0)
        should_reply = bool(transcript and score >= 0.5)
        should_orient = bool(moment.visual_target and score >= 0.2)
        return SalienceDecision(
            score=score,
            reason=",".join(reasons) or "idle",
            should_recall=bool(transcript or moment.visual_summary.strip()),
            should_reply=should_reply,
            should_orient=should_orient,
            should_writeback=bool(should_reply or should_orient),
            priority=int(score * 100),
        )
