"""Policy and engagement state machines."""

from eibrain.state.embodied import EmbodiedState


class PolicyEngine:
    def should_reply(self, state: EmbodiedState) -> bool:
        return bool(state.world.last_transcript.strip())


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
