"""Minimal attention manager."""

from eibrain.state.embodied import EmbodiedState


class AttentionManager:
    def current_target(self, state: EmbodiedState) -> str | None:
        return state.world.current_speaker_id
