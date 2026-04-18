"""Minimal observation binder."""

from eibrain.state.embodied import EmbodiedState


class ObservationBinder:
    def bind(self, state: EmbodiedState) -> EmbodiedState:
        return state
