"""Embodied state aggregate."""

from __future__ import annotations

from dataclasses import dataclass

from eibrain.body.health.capability_matrix import CapabilityMatrix
from eibrain.protocol.observations import AudioTranscriptFinal

from .body import BodyState
from .engagement import EngagementState
from .self_state import SelfState
from .session import SessionState
from .world import WorldState


@dataclass(slots=True)
class EmbodiedState:
    body: BodyState
    world: WorldState
    self_state: SelfState
    session: SessionState
    engagement: EngagementState
    capabilities: CapabilityMatrix

    @classmethod
    def create_default(cls) -> "EmbodiedState":
        return cls(
            body=BodyState(),
            world=WorldState(),
            self_state=SelfState(),
            session=SessionState(),
            engagement=EngagementState(),
            capabilities=CapabilityMatrix(),
        )

    def with_observation(self, observation: AudioTranscriptFinal) -> "EmbodiedState":
        next_state = self.create_default()
        next_state.body = self.body
        next_state.capabilities = self.capabilities
        next_state.world = WorldState(
            current_speaker_id=observation.actor_id,
            last_transcript=observation.text,
        )
        next_state.session = SessionState(active_session_id=observation.session_id)
        next_state.engagement = EngagementState(phase="listening")
        next_state.self_state = SelfState(mode="attending")
        return next_state

    def with_visual_focus(
        self,
        *,
        target_name: str,
        actor_id: str,
        summary: str,
        target_x: float | None = None,
        ts: float,
    ) -> "EmbodiedState":
        next_state = self.create_default()
        next_state.body = self.body
        next_state.capabilities = self.capabilities
        next_state.world = WorldState(
            current_speaker_id=actor_id,
            last_visual_summary=summary,
            focus_target_name=target_name,
            focus_target_x=target_x,
        )
        next_state.session = self.session
        next_state.engagement = EngagementState(phase="noticing")
        next_state.self_state = SelfState(mode="tracking")
        return next_state

    def with_interrupt(
        self,
        *,
        session_id: str,
        actor_id: str,
        ts: float,
    ) -> "EmbodiedState":
        next_state = self.create_default()
        next_state.body = self.body
        next_state.capabilities = self.capabilities
        next_state.world = WorldState(current_speaker_id=actor_id)
        next_state.session = SessionState(active_session_id=session_id)
        next_state.engagement = EngagementState(phase="interrupted")
        next_state.self_state = SelfState(mode="attending")
        return next_state

    def with_transcript(
        self,
        *,
        text: str,
        session_id: str,
        actor_id: str,
        ts: float,
    ) -> "EmbodiedState":
        return self.with_observation(
            AudioTranscriptFinal(
                ts=ts,
                source="ear.asr",
                text=text,
                session_id=session_id,
                actor_id=actor_id,
            )
        )
