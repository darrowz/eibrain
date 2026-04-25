"""Bind raw embodied state into a cognitive moment."""

from eibrain.protocol.events import Moment
from eibrain.state.embodied import EmbodiedState


class ObservationBinder:
    def bind(self, state: EmbodiedState) -> Moment:
        modalities = []
        if state.world.last_transcript.strip():
            modalities.append("audio_text")
        if state.world.last_visual_summary.strip() or state.world.focus_target_name:
            modalities.append("vision")
        return Moment(
            state=state,
            session_id=state.session.active_session_id,
            actor_id=state.world.current_speaker_id,
            transcript=state.world.last_transcript,
            visual_summary=state.world.last_visual_summary,
            visual_target=state.world.focus_target_name,
            target_x=state.world.focus_target_x,
            engagement_phase=state.engagement.phase,
            modalities=tuple(modalities),
            body_capabilities=self._capabilities(state),
        )

    def _capabilities(self, state: EmbodiedState) -> dict[str, bool]:
        payload = {}
        for name in (
            "can_hear_voice",
            "can_transcribe_speech",
            "can_see_people",
            "can_identify_person",
            "can_speak",
            "can_orient_head",
        ):
            payload[name] = bool(getattr(state.capabilities, name, False))
        return payload
