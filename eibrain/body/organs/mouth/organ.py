"""Mouth organ implementation."""

from __future__ import annotations

from eibrain.body.organs.base import BaseOrgan
from eibrain.protocol.actions import PlaySpeechAction, StopSpeechAction
from eibrain.protocol.outcomes import ActionExecuted, SpeechPlaybackCompleted


class MouthOrgan(BaseOrgan):
    name = "mouth"
    subfunction_names = ("tts_plan", "tts_playback")

    def supports_action(self, action) -> bool:
        return isinstance(action, (PlaySpeechAction, StopSpeechAction))

    def handle_action(self, action):
        if isinstance(action, PlaySpeechAction):
            result = self.drivers["tts_playback"].invoke("play_speech", {"text": action.text})
            return SpeechPlaybackCompleted(
                ts=action.ts,
                source="mouth.tts_playback",
                status=result.status,
                session_id=action.session_id,
                actor_id=action.actor_id,
                target_id=action.target_id,
            )
        if isinstance(action, StopSpeechAction):
            result = self.drivers["tts_playback"].invoke("stop_speech", {})
            return ActionExecuted(
                ts=action.ts,
                source="mouth.tts_playback",
                session_id=action.session_id,
                actor_id=action.actor_id,
                target_id=action.target_id,
                action_kind=action.kind,
                details=result.details,
            )
        return None
