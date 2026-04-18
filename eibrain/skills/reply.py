"""Reply skill."""

from __future__ import annotations

from eibrain.protocol.actions import PlaySpeechAction
from eibrain.protocol.intents import SpeakIntent

from .base import Skill


class ReplySkill(Skill):
    def compile(self, intent: SpeakIntent) -> list[PlaySpeechAction]:
        return [
            PlaySpeechAction(
                ts=intent.ts,
                source="reply_skill",
                session_id=intent.session_id,
                actor_id=intent.actor_id,
                target_id=intent.target_id,
                text=intent.text,
            )
        ]
