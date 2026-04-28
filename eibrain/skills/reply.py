"""eibrain compatibility reply skill."""

from __future__ import annotations

from eibrain.protocol.actions import PlaySpeechAction
from eibrain.protocol.intents import SpeakIntent

from eiskills.reply import ReplySkill as EIReplySkill


class ReplySkill:
    def compile(self, intent: SpeakIntent) -> list[PlaySpeechAction]:
        return [
            PlaySpeechAction(
                ts=action.ts,
                source=action.source,
                session_id=action.session_id,
                actor_id=action.actor_id,
                target_id=action.target_id,
                text=action.text,
            )
            for action in EIReplySkill().compile(intent)
        ]
