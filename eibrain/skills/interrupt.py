"""eibrain compatibility interrupt skill."""

from __future__ import annotations

from eibrain.protocol.actions import StopSpeechAction
from eibrain.protocol.intents import PauseIntent

from eiskills.interrupt import InterruptForUserSkill as EIInterruptForUserSkill


class InterruptForUserSkill:
    def compile(self, intent: PauseIntent) -> list[StopSpeechAction]:
        return [
            StopSpeechAction(
                ts=action.ts,
                source=action.source,
                session_id=action.session_id,
                actor_id=action.actor_id,
                target_id=action.target_id,
            )
            for action in EIInterruptForUserSkill().compile(intent)
        ]
