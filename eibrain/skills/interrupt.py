"""eibrain compatibility interrupt skill."""

from __future__ import annotations

from eibrain.protocol.actions import StopSpeechAction
from eibrain.protocol.intents import PauseIntent

try:
    from eiskills.interrupt import InterruptForUserSkill as EIInterruptForUserSkill
except ModuleNotFoundError:  # pragma: no cover
    EIInterruptForUserSkill = None


class InterruptForUserSkill:
    def compile(self, intent: PauseIntent) -> list[StopSpeechAction]:
        if EIInterruptForUserSkill is None:
            return [
                StopSpeechAction(
                    ts=intent.ts,
                    source=intent.source,
                    session_id=intent.session_id,
                    actor_id=intent.actor_id,
                    target_id=intent.target_id,
                )
            ]
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

