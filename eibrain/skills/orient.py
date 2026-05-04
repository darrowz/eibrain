"""eibrain compatibility orient skill."""

from __future__ import annotations

from eibrain.protocol.actions import MoveHeadAction
from eibrain.protocol.intents import OrientIntent

try:
    from eiskills.orient import OrientToSpeakerSkill as EIOrientToSpeakerSkill
except ModuleNotFoundError:  # pragma: no cover
    EIOrientToSpeakerSkill = None


class OrientToSpeakerSkill:
    def compile(self, intent: OrientIntent) -> list[MoveHeadAction]:
        if EIOrientToSpeakerSkill is None:
            return [
                MoveHeadAction(
                    ts=intent.ts,
                    source=intent.source,
                    session_id=intent.session_id,
                    actor_id=intent.actor_id,
                    target_id=intent.target_id,
                    target_name=intent.target_name,
                    target_x=intent.target_x,
                    target_angle=None,
                )
            ]
        return [
            MoveHeadAction(
                ts=action.ts,
                source=action.source,
                session_id=action.session_id,
                actor_id=action.actor_id,
                target_id=action.target_id,
                target_name=action.target_name,
                target_x=action.target_x,
                target_angle=action.target_angle,
            )
            for action in EIOrientToSpeakerSkill().compile(intent)
        ]

