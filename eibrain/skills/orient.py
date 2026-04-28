"""eibrain compatibility orient skill."""

from __future__ import annotations

from eibrain.protocol.actions import MoveHeadAction
from eibrain.protocol.intents import OrientIntent

from eiskills.orient import OrientToSpeakerSkill as EIOrientToSpeakerSkill


class OrientToSpeakerSkill:
    def compile(self, intent: OrientIntent) -> list[MoveHeadAction]:
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
