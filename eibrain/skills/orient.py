"""Orient skill."""

from __future__ import annotations

from eibrain.protocol.actions import MoveHeadAction
from eibrain.protocol.intents import OrientIntent

from .base import Skill


class OrientToSpeakerSkill(Skill):
    def compile(self, intent: OrientIntent) -> list[MoveHeadAction]:
        return [
            MoveHeadAction(
                ts=intent.ts,
                source="orient_skill",
                session_id=intent.session_id,
                actor_id=intent.actor_id,
                target_id=intent.target_id,
                target_name=intent.target_name,
                target_x=intent.target_x,
            )
        ]
