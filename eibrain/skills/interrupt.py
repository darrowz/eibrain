"""Interrupt skill."""

from __future__ import annotations

from eibrain.protocol.actions import StopSpeechAction
from eibrain.protocol.intents import PauseIntent

from .base import Skill


class InterruptForUserSkill(Skill):
    def compile(self, intent: PauseIntent) -> list[StopSpeechAction]:
        return [
            StopSpeechAction(
                ts=intent.ts,
                source="interrupt_skill",
                session_id=intent.session_id,
                actor_id=intent.actor_id,
                target_id=intent.target_id,
            )
        ]
