"""eibrain compatibility adapter for the standalone eiskills compiler."""

from __future__ import annotations

from eiskills.compiler import SkillCompiler as EISkillCompiler

from eibrain.protocol.actions import Action
from eibrain.protocol.actions import MoveHeadAction, PlaySpeechAction, StopSpeechAction
from eibrain.protocol.intents import Intent


class SkillCompiler:
    def __init__(self) -> None:
        self._compiler = EISkillCompiler()

    def compile(self, intents: Intent | list[Intent]) -> list[Action]:
        return [_to_eibrain_action(action) for action in self._compiler.compile(intents)]


def _to_eibrain_action(action) -> Action:
    if action.kind == "play_speech_action":
        return PlaySpeechAction(
            ts=action.ts,
            source=action.source,
            session_id=action.session_id,
            actor_id=action.actor_id,
            target_id=action.target_id,
            text=action.text,
        )
    if action.kind == "stop_speech_action":
        return StopSpeechAction(
            ts=action.ts,
            source=action.source,
            session_id=action.session_id,
            actor_id=action.actor_id,
            target_id=action.target_id,
        )
    if action.kind == "move_head_action":
        return MoveHeadAction(
            ts=action.ts,
            source=action.source,
            session_id=action.session_id,
            actor_id=action.actor_id,
            target_id=action.target_id,
            target_name=action.target_name,
            target_x=action.target_x,
            target_angle=action.target_angle,
        )
    raise ValueError(f"Unsupported eiskills action kind: {action.kind}")
