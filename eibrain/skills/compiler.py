"""eibrain compatibility adapter for the standalone eiskills compiler."""

from __future__ import annotations

from eibrain.protocol.actions import Action
from eibrain.protocol.actions import MoveHeadAction, PlaySpeechAction, StopSpeechAction
from eibrain.protocol.intents import Intent, OrientIntent, PauseIntent, SpeakIntent

try:
    from eiskills.compiler import SkillCompiler as EISkillCompiler
except ModuleNotFoundError:  # pragma: no cover - fallback in environments without eiskills
    EISkillCompiler = None


class SkillCompiler:
    def __init__(self) -> None:
        self._compiler = EISkillCompiler() if EISkillCompiler is not None else None

    def compile(self, intents: Intent | list[Intent]) -> list[Action]:
        if self._compiler is not None:
            return [_to_eibrain_action(action) for action in self._compiler.compile(intents)]
        return _fallback_compile(intents)


def _fallback_compile(intents: Intent | list[Intent]) -> list[Action]:
    if isinstance(intents, (list, tuple)):
        iterable = intents
    else:
        iterable = [intents]
    actions: list[Action] = []
    for intent in iterable:
        if isinstance(intent, SpeakIntent):
            actions.append(
                PlaySpeechAction(
                    ts=intent.ts,
                    source=intent.source,
                    session_id=intent.session_id,
                    actor_id=intent.actor_id,
                    target_id=intent.target_id,
                    text=intent.text,
                )
            )
            continue
        if isinstance(intent, PauseIntent):
            actions.append(
                StopSpeechAction(
                    ts=intent.ts,
                    source=intent.source,
                    session_id=intent.session_id,
                    actor_id=intent.actor_id,
                    target_id=intent.target_id,
                )
            )
            continue
        if isinstance(intent, OrientIntent):
            actions.append(
                MoveHeadAction(
                    ts=intent.ts,
                    source=intent.source,
                    session_id=intent.session_id,
                    actor_id=intent.actor_id,
                    target_id=intent.target_id,
                    target_name=intent.target_name,
                    target_x=intent.target_x,
                )
            )
            continue
        raise ValueError(f"Unsupported intent kind: {getattr(intent, 'kind', type(intent).__name__)}")
    return actions


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

