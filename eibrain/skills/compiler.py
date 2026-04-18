"""Phase 1 skill compiler."""

from __future__ import annotations

from eibrain.protocol.actions import Action
from eibrain.protocol.intents import Intent, OrientIntent, PauseIntent, SpeakIntent

from .interrupt import InterruptForUserSkill
from .orient import OrientToSpeakerSkill
from .reply import ReplySkill


class SkillCompiler:
    def __init__(self) -> None:
        self.reply_skill = ReplySkill()
        self.interrupt_skill = InterruptForUserSkill()
        self.orient_skill = OrientToSpeakerSkill()

    def compile(self, intents: Intent | list[Intent]) -> list[Action]:
        if not isinstance(intents, list):
            intents = [intents]

        actions: list[Action] = []
        for intent in intents:
            if isinstance(intent, SpeakIntent):
                actions.extend(self.reply_skill.compile(intent))
            elif isinstance(intent, PauseIntent):
                actions.extend(self.interrupt_skill.compile(intent))
            elif isinstance(intent, OrientIntent):
                actions.extend(self.orient_skill.compile(intent))
        return actions
