"""Envelope router with guard checks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

from eibrain.protocol.envelopes import Envelope

from .guards import Guard


class EnvelopeRouter:
    def __init__(self, guards: list[Guard] | None = None) -> None:
        self._handlers: dict[str, list[Callable[[Envelope], None]]] = defaultdict(list)
        self._guards = list(guards or [])

    def register(self, channel: str, callback: Callable[[Envelope], None]) -> None:
        self._handlers[channel].append(callback)

    def route(self, envelope: Envelope) -> bool:
        if not all(guard.allows(envelope) for guard in self._guards):
            return False

        for callback in self._handlers.get(envelope.channel, []):
            callback(envelope)
        return True
