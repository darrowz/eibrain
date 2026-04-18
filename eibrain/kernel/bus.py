"""Simple in-process kernel bus."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

from eibrain.protocol.envelopes import Envelope


class KernelBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[[Envelope], None]]] = defaultdict(list)

    def subscribe(self, channel: str, callback: Callable[[Envelope], None]) -> None:
        self._subscribers[channel].append(callback)

    def publish(self, envelope: Envelope) -> None:
        for callback in self._subscribers.get(envelope.channel, []):
            callback(envelope)
