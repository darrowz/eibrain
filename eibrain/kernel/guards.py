"""Kernel guard hooks."""

from __future__ import annotations

from typing import Protocol

from eibrain.protocol.envelopes import Envelope


class Guard(Protocol):
    def allows(self, envelope: Envelope) -> bool: ...


class AllowAllGuard:
    def allows(self, envelope: Envelope) -> bool:
        return True
