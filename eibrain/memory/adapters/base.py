"""Base memory adapter contract."""

from __future__ import annotations

from typing import Protocol

from eibrain.memory.contracts import MemoryQuery, MemoryResult


class MemoryAdapter(Protocol):
    def retrieve_context(self, query: MemoryQuery) -> MemoryResult: ...
