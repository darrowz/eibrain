"""Memory adapter factory."""

from __future__ import annotations

from eibrain.infra.config import OpenClawConfig
from eibrain.memory.adapters.base import MemoryAdapter
from eibrain.memory.adapters.eimemory_rpc import EIMemoryRPCAdapter
from eibrain.memory.adapters.openclaw import OpenClawMemoryAdapter


def build_memory_adapter(config: OpenClawConfig) -> MemoryAdapter:
    if config.provider == "eimemory_rpc":
        return EIMemoryRPCAdapter(config)
    return OpenClawMemoryAdapter(config)
