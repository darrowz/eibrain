from __future__ import annotations

from pathlib import Path
import sys

import pytest


def _ensure_eimemory_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    eimemory_repo = repo_root / "eimemory"
    if not (eimemory_repo / "eimemory").exists():
        pytest.skip("eimemory repository checkout is unavailable for live cross-repo test")
    repo_path = str(eimemory_repo)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def test_eimemory_rpc_adapter_reads_from_live_server(tmp_path) -> None:
    _ensure_eimemory_repo_on_path()

    from eimemory.adapters.eibrain.rpc_server import EIBrainRPCServer
    from eimemory.api.runtime import Runtime

    from eibrain.infra.config import OpenClawConfig
    from eibrain.memory.adapters.eimemory_rpc import EIMemoryRPCAdapter
    from eibrain.memory.contracts import MemoryQuery

    runtime = Runtime.create(root=tmp_path / "eimemory-runtime")
    server = EIBrainRPCServer(runtime, host="127.0.0.1", port=0)
    server.start()
    try:
        ingest_response = server.request(
            {
                "method": "memory.ingest",
                "params": {
                    "text": "Prefer concise embodied replies.",
                    "memory_type": "preference",
                    "title": "Reply style",
                    "scope": {"agent_id": "honxin", "workspace_id": "robot"},
                },
            }
        )
        assert ingest_response["ok"] is True
        adapter = EIMemoryRPCAdapter(
            OpenClawConfig(
                provider="eimemory_rpc",
                endpoint=f"http://{server.address[0]}:{server.address[1]}/",
                agent_id="honxin",
                workspace_id="robot",
            )
        )
        result = adapter.retrieve_context(MemoryQuery(query="concise reply"))
    finally:
        server.stop()
        runtime.close()

    assert "concise" in result.summary.lower()
