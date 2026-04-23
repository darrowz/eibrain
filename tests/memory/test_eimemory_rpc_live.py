from __future__ import annotations


def test_eimemory_rpc_adapter_reads_from_live_server(tmp_path) -> None:
    from eimemory.adapters.eibrain.rpc_server import EIBrainRPCServer
    from eimemory.api.runtime import Runtime

    from eibrain.infra.config import OpenClawConfig
    from eibrain.memory.adapters.eimemory_rpc import EIMemoryRPCAdapter
    from eibrain.memory.contracts import MemoryQuery

    runtime = Runtime.create(root=tmp_path / "eimemory-runtime")
    runtime.memory.ingest(
        text="Prefer concise embodied replies.",
        memory_type="preference",
        title="Reply style",
        scope={"agent_id": "honxin", "workspace_id": "robot"},
    )
    server = EIBrainRPCServer(runtime, host="127.0.0.1", port=0)
    server.start()
    try:
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
