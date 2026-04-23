from __future__ import annotations


def test_eimemory_rpc_adapter_posts_recall_request(monkeypatch) -> None:
    import json

    from eibrain.infra.config import OpenClawConfig
    from eibrain.memory.adapters.eimemory_rpc import EIMemoryRPCAdapter
    from eibrain.memory.contracts import MemoryQuery

    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps({"ok": True, "result": {"items": [], "rules": [], "reflections": [], "explanation": {}}}).encode("utf-8")

    def _fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["auth"] = req.get_header("Authorization")
        return _Response()

    monkeypatch.setattr("eibrain.memory.adapters.eimemory_rpc.request.urlopen", _fake_urlopen)

    adapter = EIMemoryRPCAdapter(
        OpenClawConfig(
            provider="eimemory_rpc",
            endpoint="http://127.0.0.1:8091/",
            api_key="secret",
            agent_id="honxin",
            workspace_id="robot",
        )
    )
    adapter.retrieve_context(MemoryQuery(query="where is the user", session_id="session-1", actor_id="user-1"))

    assert captured["url"] == "http://127.0.0.1:8091/"
    assert captured["auth"] == "Bearer secret"
    assert captured["body"] == {
        "method": "memory.recall",
        "params": {
            "query": "where is the user",
            "limit": 8,
            "scope": {
                "agent_id": "honxin",
                "workspace_id": "robot",
                "session_id": "session-1",
                "actor_id": "user-1",
            },
            "task_context": {
                "task_type": "brain.respond",
                "goal": "retrieve memory for embodied response",
            },
        },
    }


def test_eimemory_rpc_adapter_maps_recall_bundle_to_memory_result(monkeypatch) -> None:
    import json

    from eibrain.infra.config import OpenClawConfig
    from eibrain.memory.adapters.eimemory_rpc import EIMemoryRPCAdapter
    from eibrain.memory.contracts import MemoryQuery

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "ok": True,
                    "result": {
                        "items": [
                            {"title": "Reply Style", "summary": "Prefer concise spoken replies."},
                            {"title": "Context", "summary": "The current speaker is in front of the robot."},
                        ],
                        "rules": [{"title": "Speak briefly", "summary": "Keep replies short."}],
                        "reflections": [],
                        "explanation": {"session_summary": "Recent dialogue is concise."},
                    },
                }
            ).encode("utf-8")

    monkeypatch.setattr(
        "eibrain.memory.adapters.eimemory_rpc.request.urlopen",
        lambda req, timeout=0: _Response(),
    )

    adapter = EIMemoryRPCAdapter(OpenClawConfig(provider="eimemory_rpc", endpoint="http://127.0.0.1:8091/"))
    result = adapter.retrieve_context(MemoryQuery(query="reply to user"))

    assert result.summary == (
        "Prefer concise spoken replies. | The current speaker is in front of the robot. | Speak briefly"
    )
    assert result.relevant_memories == [
        "Reply Style: Prefer concise spoken replies.",
        "Context: The current speaker is in front of the robot.",
    ]
    assert result.actor_profile == {}
    assert result.session_summary == "Recent dialogue is concise."
