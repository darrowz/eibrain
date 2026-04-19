from __future__ import annotations


def test_openclaw_adapter_returns_memory_result_summary() -> None:
    from eibrain.memory.adapters.openclaw import OpenClawMemoryAdapter
    from eibrain.memory.contracts import MemoryQuery

    adapter = OpenClawMemoryAdapter()
    result = adapter.retrieve_context(
        MemoryQuery(query="hello", session_id="s1", actor_id="user-1")
    )

    assert "hello" in result.summary
    assert result.relevant_memories == []


def test_openclaw_adapter_supports_http_json_write_and_read(monkeypatch) -> None:
    import json

    from eibrain.infra.config import OpenClawConfig
    from eibrain.memory.adapters.openclaw import OpenClawMemoryAdapter
    from eibrain.memory.contracts import MemoryQuery

    captured: list[tuple[str, str]] = []

    class _Response:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(self.payload).encode("utf-8")

    def _fake_urlopen(req, timeout=0):
        captured.append((req.get_method(), req.full_url))
        if req.get_method() == "POST" and req.full_url.endswith("/retrieve_context"):
            return _Response({"summary": "remote-summary", "relevant_memories": ["m1"]})
        return _Response({"ok": True})

    monkeypatch.setattr("eibrain.memory.adapters.openclaw.request.urlopen", _fake_urlopen)

    adapter = OpenClawMemoryAdapter(
        OpenClawConfig(provider="http_json", endpoint="https://memory.example", api_key="secret")
    )
    result = adapter.retrieve_context(MemoryQuery(query="hello"))
    adapter.remember_episode(session_id="s1", summary="episode")

    assert result.summary == "remote-summary"
    assert result.relevant_memories == ["m1"]
    assert ("POST", "https://memory.example/retrieve_context") in captured
    assert ("POST", "https://memory.example/remember_episode") in captured


def test_openclaw_adapter_gracefully_degrades_on_http_failure(monkeypatch) -> None:
    from urllib.error import URLError

    from eibrain.infra.config import OpenClawConfig
    from eibrain.memory.adapters.openclaw import OpenClawMemoryAdapter
    from eibrain.memory.contracts import MemoryQuery

    def _failing_urlopen(req, timeout=0):
        raise URLError("network down")

    monkeypatch.setattr("eibrain.memory.adapters.openclaw.request.urlopen", _failing_urlopen)

    adapter = OpenClawMemoryAdapter(
        OpenClawConfig(provider="http_json", endpoint="https://memory.example", api_key="secret")
    )

    result = adapter.retrieve_context(MemoryQuery(query="hello", session_id="s1", actor_id="user-1"))
    adapter.remember_episode(session_id="s1", summary="episode")
    adapter.remember_preference(actor_id="user-1", profile={"tea": "oolong"})

    assert "openclaw-context:hello" in result.summary
    assert adapter.summarize_session("s1") == "episode"
    assert adapter.load_actor_profile("user-1") == {"tea": "oolong"}
