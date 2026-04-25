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


def test_eimemory_rpc_adapter_merges_custom_task_context(monkeypatch) -> None:
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
            return json.dumps({"ok": True, "result": {"items": [], "rules": [], "explanation": {}}}).encode("utf-8")

    def _fake_urlopen(req, timeout=0):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Response()

    monkeypatch.setattr("eibrain.memory.adapters.eimemory_rpc.request.urlopen", _fake_urlopen)

    adapter = EIMemoryRPCAdapter(OpenClawConfig(provider="eimemory_rpc", endpoint="http://127.0.0.1:8091/"))
    adapter.retrieve_context(
        MemoryQuery(
            query="hello",
            task_context={
                "goal": "respond to spoken user input",
                "organ": "ear",
                "modality": "audio_text",
                "recall_profile": "precision",
            },
        )
    )

    task_context = captured["body"]["params"]["task_context"]
    assert task_context["task_type"] == "brain.respond"
    assert task_context["goal"] == "respond to spoken user input"
    assert task_context["organ"] == "ear"
    assert task_context["modality"] == "audio_text"
    assert task_context["recall_profile"] == "precision"


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


def test_eimemory_rpc_adapter_posts_memory_ingest_for_episode(monkeypatch) -> None:
    import json

    from eibrain.infra.config import OpenClawConfig
    from eibrain.memory.adapters.eimemory_rpc import EIMemoryRPCAdapter

    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps({"ok": True, "result": {"record_id": "mem_1"}}).encode("utf-8")

    def _fake_urlopen(req, timeout=0):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Response()

    monkeypatch.setattr("eibrain.memory.adapters.eimemory_rpc.request.urlopen", _fake_urlopen)

    adapter = EIMemoryRPCAdapter(
        OpenClawConfig(
            provider="eimemory_rpc",
            endpoint="http://127.0.0.1:8091/",
            agent_id="honxin",
            workspace_id="honjia",
        )
    )
    adapter.remember_episode(
        session_id="voice-session",
        actor_id="user-1",
        summary="user:hello | reply:hi",
        title="Audio dialogue turn",
        memory_type="conversation",
        source="eibrain.audio_dialogue",
        modality="audio_text",
        organ="ear",
        outcome={"success": True, "status": "planned", "action_count": 1},
    )

    assert captured["body"] == {
        "method": "memory.ingest",
        "params": {
            "text": "user:hello | reply:hi",
            "title": "Audio dialogue turn",
            "memory_type": "conversation",
            "source": "eibrain.audio_dialogue",
            "scope": {
                "agent_id": "honxin",
                "workspace_id": "honjia",
                "session_id": "voice-session",
                "actor_id": "user-1",
            },
            "organ": "ear",
            "modality": "audio_text",
            "outcome": {"success": True, "status": "planned", "action_count": 1},
        },
    }


def test_eimemory_rpc_adapter_observes_failed_episode_as_incident(monkeypatch) -> None:
    import json

    from eibrain.infra.config import OpenClawConfig
    from eibrain.memory.adapters.eimemory_rpc import EIMemoryRPCAdapter

    captured: list[dict[str, object]] = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps({"ok": True, "result": {"record_id": "mem_1"}}).encode("utf-8")

    def _fake_urlopen(req, timeout=0):
        captured.append(json.loads(req.data.decode("utf-8")))
        return _Response()

    monkeypatch.setattr("eibrain.memory.adapters.eimemory_rpc.request.urlopen", _fake_urlopen)

    adapter = EIMemoryRPCAdapter(OpenClawConfig(provider="eimemory_rpc", endpoint="http://127.0.0.1:8091/"))
    adapter.remember_episode(
        session_id="voice-session",
        actor_id="user-1",
        summary="user:hello | reply:",
        source="eibrain.audio_dialogue",
        modality="audio_text",
        organ="ear",
        outcome={"success": False, "status": "failed", "action_count": 0},
    )

    assert [payload["method"] for payload in captured] == ["memory.ingest", "evolution.observe"]
    incident = captured[1]["params"]
    assert incident["signal_type"] == "incident"
    assert incident["payload"]["status"] == "failed"
    assert incident["payload"]["organ"] == "ear"


def test_eimemory_rpc_adapter_gets_active_policy(monkeypatch) -> None:
    import json

    from eibrain.infra.config import OpenClawConfig
    from eibrain.memory.adapters.eimemory_rpc import EIMemoryRPCAdapter

    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {"ok": True, "result": {"retrieval_policy": {"recall_profile": "precision"}}}
            ).encode("utf-8")

    def _fake_urlopen(req, timeout=0):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Response()

    monkeypatch.setattr("eibrain.memory.adapters.eimemory_rpc.request.urlopen", _fake_urlopen)

    adapter = EIMemoryRPCAdapter(OpenClawConfig(provider="eimemory_rpc", endpoint="http://127.0.0.1:8091/"))
    policy = adapter.get_active_policy(task_type="brain.respond", session_id="s1", actor_id="user-1")

    assert policy == {"retrieval_policy": {"recall_profile": "precision"}}
    assert captured["body"]["method"] == "evolution.get_active_policy"
    assert captured["body"]["params"]["task_type"] == "brain.respond"


def test_eimemory_rpc_adapter_observe_posts_evolution_observe(monkeypatch) -> None:
    import json

    from eibrain.infra.config import OpenClawConfig
    from eibrain.memory.adapters.eimemory_rpc import EIMemoryRPCAdapter

    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps({"ok": True, "result": {"record_id": "obs_1"}}).encode("utf-8")

    def _fake_urlopen(req, timeout=0):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Response()

    monkeypatch.setattr("eibrain.memory.adapters.eimemory_rpc.request.urlopen", _fake_urlopen)

    adapter = EIMemoryRPCAdapter(OpenClawConfig(provider="eimemory_rpc", endpoint="http://127.0.0.1:8091/"))
    adapter.observe("cognitive_turn", {"action_count": 1}, session_id="s1", actor_id="user-1")

    assert captured["body"]["method"] == "evolution.observe"
    assert captured["body"]["params"]["signal_type"] == "cognitive_turn"
    assert captured["body"]["params"]["payload"] == {"action_count": 1}


def test_eimemory_rpc_adapter_posts_memory_ingest_for_preference(monkeypatch) -> None:
    import json

    from eibrain.infra.config import OpenClawConfig
    from eibrain.memory.adapters.eimemory_rpc import EIMemoryRPCAdapter

    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps({"ok": True, "result": {"record_id": "mem_pref"}}).encode("utf-8")

    def _fake_urlopen(req, timeout=0):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Response()

    monkeypatch.setattr("eibrain.memory.adapters.eimemory_rpc.request.urlopen", _fake_urlopen)

    adapter = EIMemoryRPCAdapter(
        OpenClawConfig(
            provider="eimemory_rpc",
            endpoint="http://127.0.0.1:8091/",
            agent_id="honxin",
            workspace_id="honjia",
        )
    )
    adapter.remember_preference(actor_id="user-1", profile={"style": "concise", "tea": "oolong"})

    assert captured["body"] == {
        "method": "memory.ingest",
        "params": {
            "text": "Preference profile for user-1: style=concise, tea=oolong",
            "title": "Preference profile: user-1",
            "memory_type": "preference",
            "source": "eibrain.preference",
            "scope": {
                "agent_id": "honxin",
                "workspace_id": "honjia",
                "actor_id": "user-1",
            },
            "organ": "cognition",
            "modality": "text",
        },
    }



def test_eimemory_rpc_adapter_gracefully_degrades_on_transport_failure(monkeypatch) -> None:
    from urllib.error import URLError

    from eibrain.infra.config import OpenClawConfig
    from eibrain.memory.adapters.eimemory_rpc import EIMemoryRPCAdapter
    from eibrain.memory.contracts import MemoryQuery

    def _failing_urlopen(req, timeout=0):
        raise URLError("down")

    monkeypatch.setattr("eibrain.memory.adapters.eimemory_rpc.request.urlopen", _failing_urlopen)

    adapter = EIMemoryRPCAdapter(OpenClawConfig(provider="eimemory_rpc", endpoint="http://127.0.0.1:8091/"))
    result = adapter.retrieve_context(MemoryQuery(query="hello", session_id="s1", actor_id="user-1"))

    assert "hello" in result.summary
    assert result.session_summary == ""
