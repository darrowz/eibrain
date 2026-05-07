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
    assert captured["body"]["method"] == "memory.recall"
    params = captured["body"]["params"]
    assert params["query"] == "where is the user"
    assert params["limit"] == 8
    assert params["scope"] == {
        "tenant_id": "default",
        "agent_id": "hongtu",
        "workspace_id": "embodied",
        "user_id": "darrow",
    }
    task_context = params["task_context"]
    assert task_context["task_type"] == "brain.respond"
    assert task_context["goal"] == "retrieve memory for embodied response"
    assert task_context["session_id"] == "session-1"
    assert task_context["channel_id"] == "voice"
    assert task_context["subject_context"]["subject_id"] == "hongtu"
    assert task_context["subject_context"]["canonical_user_id"] == "darrow"
    assert task_context["subject_context"]["actor_id"] == "user-1"
    assert "user-1" in task_context["subject_context"]["user_aliases"]


def test_eimemory_rpc_adapter_recall_uses_hongtu_subject_context_and_scope(monkeypatch) -> None:
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

    adapter = EIMemoryRPCAdapter(
        OpenClawConfig(
            provider="eimemory_rpc",
            endpoint="http://127.0.0.1:8091/",
            agent_id="honxin",
            workspace_id="honjia",
        )
    )
    adapter.retrieve_context(
        MemoryQuery(
            query="diagnose prompt bridge",
            session_id="session-1",
            actor_id="user-1",
            task_context={
                "channel_id": "openclaw.feishu",
                "source": "openclaw.before_prompt_build",
            },
        )
    )

    params = captured["body"]["params"]
    assert params["scope"] == {
        "tenant_id": "default",
        "agent_id": "hongtu",
        "workspace_id": "embodied",
        "user_id": "darrow",
    }
    subject = params["task_context"]["subject_context"]
    assert subject["subject_id"] == "hongtu"
    assert subject["channel_id"] == "openclaw.feishu"
    assert subject["canonical_user_id"] == "darrow"
    assert subject["actor_id"] == "user-1"
    assert subject["session_id"] == "session-1"
    assert subject["source"] == "openclaw.before_prompt_build"
    assert subject["memory_layer"] == "trace"
    assert "user-1" in subject["user_aliases"]
    assert params["task_context"]["memory_layer"] == "trace"


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
                "recall_profile": "subject_dialogue",
                "allowed_sources": ["eibrain.audio_dialogue", "eibrain.preference"],
                "blocked_sources": ["eimemory.news", "eimemory.paper", "eimemory.knowledge_base"],
                "privacy": {"scope": "subject_conversation", "sensitivity": "personal"},
                "writeback_eligibility": {"eligible": True, "requires_explicit_memory_request": True},
                "decision_trace": {"decision": "voice_subject_dialogue_recall", "why": "subject memory only"},
            },
        )
    )

    task_context = captured["body"]["params"]["task_context"]
    assert task_context["task_type"] == "brain.respond"
    assert task_context["goal"] == "respond to spoken user input"
    assert task_context["organ"] == "ear"
    assert task_context["modality"] == "audio_text"
    assert task_context["recall_profile"] == "subject_dialogue"
    assert task_context["allowed_sources"] == ["eibrain.audio_dialogue", "eibrain.preference"]
    assert task_context["blocked_sources"] == ["eimemory.news", "eimemory.paper", "eimemory.knowledge_base"]
    assert task_context["privacy"] == {"scope": "subject_conversation", "sensitivity": "personal"}
    assert task_context["writeback_eligibility"] == {"eligible": True, "requires_explicit_memory_request": True}
    assert task_context["decision_trace"] == {
        "decision": "voice_subject_dialogue_recall",
        "why": "subject memory only",
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

    assert adapter.last_writeback_status["status"] == "ok"
    assert adapter.last_writeback_status["record_id"] == "mem_1"
    assert captured["body"]["method"] == "memory.ingest"
    params = captured["body"]["params"]
    assert params["text"] == "user:hello | reply:hi"
    assert params["title"] == "Audio dialogue turn"
    assert params["memory_type"] == "conversation"
    assert params["source"] == "eibrain.audio_dialogue"
    assert params["scope"] == {
        "tenant_id": "default",
        "agent_id": "hongtu",
        "workspace_id": "embodied",
        "user_id": "darrow",
    }
    assert params["organ"] == "ear"
    assert params["modality"] == "audio_text"
    assert params["outcome"] == {"success": True, "status": "planned", "action_count": 1}
    assert params["meta"]["session_id"] == "voice-session"
    assert params["meta"]["source_channel"] == "voice.honjia"
    assert params["meta"]["subject_context"]["canonical_user_id"] == "darrow"


def test_eimemory_rpc_adapter_memory_ingest_meta_uses_hongtu_subject_metadata(monkeypatch) -> None:
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
        memory_type="conversation",
        source="eibrain.audio_dialogue",
        meta={"trace_id": "trace-1"},
    )

    params = captured["body"]["params"]
    assert params["scope"] == {
        "tenant_id": "default",
        "agent_id": "hongtu",
        "workspace_id": "embodied",
        "user_id": "darrow",
    }
    assert params["meta"]["trace_id"] == "trace-1"
    assert params["meta"]["session_id"] == "voice-session"
    assert params["meta"]["memory_layer"] == "episodic"
    assert params["meta"]["source_channel"] == "voice.honjia"
    assert params["meta"]["raw_actor_id"] == "user-1"
    assert params["meta"]["actor_alias"] == "user-1"
    subject = params["meta"]["subject_context"]
    assert subject["subject_id"] == "hongtu"
    assert subject["channel_id"] == "voice.honjia"
    assert subject["canonical_user_id"] == "darrow"
    assert subject["source"] == "eibrain.audio_dialogue"
    assert subject["memory_layer"] == "episodic"
    assert "user-1" in subject["user_aliases"]
    assert adapter.last_writeback_status["subject_context"] == subject
    assert adapter.last_writeback_status["memory_layer"] == "episodic"


def test_eimemory_rpc_adapter_passes_structured_episode_fields(monkeypatch) -> None:
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
            return json.dumps({"ok": True, "result": {"record_id": "mem_structured"}}).encode("utf-8")

    def _fake_urlopen(req, timeout=0):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Response()

    monkeypatch.setattr("eibrain.memory.adapters.eimemory_rpc.request.urlopen", _fake_urlopen)

    adapter = EIMemoryRPCAdapter(OpenClawConfig(provider="eimemory_rpc", endpoint="http://127.0.0.1:8091/"))
    adapter.remember_episode(
        session_id="vision:user-1",
        actor_id="user-1",
        summary="person detected near desk",
        memory_type="world_observation",
        source="eibrain.visual_world",
        modality="vision",
        organ="eye",
        content={"objects": [{"label": "person", "confidence": 0.91}]},
        meta={"dedupe_key": "vision:person:desk", "confidence": 0.91},
        tags=["vision", "world_observation"],
        evidence=[{"type": "frame", "url": "https://example.com/frame.jpg"}],
        links=[{"rel": "actor", "id": "user-1"}],
    )

    params = captured["body"]["params"]
    assert params["content"] == {"objects": [{"label": "person", "confidence": 0.91}]}
    assert params["meta"]["dedupe_key"] == "vision:person:desk"
    assert params["meta"]["confidence"] == 0.91
    assert params["meta"]["session_id"] == "vision:user-1"
    assert params["meta"]["source_channel"] == "vision"
    assert params["meta"]["subject_context"]["channel_id"] == "vision"
    assert params["tags"] == ["vision", "world_observation"]
    assert params["evidence"] == [{"type": "frame", "url": "https://example.com/frame.jpg"}]
    assert params["links"] == [{"rel": "actor", "id": "user-1"}]


def test_eimemory_rpc_adapter_passes_closed_loop_writeback_governance_fields(monkeypatch) -> None:
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
            return json.dumps({"ok": True, "result": {"record_id": "mem_governed"}}).encode("utf-8")

    def _fake_urlopen(req, timeout=0):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Response()

    monkeypatch.setattr("eibrain.memory.adapters.eimemory_rpc.request.urlopen", _fake_urlopen)

    adapter = EIMemoryRPCAdapter(OpenClawConfig(provider="eimemory_rpc", endpoint="http://127.0.0.1:8091/"))
    adapter.remember_episode(
        session_id="voice-session",
        actor_id="user-1",
        summary="user prefers shorter replies",
        memory_type="preference",
        source="eibrain.preference",
        idempotency_key="voice-session:evt-1:preference",
        source_event_id="evt-1",
        conflict={
            "status": "resolved",
            "strategy": "merge",
            "conflict_record_ids": ["pref-old"],
        },
        persona_snapshot={
            "actor_id": "user-1",
            "profile_version": "persona-v3",
            "allowed_traits": ["reply_style"],
        },
    )

    params = captured["body"]["params"]
    assert params["idempotency_key"] == "voice-session:evt-1:preference"
    assert params["source_event_id"] == "evt-1"
    assert params["conflict"] == {
        "status": "resolved",
        "strategy": "merge",
        "conflict_record_ids": ["pref-old"],
    }
    assert params["persona_snapshot"] == {
        "actor_id": "user-1",
        "profile_version": "persona-v3",
        "allowed_traits": ["reply_style"],
    }
    assert adapter.last_writeback_status["idempotency_key"] == "voice-session:evt-1:preference"
    assert adapter.last_writeback_status["source_event_id"] == "evt-1"
    assert adapter.last_writeback_status["conflict"]["strategy"] == "merge"
    assert adapter.last_writeback_status["persona_snapshot"]["profile_version"] == "persona-v3"


def test_eimemory_rpc_adapter_posts_policy_candidate_trace_metadata(monkeypatch) -> None:
    import json

    from eibrain.cognition.policy.multimodal_memory import MultimodalMemoryPolicy
    from eibrain.infra.config import OpenClawConfig
    from eibrain.memory.adapters.eimemory_rpc import EIMemoryRPCAdapter

    captured: list[dict[str, object]] = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps({"ok": True, "result": {"record_id": "mem_candidate"}}).encode("utf-8")

    def _fake_urlopen(req, timeout=0):
        captured.append(json.loads(req.data.decode("utf-8")))
        return _Response()

    monkeypatch.setattr("eibrain.memory.adapters.eimemory_rpc.request.urlopen", _fake_urlopen)

    candidate = MultimodalMemoryPolicy().classify_writeback_candidate(
        event_type="action_outcome",
        summary="MoveHeadAction failed with oscillation",
        modality="multimodal_action",
        organ="neck",
        success=False,
        status="failed",
        action_count=1,
        user_feedback="tracking looked unstable",
        suggested_adjustment="increase yaw deadband before moving",
        trace_id="trace-neck-3",
        source_event_id="outcome-3",
    )
    adapter = EIMemoryRPCAdapter(OpenClawConfig(provider="eimemory_rpc", endpoint="http://127.0.0.1:8091/"))
    adapter.remember_episode(
        session_id="head-session",
        actor_id="user-1",
        summary=str(candidate["summary"]),
        title="Head feedback candidate",
        memory_type=str(candidate["memory_type"]),
        source=str(candidate["source"]),
        modality=str(candidate["modality"]),
        organ=str(candidate["organ"]),
        outcome=dict(candidate["outcome"]),
        content=dict(candidate["content"]),
        meta=dict(candidate["meta"]),
        tags=list(candidate["tags"]),
    )

    assert captured[0]["method"] == "memory.ingest"
    params = captured[0]["params"]
    assert params["memory_type"] == "procedural_adjustment_candidate"
    assert params["source"] == "eibrain.procedural_feedback"
    assert params["content"]["suggested_adjustment"] == "increase yaw deadband before moving"
    assert params["meta"]["candidate_types"] == ["procedural", "training"]
    assert params["meta"]["trace_id"] == "trace-neck-3"
    assert params["meta"]["identity_memory"] is False
    assert "training_candidate" in params["tags"]


def test_eimemory_rpc_adapter_posts_world_observation(monkeypatch) -> None:
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
            return json.dumps({"ok": True, "result": {"record_id": "mem_world"}}).encode("utf-8")

    def _fake_urlopen(req, timeout=0):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Response()

    monkeypatch.setattr("eibrain.memory.adapters.eimemory_rpc.request.urlopen", _fake_urlopen)

    adapter = EIMemoryRPCAdapter(OpenClawConfig(provider="eimemory_rpc", endpoint="http://127.0.0.1:8091/"))
    adapter.remember_world_observation(
        session_id="vision:user-1",
        actor_id="user-1",
        summary="person detected near desk",
        content={"objects": [{"label": "person", "confidence": 0.91}]},
        meta={"dedupe_key": "vision:person:desk", "confidence": 0.91},
        tags=["person"],
    )

    params = captured["body"]["params"]
    assert params["memory_type"] == "world_observation"
    assert params["source"] == "eibrain.visual_world"
    assert params["modality"] == "vision"
    assert params["organ"] == "eye"
    assert params["content"] == {"objects": [{"label": "person", "confidence": 0.91}]}
    assert params["meta"]["dedupe_key"] == "vision:person:desk"
    assert params["meta"]["confidence"] == 0.91
    assert params["meta"]["session_id"] == "vision:user-1"
    assert params["meta"]["source_channel"] == "vision"
    assert params["meta"]["subject_context"]["channel_id"] == "vision"
    assert params["tags"] == ["person", "world_observation", "vision"]


def test_eimemory_rpc_adapter_keeps_classified_writeback_failure_graceful(monkeypatch) -> None:
    from urllib.error import URLError

    from eibrain.infra.config import OpenClawConfig
    from eibrain.memory.adapters.eimemory_rpc import EIMemoryRPCAdapter

    def _failing_urlopen(req, timeout=0):
        raise URLError("down")

    monkeypatch.setattr("eibrain.memory.adapters.eimemory_rpc.request.urlopen", _failing_urlopen)

    adapter = EIMemoryRPCAdapter(OpenClawConfig(provider="eimemory_rpc", endpoint="http://127.0.0.1:8091/"))
    adapter.remember_episode(
        session_id="head-session",
        actor_id="user-1",
        summary="MoveHeadAction failed with oscillation",
        memory_type="procedural_adjustment_candidate",
        source="eibrain.procedural_feedback",
        modality="multimodal_action",
        organ="neck",
        meta={
            "trace_id": "trace-neck-4",
            "source_event_id": "outcome-4",
            "candidate_types": ["procedural", "training"],
            "identity_memory": False,
        },
    )

    assert adapter.last_writeback_status["status"] == "error"
    assert adapter.last_writeback_status["source"] == "eibrain.procedural_feedback"
    assert adapter.last_writeback_status["memory_type"] == "procedural_adjustment_candidate"
    assert adapter.last_writeback_status["trace_id"] == "trace-neck-4"
    assert adapter.last_writeback_status["candidate_types"] == ["procedural", "training"]


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
    params = captured["body"]["params"]
    assert params["task_type"] == "brain.respond"
    assert params["scope"] == {
        "tenant_id": "default",
        "agent_id": "hongtu",
        "workspace_id": "embodied",
        "user_id": "darrow",
    }
    assert params["meta"]["session_id"] == "s1"
    assert params["meta"]["subject_context"]["canonical_user_id"] == "darrow"


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
    params = captured["body"]["params"]
    assert params["signal_type"] == "cognitive_turn"
    assert params["payload"] == {"action_count": 1}
    assert params["scope"] == {
        "tenant_id": "default",
        "agent_id": "hongtu",
        "workspace_id": "embodied",
        "user_id": "darrow",
    }
    assert params["meta"]["session_id"] == "s1"
    assert params["meta"]["subject_context"]["canonical_user_id"] == "darrow"


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

    assert captured["body"]["method"] == "memory.ingest"
    params = captured["body"]["params"]
    assert params["text"] == "Preference profile for user-1: style=concise, tea=oolong"
    assert params["title"] == "Preference profile: user-1"
    assert params["memory_type"] == "preference"
    assert params["source"] == "eibrain.preference"
    assert params["scope"] == {
        "tenant_id": "default",
        "agent_id": "hongtu",
        "workspace_id": "embodied",
        "user_id": "darrow",
    }
    assert params["organ"] == "cognition"
    assert params["modality"] == "text"
    assert params["meta"]["source_channel"] == "voice.honjia"
    assert params["meta"]["raw_actor_id"] == "user-1"
    assert params["meta"]["subject_context"]["canonical_user_id"] == "darrow"



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



def test_eimemory_rpc_adapter_records_skill_trace(monkeypatch) -> None:
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
            return json.dumps({"ok": True, "result": {"record_id": "trace_1"}}).encode("utf-8")

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
    result = adapter.record_skill_trace(
        {
            "trace_id": "voice-session",
            "task_type": "brain.respond",
            "input_summary": "hello",
            "selected_skills": ["reply.default"],
            "actions": ["play_speech_action"],
            "outcome": "planned",
            "feedback": "unknown",
            "latency_ms": 12,
        },
        session_id="voice-session",
        actor_id="user-1",
    )

    assert result["ok"] is True
    assert captured["body"]["method"] == "experience.record_skill_trace"
    params = captured["body"]["params"]
    assert params["payload"] == {
        "trace_id": "voice-session",
        "task_type": "brain.respond",
        "input_summary": "hello",
        "selected_skills": ["reply.default"],
        "actions": ["play_speech_action"],
        "outcome": "planned",
        "feedback": "unknown",
        "latency_ms": 12,
    }
    assert params["scope"] == {
        "tenant_id": "default",
        "agent_id": "hongtu",
        "workspace_id": "embodied",
        "user_id": "darrow",
    }
    assert params["meta"]["session_id"] == "voice-session"
    assert params["meta"]["memory_layer"] == "trace"
    assert params["meta"]["source_channel"] == "voice.honjia"
    assert params["meta"]["subject_context"]["canonical_user_id"] == "darrow"


def test_eimemory_rpc_adapter_records_memory_trace(monkeypatch) -> None:
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
            return json.dumps({"ok": True, "result": {"record_id": "memory_trace_1"}}).encode("utf-8")

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
    result = adapter.record_memory_trace(
        {
            "schema": "eibrain.memory.visual_trace.v1",
            "trace_id": "trace-vision-12",
            "task_type": "brain.respond",
            "memory_reads": [{"query": "where is user", "selected_count": 2}],
            "writebacks": [{"memory_type": "world_observation", "record_id": "mem_world"}],
            "closure": {"status": "closed", "reason": "writeback_confirmed"},
        },
        session_id="vision-session",
        actor_id="user-1",
    )

    assert result["ok"] is True
    assert captured["body"]["method"] == "experience.record_memory_trace"
    params = captured["body"]["params"]
    assert params["payload"] == {
        "schema": "eibrain.memory.visual_trace.v1",
        "trace_id": "trace-vision-12",
        "task_type": "brain.respond",
        "memory_reads": [{"query": "where is user", "selected_count": 2}],
        "writebacks": [{"memory_type": "world_observation", "record_id": "mem_world"}],
        "closure": {"status": "closed", "reason": "writeback_confirmed"},
    }
    assert params["scope"] == {
        "tenant_id": "default",
        "agent_id": "hongtu",
        "workspace_id": "embodied",
        "user_id": "darrow",
    }
    assert params["meta"]["session_id"] == "vision-session"
    assert params["meta"]["memory_layer"] == "trace"
    assert params["meta"]["source_channel"] == "vision"
    assert params["meta"]["subject_context"]["canonical_user_id"] == "darrow"
    assert params["meta"]["subject_context"]["channel_id"] == "vision"


def test_eimemory_rpc_adapter_record_memory_trace_degrades_without_endpoint() -> None:
    from eibrain.memory.adapters.eimemory_rpc import EIMemoryRPCAdapter

    adapter = EIMemoryRPCAdapter()

    assert adapter.record_memory_trace({"trace_id": "trace-offline"}) == {}


def test_eimemory_rpc_adapter_record_memory_trace_degrades_on_failure(monkeypatch) -> None:
    from urllib.error import URLError

    from eibrain.infra.config import OpenClawConfig
    from eibrain.memory.adapters.eimemory_rpc import EIMemoryRPCAdapter

    def _failing_urlopen(req, timeout=0):
        raise URLError("down")

    monkeypatch.setattr("eibrain.memory.adapters.eimemory_rpc.request.urlopen", _failing_urlopen)

    adapter = EIMemoryRPCAdapter(OpenClawConfig(provider="eimemory_rpc", endpoint="http://127.0.0.1:8091/"))

    assert adapter.record_memory_trace({"trace_id": "trace-down"}) == {}


def test_fake_eimemory_rpc_adapter_supports_closed_loop_fixtures() -> None:
    from eibrain.memory.adapters.eimemory_rpc import FakeEIMemoryRPCAdapter
    from eibrain.memory.contracts import MemoryQuery, MemoryResult

    adapter = FakeEIMemoryRPCAdapter(
        recall_result=MemoryResult(
            summary="Prefer concise spoken replies.",
            relevant_memories=["Reply Style: Prefer concise spoken replies."],
            recall_diagnostics={
                "selected_count": 1,
                "selected_records": [{"record_id": "mem_fake_1", "title": "Reply Style"}],
            },
        )
    )

    result = adapter.retrieve_context(MemoryQuery(query="reply style", session_id="s1", actor_id="user-1"))
    adapter.remember_episode(
        session_id="s1",
        actor_id="user-1",
        summary="用户偏好简短回复。",
        memory_type="semantic_candidate",
        source="eibrain.semantic_candidate",
        modality="audio_text",
        organ="ear",
    )
    trace_result = adapter.record_memory_trace(
        {
            "trace_id": "trace-fake-1",
            "memory_trace_summary": {
                "prefetch_requested": 1,
                "prefetch_result": 1,
                "write_proposed": 1,
                "write_committed": 1,
                "reply_used": 0,
            },
        },
        session_id="s1",
        actor_id="user-1",
    )

    assert result.summary == "Prefer concise spoken replies."
    assert adapter.queries[0].query == "reply style"
    assert adapter.episodes[0]["memory_type"] == "semantic_candidate"
    assert trace_result["result"]["record_id"] == "memory_trace_1"
    assert adapter.memory_traces[0]["payload"]["memory_trace_summary"]["write_committed"] == 1


def test_eimemory_rpc_adapter_reports_missing_endpoint_writeback_status() -> None:
    from eibrain.memory.adapters.eimemory_rpc import EIMemoryRPCAdapter

    adapter = EIMemoryRPCAdapter()
    adapter.remember_world_observation(
        session_id="vision:user-1",
        actor_id="user-1",
        summary="person detected near desk",
    )

    assert adapter.last_writeback_status["status"] == "error"
    assert adapter.last_writeback_status["source"] == "eibrain.visual_world"
    assert adapter.last_writeback_status["memory_type"] == "world_observation"
    assert adapter.last_writeback_status["modality"] == "vision"
    assert adapter.last_writeback_status["organ"] == "eye"
    assert adapter.last_writeback_status["error"] == "eimemory RPC endpoint is not configured"
