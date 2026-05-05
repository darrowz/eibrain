from __future__ import annotations

from typing import Any


def _ids(*suffixes: str):
    from eiprotocol.builders import EventIdFactory

    values = iter(suffixes)
    return EventIdFactory(clock=lambda: "2026-05-05T09:40:00.000+08:00", id_factory=lambda: next(values))


def _assert_strict(event: Any) -> dict[str, Any]:
    from eiprotocol.validation import validate_event_strict

    payload = event.to_dict()
    assert validate_event_strict(payload, known_event_required=True) == []
    return payload


def test_scheduler_snapshot_bridge_emits_directional_realtime_cognition_events() -> None:
    from eibrain.protocol.eiprotocol_bridge import scheduler_snapshot_to_eiprotocol_events

    snapshot = {
        "current_round_id": "round-1",
        "current_cancellation_token": "token-1",
        "current": {
            "round_id": "round-1",
            "cancellation_token": "token-1",
            "emotion_state": {"context_id": "emotion-1", "mood": "curious", "confidence": 0.72},
            "speech_plan": {
                "plan_id": "plan-1",
                "stable": True,
                "speech_segments": [{"text": "I will reply softly.", "emotion": "warm", "startOffsetMs": 0, "stable": True}],
                "action_segments": [{"capabilityId": "neck.pan", "startOffsetMs": 120, "durationMs": 200, "style": "gentle"}],
            },
        },
        "memory_prefetch": [
            {"prefetch_id": "prefetch-1", "query": "user previous snack preference", "reason": "partial_asr_memory_hint"}
        ],
        "proactive_activity": {
            "proposal_id": "activity-1",
            "channel": "visual_only",
            "reason": "memory_nudge",
            "should_emit": True,
            "urgency": 0.66,
        },
        "cancellation": {
            "cancellation_id": "cancel-evt-1",
            "cancelled_round_id": "round-0",
            "cancellation_token": "token-0",
            "reason": "user_barge_in",
            "applied_to": ["slow_reasoner", "tts"],
        },
    }

    events = scheduler_snapshot_to_eiprotocol_events(
        snapshot,
        source={"domain": "eibrain", "instanceId": "honxin"},
        target={"domain": "eihead", "instanceId": "honjia"},
        session_id="session-1",
        ids=_ids(
            "emotion-event",
            "emotion-request",
            "prefetch-event",
            "prefetch-request",
            "plan-event",
            "plan-request",
            "activity-event",
            "activity-request",
            "cancel-event",
            "cancel-request",
        ),
        sequence_start=30,
    )

    payloads = [_assert_strict(event) for event in events]
    assert [payload["name"] for payload in payloads] == [
        "ei.observation.emotion.context",
        "ei.memory.prefetch.requested",
        "ei.dialogue.speech_action.plan",
        "ei.activity.proactive.proposed",
        "ei.dialogue.cancellation.applied",
    ]
    assert [payload["sequence"] for payload in payloads] == [30, 31, 32, 33, 34]
    assert {payload["roundId"] for payload in payloads} == {"round-1"}
    assert payloads[0]["source"]["domain"] == "eihead"
    assert payloads[0]["target"]["domain"] == "eibrain"
    assert payloads[1]["source"]["domain"] == "eibrain"
    assert payloads[1]["target"]["domain"] == "eimemory"
    assert payloads[2]["source"]["domain"] == "eibrain"
    assert payloads[2]["target"]["domain"] == "eihead"
    assert payloads[3]["target"]["domain"] == "eihead"
    assert payloads[4]["target"]["domain"] == "eihead"
    assert payloads[2]["content"]["planId"] == "plan-1"
    assert payloads[4]["content"]["cancelledRoundId"] == "round-0"


def test_scheduler_snapshot_bridge_preserves_real_scheduler_action_segments() -> None:
    from eibrain.cognition.realtime import RealtimeCognitiveScheduler
    from eibrain.protocol.eiprotocol_bridge import scheduler_snapshot_to_eiprotocol_events

    scheduler = RealtimeCognitiveScheduler()
    scheduler.observe_partial("请打开灯")
    scheduler.observe_final("请打开灯")
    scheduler.decide()

    events = scheduler_snapshot_to_eiprotocol_events(
        scheduler.snapshot(),
        source={"domain": "eibrain", "instanceId": "honxin"},
        target={"domain": "eihead", "instanceId": "honjia"},
        session_id="session-1",
        ids=_ids("plan-event", "plan-request"),
        sequence_start=1,
    )

    assert [event.name for event in events] == ["ei.dialogue.speech_action.plan"]
    plan_payload = next(_assert_strict(event) for event in events if event.name == "ei.dialogue.speech_action.plan")
    assert plan_payload["content"]["actionSegments"]
    assert plan_payload["content"]["actionSegments"][0]["capabilityId"] == "action.request"


def test_scheduler_snapshot_bridge_emits_memory_trace_result_and_write_events() -> None:
    from eibrain.protocol.eiprotocol_bridge import scheduler_snapshot_to_eiprotocol_events

    snapshot = {
        "current": {
            "round_id": "round-memory",
            "memory_traces": [
                {
                    "schema": "eibrain.memory.closed_loop_trace.v1",
                    "round_id": "round-memory",
                    "session_id": "session-memory",
                    "recall": {
                        "count": 1,
                        "items": [
                            {
                                "query": "用户偏好简短回复",
                                "summary": "Prefer concise spoken replies.",
                                "selected_count": 1,
                                "selected_records": [
                                    {"record_id": "mem_1", "title": "Reply style", "source": "eibrain.preference"}
                                ],
                                "source_composition": {"eibrain.preference": 1},
                            }
                        ],
                    },
                    "writeback": {
                        "count": 1,
                        "items": [
                            {
                                "status": "ok",
                                "summary": "用户偏好更短的语音回复。",
                                "source": "eibrain.semantic_candidate",
                                "memory_type": "semantic_candidate",
                                "diagnostics": {"record_id": "mem_2"},
                            }
                        ],
                    },
                    "errors": [],
                }
            ],
        },
        "scheduler": {"memory_trace_count": 1},
    }

    events = scheduler_snapshot_to_eiprotocol_events(
        snapshot,
        source={"domain": "eibrain", "instanceId": "honxin"},
        target={"domain": "eihead", "instanceId": "honjia"},
        session_id="session-memory",
        ids=_ids("recall-event", "recall-request", "write-event", "write-request"),
        sequence_start=40,
    )

    payloads = [_assert_strict(event) for event in events]
    assert [payload["name"] for payload in payloads] == [
        "ei.memory.recall.result",
        "ei.memory.write.committed",
    ]
    assert [payload["sequence"] for payload in payloads] == [40, 41]
    assert {payload["roundId"] for payload in payloads} == {"round-memory"}
    assert {payload["source"]["domain"] for payload in payloads} == {"eibrain"}
    assert {payload["target"]["domain"] for payload in payloads} == {"eimemory"}
    assert payloads[0]["content"]["query"] == "用户偏好简短回复"
    assert payloads[0]["content"]["results"][0]["record_id"] == "mem_1"
    assert payloads[0]["content"]["sourceComposition"]["eibrain.preference"] == 1
    assert payloads[1]["content"]["memoryId"] == "mem_2"
    assert payloads[1]["content"]["summary"] == "用户偏好更短的语音回复。"
