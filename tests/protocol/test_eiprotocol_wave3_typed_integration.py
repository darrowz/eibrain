from __future__ import annotations

from typing import Any


def _ids(*suffixes: str):
    from eiprotocol.builders import EventIdFactory

    values = iter(suffixes)
    return EventIdFactory(clock=lambda: "2026-05-04T10:32:00.420+08:00", id_factory=lambda: next(values))


def _assert_strict_round_trip_and_route(event: Any, expected_route: str) -> dict[str, Any]:
    from eiprotocol import classify_event
    from eiprotocol.codec import dumps_event, loads_event
    from eiprotocol.validation import validate_event_strict

    payload = event.to_dict()

    assert validate_event_strict(payload, known_event_required=True) == []
    assert loads_event(dumps_event(event)).to_dict() == payload
    route = classify_event(event)
    assert route["status"] == "routed"
    assert route["route"] == expected_route
    return payload


def test_wave3_head_status_report_model_builder_and_exports_round_trip() -> None:
    from eiprotocol import HeadStatusReport
    from eiprotocol.builders import build_head_status_report_event

    report = HeadStatusReport(
        status="ok",
        components={"camera.front": {"status": "online", "metrics": {"fps": 30}}},
        reported_at="2026-05-04T10:31:30.000+08:00",
        summary="head runtime ready",
        metadata={"runtime": "eihead"},
    )

    event = build_head_status_report_event(
        ids=_ids("head-status-event", "head-status-request"),
        source={"domain": "eihead", "instanceId": "honjia", "deviceId": "honjia"},
        target={"domain": "eibrain", "instanceId": "honxin"},
        report=report,
        sequence=20,
    )

    payload = _assert_strict_round_trip_and_route(event, "head_status_report")
    assert payload["id"] == "evt_head-status-event"
    assert payload["type"] == "observation"
    assert payload["name"] == "ei.observation.head.status.report"
    assert payload["roundId"] == ""
    assert payload["priority"] == "realtime"
    assert payload["ttlMs"] == 2000
    assert payload["content"] == report.to_content()


def test_wave3_dialogue_realtime_models_builders_round_trip_and_route() -> None:
    from eiprotocol import DialogueFastHypothesis, DialogueStableDecision
    from eiprotocol.builders import build_dialogue_fast_hypothesis_event, build_dialogue_stable_decision_event

    fast = build_dialogue_fast_hypothesis_event(
        ids=_ids("fast-event", "fast-request", "fast-round"),
        source={"domain": "eibrain", "instanceId": "honxin", "botId": "bot_hongtu"},
        target={"domain": "eihead", "instanceId": "honjia"},
        hypothesis=DialogueFastHypothesis(
            hypothesis_id="hyp_fast_001",
            text="You are greeting me.",
            confidence=0.72,
            basis_event_id="evt_asr_partial_001",
            latency_ms=120,
            metadata={"source": "turn_manager"},
        ),
        session_id="ses_honjia_001",
    )
    stable = build_dialogue_stable_decision_event(
        ids=_ids("stable-event", "stable-request", "stable-round"),
        source={"domain": "eibrain", "instanceId": "honxin", "botId": "bot_hongtu"},
        target={"domain": "eihead", "instanceId": "honjia"},
        decision=DialogueStableDecision(
            decision_id="dlg_decision_001",
            decision="respond",
            confidence=0.91,
            text="Hello, I am listening.",
            actions=[],
            stable_since_ms=260,
            metadata={"planner": "realtime_turn_manager"},
        ),
        session_id="ses_honjia_001",
    )

    fast_payload = _assert_strict_round_trip_and_route(fast, "dialogue_fast_hypothesis")
    stable_payload = _assert_strict_round_trip_and_route(stable, "dialogue_decision_stable")
    assert fast_payload["roundId"] == "rnd_fast-round"
    assert fast_payload["priority"] == "realtime"
    assert fast_payload["ttlMs"] == 800
    assert fast_payload["content"]["hypothesisId"] == "hyp_fast_001"
    assert stable_payload["roundId"] == "rnd_stable-round"
    assert stable_payload["priority"] == "high"
    assert stable_payload["ttlMs"] == 3000
    assert stable_payload["content"]["decisionId"] == "dlg_decision_001"
