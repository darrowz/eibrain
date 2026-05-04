from __future__ import annotations

import json


def test_head_feedback_builds_episodic_execution_record_from_protocol_outcome() -> None:
    from eibrain.memory.head_feedback import build_head_feedback_record
    from eibrain.protocol.outcomes import ActionExecuted

    outcome = ActionExecuted(
        ts=123.0,
        source="eihead.runtime",
        status="ok",
        action_kind="move_head",
        details={"latency_ms": 42, "trace_id": "trace-1", "executed_by": "eihead.neck"},
    )

    record = build_head_feedback_record(
        action={"kind": "MoveHeadAction", "planned_by": "eibrain.attention", "organ": "neck"},
        outcome=outcome,
        feedback={"text": "tracking looks stable"},
        timestamp_ms=123000,
    )

    assert record["memory_kind"] == "episodic"
    assert record["category"] == "head.execution_outcome"
    assert record["what_attempted"] == "MoveHeadAction"
    assert record["planned_by"] == "eibrain.attention"
    assert record["executed_by"] == "eihead.neck"
    assert record["success"] is True
    assert record["latency_ms"] == 42
    assert record["error"] == ""
    assert record["user_feedback"] == "tracking looks stable"
    assert record["suggested_adjustment"] == ""
    assert record["trace_id"] == "trace-1"
    assert record["timestamp_ms"] == 123000
    assert record["persona_memory"] is False
    assert "tracking looks stable" in record["summary"]


def test_head_feedback_marks_suggested_adjustment_as_procedural_candidate() -> None:
    from eibrain.memory.head_feedback import HeadFeedbackRecordBuilder

    record = HeadFeedbackRecordBuilder().build(
        action={"kind": "MoveHeadAction", "planned_by": "eibrain.attention"},
        outcome={"status": "failed", "latency_ms": 80, "error": "oscillation"},
        feedback={"suggested_adjustment": "increase yaw deadband before moving"},
        trace_id="trace-2",
        timestamp_ms=124000,
    )

    assert record["memory_kind"] == "procedural"
    assert record["category"] == "head.procedural_adjustment_candidate"
    assert record["success"] is False
    assert record["error"] == "oscillation"
    assert record["suggested_adjustment"] == "increase yaw deadband before moving"
    assert record["retention"] == "adjustment_candidate"
    assert record["promotion_status"] == "candidate"
    assert record["persona_memory"] is False
    assert "adjustment_candidate" in record["tags"]


def test_head_feedback_keeps_unknown_transient_signal_in_working_memory_kind() -> None:
    from eibrain.memory.head_feedback import build_head_feedback_record

    record = build_head_feedback_record(
        action="WakeWordDetected",
        outcome={},
        feedback=None,
        trace_id="trace-3",
        timestamp_ms=125000,
    )

    assert record["memory_kind"] == "working"
    assert record["category"] == "head.transient_execution_signal"
    assert record["success"] is None
    assert record["retention"] == "short_lived"
    assert record["persona_memory"] is False


def test_head_feedback_builds_eimemory_and_eitraining_payloads_without_network() -> None:
    from eibrain.memory.head_feedback import (
        REQUIRED_FEEDBACK_FIELDS,
        build_eimemory_ingest_params,
        build_eitraining_trace,
        build_head_feedback_record,
    )

    record = build_head_feedback_record(
        action={"kind": "SpeakAction", "planned_by": "eibrain.dialogue", "modality": "audio_text"},
        outcome={"success": True, "latency_ms": 15, "executed_by": "eihead.mouth"},
        feedback="reply felt faster",
        trace_id="trace-4",
        timestamp_ms=126000,
    )

    ingest_params = build_eimemory_ingest_params(record)
    training_payload = build_eitraining_trace(record)

    assert ingest_params["memory_type"] == "head_execution_feedback"
    assert ingest_params["source"] == "eibrain.head_feedback"
    assert ingest_params["meta"]["memory_kind"] == "episodic"
    assert ingest_params["meta"]["persona_memory"] is False
    assert ingest_params["outcome"] == {
        "success": True,
        "latency_ms": 15,
        "error": "",
        "trace_id": "trace-4",
    }
    for field in REQUIRED_FEEDBACK_FIELDS:
        assert field in ingest_params["content"]
    assert training_payload["signal_type"] == "head_execution_feedback"
    assert training_payload["trace_id"] == "trace-4"
    assert training_payload["what_attempted"] == "SpeakAction"

    json.dumps(ingest_params)
    json.dumps(training_payload)


def test_head_feedback_rejects_unknown_memory_kind() -> None:
    import pytest

    from eibrain.memory.head_feedback import build_head_feedback_record

    with pytest.raises(ValueError):
        build_head_feedback_record(
            action={"kind": "MoveHeadAction"},
            outcome={"success": True},
            memory_kind="personality",
        )
