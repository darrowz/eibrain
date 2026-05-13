from __future__ import annotations

import json


def test_visual_feedback_records_follow_success_for_training() -> None:
    from eibrain.memory.visual_feedback import build_visual_feedback_record

    record = build_visual_feedback_record(
        feedback_type="follow_result",
        subject={"track_id": "person-1", "label": "person"},
        outcome={"success": True, "latency_ms": 180},
        follow_score={"success": True, "score": 0.92, "metrics": {"before_error": 0.24, "after_error": 0.03}},
        round_id="rnd-1",
        session_id="sess-1",
        timestamp_ms=123000,
    )

    assert record["record_type"] == "visual_feedback"
    assert record["feedback_type"] == "follow_result"
    assert record["outcome"] == "success"
    assert record["subject"]["track_id"] == "person-1"
    assert record["before_metrics"]["before_error"] == 0.24
    assert record["after_metrics"]["after_error"] == 0.03
    assert record["importance"] >= 0.7
    assert record["round_id"] == "rnd-1"
    assert record["session_id"] == "sess-1"
    assert record["privacy"]["redacted"] is False
    assert record["writeback"] == {
        "eligible": True,
        "durable": True,
        "reason": "visual_feedback_event",
        "target_memory_type": "visual_feedback",
    }
    json.dumps(record)


def test_visual_feedback_records_follow_failure_and_action_failure() -> None:
    from eibrain.memory.visual_feedback import build_visual_feedback_record

    record = build_visual_feedback_record(
        feedback_type="action_result",
        subject={"track_id": "person-1", "label": "person"},
        outcome={"success": False, "error": "servo_timeout", "latency_ms": 900},
        follow_score={"success": False, "reason": "action_slow", "metrics": {"after_error": 0.31}},
        action={"kind": "pan", "angle_delta": 8},
        round_id="rnd-2",
        session_id="sess-1",
        timestamp_ms=124000,
    )

    assert record["outcome"] == "failure"
    assert record["error"] == "servo_timeout"
    assert record["action"]["kind"] == "pan"
    assert record["importance"] >= 0.8
    assert record["writeback"]["reason"] == "visual_feedback_event"
    assert "servo_timeout" in record["summary"]


def test_visual_feedback_records_user_identity_correction() -> None:
    from eibrain.memory.visual_feedback import build_visual_feedback_record

    record = build_visual_feedback_record(
        feedback_type="identity_correction",
        subject={"track_id": "face-7", "label": "face", "person_id": "wrong-person"},
        user_feedback="不是我，这是小明",
        label_correction={"from": "wrong-person", "to": "xiaoming", "confirmed": True},
        round_id="rnd-3",
        session_id="sess-2",
        timestamp_ms=125000,
    )

    assert record["outcome"] == "corrected"
    assert record["label_correction"] == {"from": "wrong-person", "to": "xiaoming", "confirmed": True}
    assert record["importance"] == 1.0
    assert "xiaoming" in record["summary"]


def test_visual_feedback_records_target_lost() -> None:
    from eibrain.memory.visual_feedback import build_visual_feedback_record

    record = build_visual_feedback_record(
        feedback_type="target_lost",
        subject={"track_id": "person-1", "label": "person"},
        outcome={"success": False, "reason": "target_lost"},
        round_id="rnd-4",
        session_id="sess-3",
        timestamp_ms=126000,
    )

    assert record["feedback_type"] == "target_lost"
    assert record["outcome"] == "lost"
    assert record["importance"] >= 0.7


def test_visual_feedback_redacts_private_image_paths() -> None:
    from eibrain.memory.visual_feedback import build_visual_feedback_record

    record = build_visual_feedback_record(
        feedback_type="user_feedback",
        subject={"track_id": "person-1", "label": "person", "image_path": "/home/darrow/private/frame.jpg"},
        user_feedback="这次别记我的照片",
        frame={"image_path": "/home/darrow/private/frame.jpg", "snapshot_path": "secret.jpg"},
        privacy={"redact_image_paths": True},
        round_id="rnd-5",
        session_id="sess-4",
        timestamp_ms=127000,
    )

    assert record["privacy"]["redacted"] is True
    assert record["subject"]["image_path"] == "<redacted>"
    assert record["frame"]["image_path"] == "<redacted>"
    assert record["frame"]["snapshot_path"] == "<redacted>"
    assert record["writeback"]["eligible"] is True
    json.dumps(record)


def test_visual_feedback_builds_eimemory_and_training_payloads() -> None:
    from eibrain.memory.visual_feedback import (
        build_eimemory_visual_feedback_params,
        build_eitraining_visual_feedback_trace,
        build_visual_feedback_record,
    )

    record = build_visual_feedback_record(
        feedback_type="follow_result",
        subject={"track_id": "person-1", "label": "person", "confidence": 0.91},
        outcome={"success": True},
        follow_score={"score": 0.93, "window_ms": 3_000},
        frame={"scene_id": "scene-6", "frame_id": "frame-6"},
        round_id="rnd-6",
        session_id="sess-5",
        timestamp_ms=128000,
    )

    memory_params = build_eimemory_visual_feedback_params(record)
    training_trace = build_eitraining_visual_feedback_trace(record)

    assert memory_params["memory_type"] == "visual_feedback"
    assert memory_params["source"] == "eibrain.vision_feedback"
    assert memory_params["modality"] == "vision"
    assert memory_params["confidence"] == 0.93
    assert memory_params["content"]["feedback_type"] == "follow_result"
    assert memory_params["content"]["tracking_provenance"] == {
        "track_id": "person-1",
        "round_id": "rnd-6",
        "session_id": "sess-5",
        "follow_score": 0.93,
        "window_ms": 3000,
    }
    assert memory_params["content"]["scene_provenance"] == {
        "scene_id": "scene-6",
        "frame_id": "frame-6",
    }
    assert memory_params["meta"]["writeback"]["durable"] is True
    assert training_trace["signal_type"] == "visual_feedback"
    assert training_trace["round_id"] == "rnd-6"
    json.dumps(memory_params)
    json.dumps(training_trace)


def test_visual_feedback_skips_low_confidence_observed_memory_params() -> None:
    from eibrain.memory.visual_feedback import (
        build_eimemory_visual_feedback_params,
        build_visual_feedback_record,
    )

    record = build_visual_feedback_record(
        feedback_type="observed",
        subject={"track_id": "person-weak", "label": "person", "confidence": 0.2},
        outcome={"success": True},
        round_id="rnd-low",
        session_id="sess-low",
        frame={"source_event_id": "evt-low", "frame_id": "frame-low"},
    )

    assert record["writeback"]["eligible"] is False
    assert build_eimemory_visual_feedback_params(record) is None


def test_visual_feedback_payload_backfills_memory_score_from_legacy_quality() -> None:
    from eibrain.memory.visual_feedback import (
        build_eimemory_visual_feedback_params,
        build_visual_feedback_record,
    )

    record = build_visual_feedback_record(
        feedback_type="follow_result",
        subject={"track_id": "person-2", "label": "person", "confidence": 0.61},
        outcome={"success": True},
        follow_score={"score": 0.64, "window_ms": 2_000},
        round_id="rnd-compat",
        session_id="sess-compat",
        timestamp_ms=129000,
    )
    record["quality"] = {
        "salience_score": 0.44,
        "confidence": 0.61,
        "quality_tier": "candidate",
        "capture_decision": "accept",
    }

    memory_params = build_eimemory_visual_feedback_params(record)
    memory_score = memory_params["meta"]["scoring"]["memory_score_v1"]

    assert memory_score["tier"] == "candidate"
    assert "lifecycle.candidate" in memory_score["labels"]
    assert memory_params["meta"]["quality"]["quality_tier"] == "candidate"
    assert memory_params["meta"]["quality"]["capture_decision"] == "accept"
