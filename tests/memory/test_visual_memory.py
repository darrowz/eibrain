from __future__ import annotations

import json

from eibrain.memory.visual_memory import VisualMemoryPolicy


def _context() -> dict[str, object]:
    return {
        "session_id": "vision-session-1",
        "actor_id": "user-1",
        "device_id": "eihead-camera-1",
    }


def test_noise_frame_without_stable_lock_is_not_written() -> None:
    policy = VisualMemoryPolicy()

    candidate = policy.evaluate(
        event={
            "event_type": "frame",
            "event_id": "frame-1",
            "trace_id": "trace-frame-1",
            "timestamp_ms": 1_000,
            "objects": [{"label": "person", "confidence": 0.58}],
        },
        target_lock={"locked": False, "stable_frames": 1},
        follow_score={"score": 0.52, "window_ms": 250},
        context=_context(),
    )

    assert candidate is None


def test_stable_target_presence_builds_memory_trace_and_upsert_candidate() -> None:
    policy = VisualMemoryPolicy()

    candidate = policy.evaluate(
        event={
            "event_type": "frame",
            "event_id": "frame-42",
            "trace_id": "trace-vision-42",
            "timestamp_ms": 42_000,
            "objects": [{"label": "person", "confidence": 0.94}],
        },
        target_lock={
            "locked": True,
            "target_id": "user-1",
            "label": "person",
            "stable_frames": 18,
            "duration_ms": 12_500,
        },
        follow_score={"score": 0.88, "window_ms": 3_000},
        context=_context(),
    )

    assert candidate is not None
    assert candidate["event_type"] == "target_long_present"
    assert candidate["source"] == "eibrain.vision"
    assert candidate["importance"] >= 0.75
    assert candidate["dedupe_key"].startswith("visual_memory:target_long_present:vision-session-1:user-1")

    trace = candidate["memory_trace"]
    assert trace["source"] == "eibrain.vision"
    assert trace["event_reference"] == {
        "protocol": "eiprotocol",
        "event_id": "frame-42",
        "trace_id": "trace-vision-42",
        "event_type": "frame",
        "source": "eibrain.vision",
    }
    assert trace["decision"]["decision"] == "visual_memory_candidate"
    assert trace["decision"]["durable"] is True

    upsert = candidate["upsert_payload"]
    assert upsert["method"] == "memory.upsert"
    assert upsert["params"]["source"] == "eibrain.vision"
    assert upsert["params"]["memory_type"] == "visual_event"
    assert upsert["params"]["scope"] == {"session_id": "vision-session-1", "actor_id": "user-1"}
    assert upsert["params"]["meta"]["dedupe_key"] == candidate["dedupe_key"]
    assert upsert["params"]["meta"]["retention"] == "episode"
    assert upsert["params"]["meta"]["ttl_ms"] == 7 * 24 * 60 * 60 * 1000
    assert upsert["params"]["meta"]["writeback"]["eligible"] is True
    assert upsert["params"]["content"]["target_lock"]["stable_frames"] == 18


def test_repeated_stable_target_presence_is_deduped() -> None:
    policy = VisualMemoryPolicy(dedupe_window_ms=60_000)
    first_event = {
        "event_type": "frame",
        "event_id": "frame-100",
        "trace_id": "trace-vision-100",
        "timestamp_ms": 100_000,
    }
    next_frame_same_target = {
        "event_type": "frame",
        "event_id": "frame-101",
        "trace_id": "trace-vision-101",
        "timestamp_ms": 101_000,
    }
    target_lock = {"locked": True, "target_id": "user-1", "stable_frames": 45, "duration_ms": 30_000}

    first = policy.evaluate(event=first_event, target_lock=target_lock, follow_score={"score": 0.9}, context=_context())
    duplicate = policy.evaluate(
        event=next_frame_same_target,
        target_lock={**target_lock, "duration_ms": 31_000},
        follow_score={"score": 0.91},
        context=_context(),
    )

    assert first is not None
    assert duplicate is None


def test_registered_face_and_device_interaction_are_important_visual_events() -> None:
    policy = VisualMemoryPolicy()

    face_candidate = policy.evaluate(
        event={
            "event_type": "registered_face_recognized",
            "event_id": "face-event-1",
            "trace_id": "trace-face-1",
            "timestamp_ms": 200_000,
            "person_id": "registered-user-1",
            "display_name": "Darrow",
            "confidence": 0.97,
            "source": "eye.face",
        },
        target_lock={"locked": True, "target_id": "registered-user-1", "stable_frames": 6},
        follow_score={"score": 0.82},
        context=_context(),
    )
    interaction_candidate = policy.evaluate(
        event={
            "event_type": "user_device_interaction",
            "event_id": "touch-1",
            "trace_id": "trace-touch-1",
            "timestamp_ms": 205_000,
            "device_id": "eihead-camera-1",
            "interaction": "user adjusted camera angle",
        },
        target_lock={"locked": True, "target_id": "registered-user-1", "stable_frames": 6},
        follow_score={"score": 0.81},
        context=_context(),
    )

    assert face_candidate is not None
    assert face_candidate["event_type"] == "registered_face_recognized"
    assert face_candidate["upsert_payload"]["params"]["memory_type"] == "visual_identity_event"
    assert face_candidate["upsert_payload"]["params"]["meta"]["retention"] == "identity_episode"
    assert face_candidate["importance"] >= 0.9
    assert face_candidate["memory_trace"]["event_reference"]["event_id"] == "face-event-1"

    assert interaction_candidate is not None
    assert interaction_candidate["event_type"] == "user_device_interaction"
    assert interaction_candidate["upsert_payload"]["params"]["memory_type"] == "visual_interaction"
    assert interaction_candidate["upsert_payload"]["params"]["meta"]["retention"] == "episode"
    assert interaction_candidate["importance"] >= 0.8


def test_follow_success_and_failure_are_scored_and_retained_differently() -> None:
    policy = VisualMemoryPolicy()

    success = policy.evaluate(
        event={
            "event_type": "follow_success",
            "event_id": "follow-ok-1",
            "trace_id": "trace-follow-ok-1",
            "timestamp_ms": 300_000,
        },
        target_lock={"locked": True, "target_id": "user-1", "stable_frames": 16},
        follow_score={"score": 0.94, "duration_ms": 4_500},
        context=_context(),
    )
    failure = policy.evaluate(
        event={
            "event_type": "follow_failed",
            "event_id": "follow-failed-1",
            "trace_id": "trace-follow-failed-1",
            "timestamp_ms": 301_000,
            "error": "target lost during pan",
        },
        target_lock={"locked": False, "target_id": "user-1", "lost_duration_ms": 2_500},
        follow_score={"score": 0.18, "duration_ms": 2_500},
        context=_context(),
    )

    assert success is not None
    assert failure is not None
    assert success["event_type"] == "follow_success"
    assert failure["event_type"] == "follow_failed"
    assert failure["importance"] > success["importance"]
    assert success["upsert_payload"]["params"]["meta"]["retention"] == "episode"
    assert failure["upsert_payload"]["params"]["meta"]["retention"] == "adjustment_candidate"
    assert failure["upsert_payload"]["params"]["meta"]["training_candidate"] is True
    assert failure["upsert_payload"]["params"]["meta"]["writeback"]["reason"] == "important_visual_event"


def test_user_feedback_payload_is_json_serializable_training_candidate() -> None:
    policy = VisualMemoryPolicy()

    candidate = policy.evaluate(
        event={
            "event_type": "user_feedback",
            "event_id": "feedback-1",
            "trace_id": "trace-feedback-1",
            "timestamp_ms": 400_000,
            "feedback": "tracking felt jumpy when I moved left",
        },
        target_lock={"locked": True, "target_id": "user-1", "stable_frames": 12},
        follow_score={"score": 0.44},
        context=_context(),
    )

    assert candidate is not None
    assert candidate["event_type"] == "user_feedback"
    assert candidate["importance"] >= 0.95
    assert candidate["upsert_payload"]["params"]["memory_type"] == "visual_feedback"
    assert candidate["upsert_payload"]["params"]["meta"]["retention"] == "training_candidate"
    assert candidate["upsert_payload"]["params"]["meta"]["ttl_ms"] == 90 * 24 * 60 * 60 * 1000
    assert candidate["upsert_payload"]["params"]["meta"]["writeback"]["durable"] is True
    assert candidate["upsert_payload"]["params"]["content"]["user_feedback"] == "tracking felt jumpy when I moved left"

    json.dumps(candidate)
