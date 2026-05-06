from __future__ import annotations

from eibrain.cognition.vision_voice_context import build_vision_voice_context


def _person(
    *,
    label: str = "person",
    stable_id: str = "person:01",
    region: str = "left_middle",
    confidence: float = 0.92,
) -> dict[str, object]:
    return {
        "label": label,
        "stable_id": stable_id,
        "trackId": stable_id,
        "region": region,
        "confidence": confidence,
        "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.8},
    }


def test_person_face_and_locked_target_become_grounded_dialogue_context() -> None:
    context = build_vision_voice_context(
        visual_state={"updated_at_ts": 100.0, "frame_id": "frame-100"},
        scene={
            "summary": "Observed face, person",
            "objects": [
                _person(),
                _person(label="face", stable_id="face:01", region="left_top", confidence=0.88),
            ],
        },
        target={
            "locked": True,
            "track_id": "person:01",
            "label": "person",
            "region": "left_middle",
            "distance_m": 1.7,
            "bearing": "left",
            "confidence": 0.91,
            "following": True,
        },
        now_ts=101.0,
        stale_after_s=3.0,
    )

    assert context["has_person"] is True
    assert context["has_face"] is True
    assert context["person_count"] == 1
    assert context["face_count"] == 1
    assert context["target"]["status"] == "locked"
    assert context["target"]["region"] == "left_middle"
    assert context["target"]["distance_m"] == 1.7
    assert context["target"]["bearing"] == "left"
    assert context["tracking"]["following"] is True
    assert context["reference"]["can_resolve_deictic"] is True
    assert context["reference"]["can_use_this"] is True
    assert context["reference"]["can_use_there"] is True
    assert context["reference"]["can_use_look"] is True
    assert context["freshness"]["stale"] is False
    assert context["reliability"] == 0.91
    assert "person at left_middle" in context["dialogue_context_text"]
    assert "target locked left_middle" in context["dialogue_context_text"]


def test_no_people_context_does_not_claim_referable_visual_target() -> None:
    context = build_vision_voice_context(
        visual_state={"updated_at_ts": 200.0},
        scene={"summary": "Observed cup", "objects": [_person(label="cup", stable_id="cup:01")]},
        now_ts=200.5,
    )

    assert context["has_person"] is False
    assert context["has_face"] is False
    assert context["person_count"] == 0
    assert context["target"]["status"] == "none"
    assert context["reference"]["can_resolve_deictic"] is False
    assert context["reference"]["can_use_this"] is False
    assert context["reference"]["can_use_look"] is False
    assert context["reference"]["reason"] == "no_live_person_or_target"
    assert "no person visible" in context["dialogue_context_text"]


def test_zero_target_metrics_are_preserved_as_observed_values() -> None:
    context = build_vision_voice_context(
        visual_state={"updated_at_ts": 250.0},
        scene={"objects": [_person()]},
        target={
            "locked": True,
            "track_id": "person:01",
            "label": "person",
            "distance_m": 0.0,
            "last_seen_age_s": 0.0,
        },
        now_ts=250.0,
    )

    assert context["target"]["distance_m"] == 0.0
    assert context["target"]["last_seen_age_s"] == 0.0


def test_lost_target_is_preserved_but_not_usable_for_this_or_that() -> None:
    context = build_vision_voice_context(
        visual_state={"updated_at_ts": 300.0},
        scene={"objects": []},
        target={
            "lost": True,
            "track_id": "person:01",
            "label": "person",
            "last_region": "right_middle",
            "last_seen_age_s": 2.4,
            "following": False,
        },
        now_ts=301.0,
    )

    assert context["target"]["status"] == "lost"
    assert context["target"]["track_id"] == "person:01"
    assert context["target"]["region"] == "right_middle"
    assert context["target"]["last_seen_age_s"] == 2.4
    assert context["tracking"]["target_lost"] is True
    assert context["reference"]["can_resolve_deictic"] is False
    assert context["reference"]["reason"] == "target_lost"
    assert "target lost right_middle" in context["dialogue_context_text"]


def test_stale_visual_state_is_marked_and_downweighted() -> None:
    context = build_vision_voice_context(
        visual_state={"updated_at_ts": 10.0},
        scene={"objects": [_person()]},
        target={"locked": True, "track_id": "person:01", "label": "person", "region": "left_middle"},
        now_ts=20.0,
        stale_after_s=3.0,
    )

    assert context["freshness"] == {
        "observed_at_ts": 10.0,
        "age_s": 10.0,
        "stale_after_s": 3.0,
        "stale": True,
    }
    assert context["target"]["status"] == "stale"
    assert context["reference"]["can_resolve_deictic"] is False
    assert context["reference"]["reason"] == "visual_context_stale"
    assert context["reliability"] < 0.5
    assert "stale 10.0s" in context["dialogue_context_text"]


def test_recent_visual_events_are_compacted_into_voice_safe_summary() -> None:
    context = build_vision_voice_context(
        visual_state={"updated_at_ts": 400.0},
        scene={"objects": [_person()]},
        events=[
            {
                "eventType": "appeared",
                "observedAtTs": 399.6,
                "subject": {"label": "person", "trackId": "person:01"},
                "details": {"toRegion": "center_middle"},
            },
            {"type": "moved", "label": "cup", "to_region": "right_middle", "observed_at_ts": 390.0},
            {"type": "attention", "label": "person", "region": "center_middle", "observed_at_ts": 399.9},
        ],
        now_ts=400.0,
        max_events=2,
    )

    assert context["events"] == [
        {
            "type": "attention",
            "label": "person",
            "track_id": "",
            "region": "center_middle",
            "age_s": 0.1,
            "summary": "attention person center_middle",
        },
        {
            "type": "appeared",
            "label": "person",
            "track_id": "person:01",
            "region": "center_middle",
            "age_s": 0.4,
            "summary": "appeared person center_middle",
        },
    ]
    assert "events: attention person center_middle; appeared person center_middle" in context["dialogue_context_text"]
