from __future__ import annotations


def test_speaking_person_is_prioritized_over_motion() -> None:
    from eibrain.cognition.attention.active_attention import rank_attention_candidates

    result = rank_attention_candidates(
        [
            {
                "source": "vision",
                "label": "moving chair",
                "confidence": 0.95,
                "motion_score": 0.9,
            },
            {
                "source": "audio",
                "label": "user",
                "confidence": 0.7,
                "is_speaking": True,
            },
        ]
    )

    assert result.decision == "attend"
    assert result.top is not None
    assert result.top.label == "user"
    assert "speaking" in result.top.reason


def test_risk_is_prioritized_over_known_identity() -> None:
    from eibrain.cognition.attention.active_attention import rank_attention_candidates

    result = rank_attention_candidates(
        [
            {
                "source": "memory",
                "label": "familiar person",
                "confidence": 0.9,
                "identity_known": True,
            },
            {
                "source": "vision",
                "label": "falling object",
                "confidence": 0.75,
                "risk_score": 0.95,
            },
        ]
    )

    assert result.top is not None
    assert result.top.label == "falling object"
    assert "risk" in result.top.reason


def test_stale_visual_signal_is_decayed() -> None:
    from eibrain.cognition.attention.active_attention import rank_attention_candidates

    result = rank_attention_candidates(
        [
            {
                "source": "vision",
                "label": "old person",
                "confidence": 1.0,
                "motion_score": 1.0,
                "recency_s": 30.0,
            },
            {
                "source": "vision",
                "label": "fresh person",
                "confidence": 0.4,
                "motion_score": 0.2,
                "recency_s": 0.2,
            },
        ]
    )

    assert result.top is not None
    assert result.top.label == "fresh person"
    assert result.ranked[0].breakdown["recency_multiplier"] > result.ranked[1].breakdown["recency_multiplier"]


def test_task_relevant_object_beats_ordinary_object() -> None:
    from eibrain.cognition.attention.active_attention import rank_attention_candidates

    result = rank_attention_candidates(
        [
            {
                "source": "vision",
                "label": "ordinary mug",
                "confidence": 0.95,
            },
            {
                "source": "task",
                "label": "requested keys",
                "confidence": 0.65,
                "task_relevance": 1.0,
            },
        ]
    )

    assert result.top is not None
    assert result.top.label == "requested keys"
    assert result.top.source_weights["task"] > 0


def test_empty_candidates_return_idle() -> None:
    from eibrain.cognition.attention.active_attention import rank_attention_candidates

    result = rank_attention_candidates([])

    assert result.decision == "idle"
    assert result.top is None
    assert result.ranked == []
