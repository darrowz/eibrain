from __future__ import annotations


def test_affective_response_keeps_embodied_output_contract() -> None:
    from eibrain.cognition.dialogue.affective_response import AffectiveResponse

    response = AffectiveResponse.from_payload(
        {
            "text": "我在，慢慢说。",
            "emotion": "concerned",
            "speaking_style": "reassuring",
            "gaze_intent": "track_speaker",
            "memory_writeback": True,
            "tags": ["wake_ack", "supportive"],
        }
    )

    assert response.to_dict() == {
        "text": "我在，慢慢说。",
        "emotion": "concerned",
        "speaking_style": "reassuring",
        "gaze_intent": "track_speaker",
        "memory_writeback": True,
        "tags": ["wake_ack", "supportive"],
    }


def test_affective_response_sanitizes_unknown_style_fields() -> None:
    from eibrain.cognition.dialogue.affective_response import AffectiveResponse

    response = AffectiveResponse.from_payload(
        {
            "text": "你好",
            "emotion": "dramatic",
            "speaking_style": "shouting",
            "gaze_intent": "spin",
            "memory_writeback": False,
        }
    )

    assert response.emotion == "warm"
    assert response.speaking_style == "gentle"
    assert response.gaze_intent == "track_speaker"
    assert response.memory_writeback is False
