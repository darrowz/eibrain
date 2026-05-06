from __future__ import annotations

from eibrain.cognition.realtime import EmotionContextBuilder, ProactiveActivityManager


def test_proactive_activity_rate_limits_repeated_spoken_check_ins() -> None:
    manager = ProactiveActivityManager(
        channel_cooldowns={"speak": 300.0, "visual_only": 60.0},
        channel_caps={"speak": (1, 600.0), "visual_only": (3, 600.0)},
        event_type_caps={"emotion_check_in": (1, 600.0)},
    )

    first = manager.propose(
        idle_seconds=180,
        emotion_context={"emotion_state": {"mood": "sad", "environment": {"noise": "low"}}},
        now_seconds=1000.0,
    )
    second = manager.propose(
        idle_seconds=190,
        emotion_context={"emotion_state": {"mood": "sad", "environment": {"noise": "low"}}},
        now_seconds=1010.0,
    )

    assert first["channel"] == "speak"
    assert first["should_emit"] is True
    assert second["channel"] == "silent"
    assert second["should_emit"] is False
    assert second["suppression_reason"] in {"channel_cooldown", "event_type_cap"}
    assert second["next_allowed_at"] >= 1300.0
    assert second["metrics"]["proposed"] == 2
    assert second["metrics"]["emitted"] == 1
    assert second["metrics"]["suppressed"] == 1
    assert second["metrics"]["reason_counts"][second["suppression_reason"]] == 1
    assert second["metrics"]["next_allowed_at"] == second["next_allowed_at"]


def test_proactive_activity_negative_feedback_opens_quiet_budget() -> None:
    manager = ProactiveActivityManager(feedback_cooldown_seconds=900.0)

    rejected = manager.propose(
        idle_seconds=240,
        emotion_context={
            "emotion_state": {"mood": "sad", "environment": {"noise": "low"}},
            "user_feedback": {"rejected": True},
        },
        now_seconds=2000.0,
    )
    still_quiet = manager.propose(
        idle_seconds=360,
        emotion_context={"emotion_state": {"mood": "sad", "environment": {"noise": "low"}}},
        now_seconds=2500.0,
    )

    assert rejected["channel"] == "silent"
    assert rejected["should_emit"] is False
    assert rejected["suppression_reason"] == "negative_feedback_cooldown"
    assert rejected["next_allowed_at"] == 2900.0
    assert still_quiet["channel"] == "silent"
    assert still_quiet["suppression_reason"] == "negative_feedback_cooldown"
    assert still_quiet["next_allowed_at"] == 2900.0
    assert still_quiet["metrics"]["proposed"] == 2
    assert still_quiet["metrics"]["suppressed"] == 2


def test_emotion_strategy_marks_noisy_far_night_as_nonverbal_low_priority() -> None:
    context = EmotionContextBuilder().build(
        prosody={"stress": 0.74, "arousal": 0.78, "valence": -0.35},
        environment={"noise_db": 78, "time_of_day": "night"},
        vision={"distance_m": 2.8, "attention": "away"},
    )

    strategy = context["response_strategy"]
    proactive = context["emotion_state"]["proactive"]

    assert strategy["proactive_priority"] == "defer"
    assert strategy["proactive_disturbance"] == "nonverbal"
    assert proactive["priority"] == "defer"
    assert proactive["disturbance"] == "nonverbal"
    assert proactive["suppress_speech"] is True
    assert proactive["suppression_reasons"] == ["night", "high_noise", "far_distance"]
