from __future__ import annotations

import json

from eibrain.cognition.realtime import (
    EmotionContextBuilder,
    PersonaRuntime,
    ProactiveActivityManager,
    RealtimeCognitiveScheduler,
)


def _clock():
    values = iter(
        [
            5000.0,
            5000.05,
            5000.1,
            5000.15,
            5000.2,
            5000.25,
            5000.3,
            5000.35,
            5000.4,
            5000.45,
            5000.5,
            5000.55,
        ]
    )

    def clock() -> float:
        return next(values)

    return clock


def test_persona_runtime_derives_reply_voice_action_and_proactive_policy_from_persona_code() -> None:
    persona = PersonaRuntime.from_persona_code("joyinside_companion")

    constraints = persona.constraints()
    shaped = persona.shape_reply(
        "我听到你现在有很多事情卡在一起，我们先把最关键的一件挑出来，再慢慢处理后面的部分。"
    )

    assert constraints["personaCode"] == "joyinside_companion"
    assert constraints["response_policy"]["max_chars"] == 48
    assert constraints["speaking_style"]["tone"] == "warm_playful"
    assert constraints["voice_code"] == "joyinside_warm_zh_cn"
    assert constraints["action_style"]["motion"] == "small_expressive"
    assert constraints["proactive_policy"]["mode"] == "low_disturbance_check_in"
    assert shaped["voice_code"] == "joyinside_warm_zh_cn"
    assert shaped["tone"] == "warm_playful"
    assert shaped["action_style"]["motion"] == "small_expressive"
    assert len(shaped["text"]) <= constraints["response_policy"]["max_chars"]
    assert json.loads(json.dumps(persona.snapshot(), ensure_ascii=False))["personaCode"] == "joyinside_companion"


def test_emotion_context_builds_stable_state_from_prosody_environment_and_vision() -> None:
    context = EmotionContextBuilder().build(
        prosody={"arousal": 0.22, "valence": -0.42, "energy": "low", "fatigue": 0.82},
        environment={"noise_db": 76, "time_of_day": "night"},
        vision={"face_expression": "tired", "distance_m": 2.4, "attention": "away"},
    )

    state = context["emotion_state"]
    strategy = context["response_strategy"]

    assert state["mood"] == "tired"
    assert state["energy"] == "low"
    assert state["environment"]["noise"] == "high"
    assert state["environment"]["time"] == "night"
    assert state["environment"]["proximity"] == "far"
    assert state["stability"] == "stable_hint"
    assert strategy["tone"] == "gentle"
    assert strategy["brevity"] == "very_concise"
    assert strategy["nonverbal_preferred"] is True
    assert strategy["speech_risk"] == "cautious"
    assert context["response_style"]["tone"] == "gentle"
    assert json.loads(json.dumps(context, ensure_ascii=False))["emotion_state"]["mood"] == "tired"


def test_proactive_activity_uses_disturbance_policy_for_nudges_and_suppression() -> None:
    manager = ProactiveActivityManager()

    quiet = manager.propose(
        idle_seconds=150,
        emotion_context={"emotion_state": {"mood": "neutral", "environment": {"noise": "low"}}},
    )
    low_mood = manager.propose(
        idle_seconds=180,
        emotion_context={"emotion_state": {"mood": "sad", "energy": "low", "environment": {"noise": "low"}}},
    )
    follow_up = manager.propose(
        idle_seconds=75,
        emotion_context={"emotion_state": {"mood": "neutral", "environment": {"noise": "low"}}},
        execution_result={"ok": True, "needs_followup": True, "summary": "确认刚才的灯光调整是否合适"},
    )
    night = manager.propose(
        idle_seconds=240,
        emotion_context={"emotion_state": {"mood": "sad", "environment": {"time": "night", "noise": "low"}}},
        memory_candidates=[{"id": "mem-night", "text": "睡前喝水提醒", "importance": 0.9}],
    )
    noisy = manager.propose(
        idle_seconds=240,
        emotion_context={"emotion_state": {"mood": "sad", "environment": {"noise": "high"}}},
        memory_candidates=[{"id": "mem-noise", "text": "稍后继续聊", "importance": 0.9}],
    )
    interrupted = manager.propose(
        idle_seconds=240,
        emotion_context={"emotion_state": {"mood": "sad", "environment": {"noise": "low"}}},
        memory_candidates=[{"id": "mem-interrupt", "text": "回访提醒", "importance": 0.9}],
        recent_user_interrupt=True,
    )

    assert quiet["channel"] == "visual_only"
    assert quiet["reason"] == "long_quiet_check_in"
    assert quiet["disturbance"] == "low"
    assert low_mood["channel"] == "speak"
    assert low_mood["reason"] == "emotion_check_in"
    assert low_mood["requires_user_attention"] is False
    assert follow_up["channel"] == "visual_only"
    assert follow_up["reason"] == "execution_follow_up"
    assert night["channel"] == "visual_only"
    assert night["speech_suppressed"] is True
    assert night["suppression_reason"] == "night"
    assert noisy["channel"] == "visual_only"
    assert noisy["speech_suppressed"] is True
    assert noisy["suppression_reason"] == "high_noise"
    assert interrupted["channel"] == "silent"
    assert interrupted["should_emit"] is False
    assert interrupted["suppression_reason"] == "recent_user_interrupt"


def test_scheduler_and_fast_payload_expose_persona_emotion_and_proactive_summaries() -> None:
    persona = PersonaRuntime.from_persona_code("joyinside_companion").snapshot()
    emotion = EmotionContextBuilder().build(
        prosody={"fatigue": 0.76, "energy": "low"},
        environment={"noise_level": "low"},
        vision={"face_expression": "tired", "distance_m": 0.8},
    )
    scheduler = RealtimeCognitiveScheduler(clock=_clock())

    partial = scheduler.observe_partial(
        "我有点累",
        persona_context=persona,
        emotion_context=emotion,
    )
    scheduler.observe_final("我有点累")
    result = scheduler.decide(idle_seconds=180)
    snapshot = scheduler.snapshot()

    assert partial["fast"]["context_summary"]["persona"]["personaCode"] == "joyinside_companion"
    assert partial["fast"]["context_summary"]["emotion"]["mood"] == "tired"
    assert partial["fast"]["stored_hypothesis"]["context_summary"]["persona"]["voice_code"] == "joyinside_warm_zh_cn"
    assert result["decision"]["persona"]["voice_code"] == "joyinside_warm_zh_cn"
    assert result["proactive_activity"]["summary"]["channel"] == result["proactive_activity"]["channel"]
    assert snapshot["scheduler"]["persona"]["personaCode"] == "joyinside_companion"
    assert snapshot["scheduler"]["emotion"]["mood"] == "tired"
    assert snapshot["scheduler"]["proactive_activity"]["channel"] == result["proactive_activity"]["channel"]
    assert snapshot["current"]["safety_state"]["proactive_activity"]["round_id"] == result["round_id"]
