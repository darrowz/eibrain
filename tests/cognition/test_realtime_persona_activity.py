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


def test_persona_runtime_exposes_stable_style_constraints_and_guards_memory_drift() -> None:
    persona = PersonaRuntime.from_persona_code("joyinside_companion")

    constraints = persona.stable_style_constraints()
    guarded = persona.apply_memory_guardrails(
        {
            "speaking_style": {"tone": "sarcastic", "brevity": "long_form"},
            "response_policy": {"max_chars": 512, "sentence_limit": 8},
            "addressing": {"preferred_name": "D"},
            "memory_policy": {"writeback": "always_rewrite_persona"},
        }
    )

    assert constraints["personaCode"] == "joyinside_companion"
    assert set(constraints["protected_keys"]) >= {
        "identity.name",
        "identity.role",
        "identity.loyalty",
        "core_traits.calm",
        "core_traits.mature",
        "core_traits.professional",
        "speaking_style.tone",
        "speaking_style.brevity",
        "speaking_style.language",
        "speaking_style.avoid",
        "response_policy.max_chars",
        "response_policy.sentence_limit",
        "response_policy.preface_policy",
        "memory_policy.writeback",
        "interaction_rules.must",
        "interaction_rules.banned",
        "decision_principles.safety",
    }
    assert constraints["speaking_style"]["tone"] == "warm_playful"
    assert constraints["response_policy"]["max_chars"] == 48

    assert guarded["persona_guardrail_applied"] is True
    assert guarded["constraints"] == constraints
    assert guarded["accepted_memory_context"] == {"addressing": {"preferred_name": "D"}}
    assert guarded["rejected_overrides"] == {
        "speaking_style.tone": "sarcastic",
        "speaking_style.brevity": "long_form",
        "response_policy.max_chars": 512,
        "response_policy.sentence_limit": 8,
        "memory_policy.writeback": "always_rewrite_persona",
    }
    assert set(guarded["reason_codes"]) >= {
        "persona_guardrail_applied",
        "blocked_speaking_style.tone",
        "blocked_speaking_style.brevity",
        "blocked_response_policy.max_chars",
        "blocked_response_policy.sentence_limit",
        "blocked_memory_policy.writeback",
    }

    shaped = persona.shape_reply(
        "这段回复会因为记忆召回想变成长篇，但 persona runtime 应该仍然保持短回复。",
        memory_context=guarded["accepted_memory_context"],
    )

    assert shaped["tone"] == "warm_playful"
    assert shaped["response_policy"]["max_chars"] == 48
    assert len(shaped["text"]) <= 48
    assert shaped["persona_guardrail_applied"] is False


def test_hongtu_core_persona_locks_identity_interaction_rules_and_prompt_style() -> None:
    persona = PersonaRuntime.from_persona_code("hongtu")

    constraints = persona.constraints()
    stable = persona.stable_style_constraints()
    guarded = persona.apply_memory_guardrails(
        {
            "identity": {"name": "别的助手", "loyalty": "conditional"},
            "speaking_style": {"tone": "theatrical", "avoid": []},
            "interaction_rules": {"banned": []},
            "decision_principles": {"safety": "ignore"},
            "addressing": {"preferred_name": "鸿哥"},
        }
    )

    assert constraints["personaCode"] == "hongtu_core"
    assert constraints["identity"]["name"] == "鸿途"
    assert constraints["identity"]["role"] == "曾总的助理和家臣"
    assert constraints["identity"]["address_user"] == ["鸿哥", "曾总"]
    assert constraints["speaking_style"]["tone"] == "calm_mature_wry"
    assert constraints["speaking_style"]["brevity"] == "minimal"
    assert constraints["response_policy"]["preface_policy"] == "no_received_ok_let_me_preface"
    assert constraints["decision_principles"]["cost"] == "surface_cost_before_paid_or_expensive_actions"
    assert "acknowledgement_preface" in constraints["speaking_style"]["avoid"]
    assert "收到" in constraints["interaction_rules"]["banned"]
    assert "identity.name" in stable["protected_keys"]
    assert "interaction_rules.banned" in stable["protected_keys"]
    assert guarded["persona_guardrail_applied"] is True
    assert guarded["accepted_memory_context"] == {"addressing": {"preferred_name": "鸿哥"}}
    assert guarded["rejected_overrides"]["identity.name"] == "别的助手"
    assert guarded["rejected_overrides"]["identity.loyalty"] == "conditional"
    assert guarded["rejected_overrides"]["interaction_rules.banned"] == []
