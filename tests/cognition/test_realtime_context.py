from __future__ import annotations

import json

from eibrain.cognition.realtime import ContextTurnBlackboard, EmotionContextBuilder, PersonaRuntime
from eibrain.cognition.realtime.events import (
    asr_final,
    asr_partial,
    environment,
    prosody,
    user_interrupt,
    vision,
)


def test_realtime_observation_helpers_are_json_ready() -> None:
    observations = [
        asr_partial(round_id="round-1", cancellation_token="tok-1", text="打开"),
        asr_final(round_id="round-1", cancellation_token="tok-1", text="打开灯"),
        vision(round_id="round-1", hints={"attention": "present", "objects": ("lamp",)}),
        prosody(round_id="round-1", hints={"arousal": 0.42, "valence": 0.1}),
        environment(round_id="round-1", hints={"noise_db": 48, "location": "home"}),
        user_interrupt(round_id="round-1", reason="barge_in", interrupted_round_id="round-0"),
    ]

    dumped = json.dumps(observations, ensure_ascii=False)
    loaded = json.loads(dumped)

    assert [item["kind"] for item in loaded] == [
        "asr_partial",
        "asr_final",
        "vision",
        "prosody",
        "environment",
        "user_interrupt",
    ]
    assert loaded[0]["payload"]["text"] == "打开"
    assert loaded[1]["stable"] is True
    assert loaded[2]["payload"]["objects"] == ["lamp"]
    assert loaded[5]["payload"]["interrupted_round_id"] == "round-0"


def test_context_blackboard_appends_lanes_and_snapshots_without_aliasing() -> None:
    blackboard = ContextTurnBlackboard(round_id="round-1", cancellation_token="tok-1")

    observation = blackboard.append_observation(
        asr_partial(round_id="round-1", cancellation_token="tok-1", text="请帮我")
    )
    hypothesis = blackboard.append_hypothesis({"intent": "request_help", "confidence": 0.68})
    blackboard.append_memory({"memory_id": "m-1", "summary": "prefers concise answers"})
    blackboard.append_persona({"persona_id": "gentle_companion"})
    blackboard.append_emotion({"emotion_hint": {"label": "calm"}})
    blackboard.append_speech({"text": "我在听。", "stable": False})
    blackboard.append_action({"type": "look_at_user", "startOffsetMs": 0})

    snapshot = blackboard.snapshot()
    snapshot["observations"][0]["payload"]["text"] = "mutated"

    assert observation["kind"] == "asr_partial"
    assert hypothesis["stable"] is False
    assert blackboard.observations[0]["payload"]["text"] == "请帮我"
    assert snapshot["round_id"] == "round-1"
    assert snapshot["memory"][0]["memory_id"] == "m-1"
    assert snapshot["persona"][-1]["persona_id"] == "gentle_companion"
    assert snapshot["emotion"][-1]["emotion_hint"]["label"] == "calm"
    assert snapshot["speech"][0]["text"] == "我在听。"
    assert snapshot["action"][0]["type"] == "look_at_user"
    assert json.loads(json.dumps(snapshot, ensure_ascii=False))["cancellation_token"] == "tok-1"


def test_persona_runtime_defaults_to_gentle_companion_constraints() -> None:
    persona = PersonaRuntime()
    constraints = persona.constraints()

    assert persona.persona_id == "gentle_companion"
    assert constraints["speaking_style"]["tone"] == "gentle"
    assert constraints["voice_code"] == "gentle_companion_zh_cn"
    assert constraints["emotion_policy"]["de_escalate_on_stress"] is True
    assert constraints["action_style"]["interruptibility"] == "high"
    assert constraints["memory_policy"]["writeback"] == "salient_or_user_requested"
    assert json.loads(json.dumps(persona.to_dict(), ensure_ascii=False))["persona_id"] == "gentle_companion"


def test_emotion_context_builder_merges_prosody_environment_and_vision_hints() -> None:
    builder = EmotionContextBuilder()
    context = builder.build(
        observations=[
            prosody(round_id="round-1", hints={"arousal": 0.86, "valence": -0.3, "stress": 0.8}),
            environment(round_id="round-1", hints={"noise_db": 74, "room": "kitchen"}),
            vision(round_id="round-1", hints={"attention": "present", "face_expression": "tired"}),
        ]
    )

    assert context["emotion_hint"]["label"] == "stressed"
    assert context["emotion_hint"]["sources"] == ["prosody", "vision"]
    assert context["noise_policy"]["mode"] == "reduce_verbal_density"
    assert context["noise_policy"]["reason"] == "high_environment_noise"
    assert context["response_style"]["tone"] == "gentle"
    assert context["response_style"]["pace"] == "slow"
    assert context["response_style"]["brevity"] == "concise"
    assert json.loads(json.dumps(context, ensure_ascii=False))["emotion_hint"]["label"] == "stressed"
