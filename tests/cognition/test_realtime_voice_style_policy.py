from __future__ import annotations

from eibrain.cognition.realtime import RealtimeTurnManager
from eibrain.cognition.realtime.persona import PersonaRuntime
from eibrain.cognition.realtime.planner import SpeechActionPlanner


def test_persona_runtime_maps_emotion_state_to_voice_styles() -> None:
    persona = PersonaRuntime.from_persona_code("joyinside_companion")

    warm = persona.voice_style_for_emotion({"mood": "happy", "environment": {"noise": "low"}})
    tired = persona.voice_style_for_emotion({"mood": "tired", "energy": "low"})
    noisy = persona.voice_style_for_emotion({"mood": "neutral", "environment": {"noise": "high"}})
    night = persona.voice_style_for_emotion({"mood": "sad", "environment": {"time": "night"}})

    assert warm["voice_style"] == "warm"
    assert warm["voice_code"] == "joyinside_warm_zh_cn"
    assert warm["emotion"] == "warm"
    assert tired["voice_style"] == "tired"
    assert tired["voice_code"] == "joyinside_soft_zh_cn"
    assert tired["speed"] < warm["speed"]
    assert noisy["voice_style"] == "noisy"
    assert noisy["voice_code"] == "joyinside_clear_zh_cn"
    assert noisy["volume"] > warm["volume"]
    assert night["voice_style"] == "night"
    assert night["voice_code"] == "joyinside_night_zh_cn"
    assert night["volume"] < warm["volume"]


def test_speech_action_planner_adds_voice_style_from_turn_persona_and_emotion_state() -> None:
    manager = RealtimeTurnManager()
    turn = manager.start_round()
    manager.finalize_asr(
        round_id=turn.round_id,
        cancellation_token=turn.cancellation_token,
        asr_text="我有点累",
    )
    turn.persona_state = PersonaRuntime.from_persona_code("joyinside_companion").snapshot()
    turn.emotion_state = {
        "mood": "tired",
        "energy": "low",
        "environment": {"noise": "low"},
    }

    plan = SpeechActionPlanner().plan(turn, speech_text="我们放慢一点，我陪你先休息一下。")

    segment = plan["speech_segments"][0]
    assert segment["emotion"] == "tired"
    assert segment["voice_style"] == "tired"
    assert segment["voice_code"] == "joyinside_soft_zh_cn"
    assert segment["speed"] < 1.0
    assert segment["volume"] < 0.8
    assert plan["speech"][0]["voice_style"] == "tired"
