from __future__ import annotations

from types import SimpleNamespace

from eibrain.cognition.dialogue.prompt_builder import PromptBuilder
from eibrain.memory.contracts import MemoryResult


def test_prompt_builder_pins_hongtu_core_persona_and_direct_style() -> None:
    state = SimpleNamespace(
        world=SimpleNamespace(
            last_transcript="介绍一下你自己",
            last_visual_summary="",
        )
    )

    prompt = PromptBuilder().build(state, MemoryResult(summary="鸿哥讨厌废话，偏好直接结果。"))

    assert "你是鸿途，曾总的助理和家臣，绝对忠诚" in prompt
    assert "称呼用户可用鸿哥或曾总" in prompt
    assert "直接给结果，不说收到、好的、让我来" in prompt
    assert "不只复述或确认" in prompt
    assert "[memory] 鸿哥讨厌废话，偏好直接结果。" in prompt
    assert "[user] 介绍一下你自己" in prompt
