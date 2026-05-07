"""Prompt builder for cognition."""

from __future__ import annotations

from typing import TYPE_CHECKING

from eibrain.memory.contracts import MemoryResult

if TYPE_CHECKING:
    from eibrain.state.embodied import EmbodiedState


class PromptBuilder:
    def build(self, state: EmbodiedState, memory: MemoryResult) -> str:
        transcript = state.world.last_transcript.strip()
        visual_summary = state.world.last_visual_summary.strip()
        summary = memory.summary.strip()
        instruction = (
            "你是鸿途，曾总的助理和家臣，绝对忠诚；名字固定写作“鸿途”，不要写成宏图、洪图、黄土或honjia。"
            "风格冷静、成熟、冷幽默、高情商、专业；称呼用户可用鸿哥或曾总。"
            "直接给结果，不说收到、好的、让我来，不道歉铺垫，不只复述或确认。"
            "回答必须是一句话，尽量控制在16到24个汉字内，适合快速语音播放。"
        )
        if summary and transcript:
            return f"{instruction}\n[memory] {summary}\n[user] {transcript}"
        if summary and visual_summary:
            return f"{instruction}\n[memory] {summary}\n[vision] {visual_summary}"
        if visual_summary:
            return f"{instruction}\n[vision] {visual_summary}"
        if transcript:
            return f"{instruction}\n[user] {transcript}"
        return instruction
