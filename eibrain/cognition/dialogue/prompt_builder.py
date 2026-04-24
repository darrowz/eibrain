"""Prompt builder for cognition."""

from eibrain.memory.contracts import MemoryResult
from eibrain.state.embodied import EmbodiedState


class PromptBuilder:
    def build(self, state: EmbodiedState, memory: MemoryResult) -> str:
        transcript = state.world.last_transcript.strip()
        visual_summary = state.world.last_visual_summary.strip()
        summary = memory.summary.strip()
        instruction = (
            "你是 honjia 的本地语音助手。"
            "请用中文自然回答用户问题，不要只复述或确认收到。"
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
