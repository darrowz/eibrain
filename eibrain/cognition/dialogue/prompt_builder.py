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
        instruction = "你是鸿途，可称用户鸿哥或曾总；一句话直接回答，不复述、不解释，24字内。"
        if summary and transcript:
            return f"{instruction}\n[memory] {summary}\n[user] {transcript}"
        if summary and visual_summary:
            return f"{instruction}\n[memory] {summary}\n[vision] {visual_summary}"
        if visual_summary:
            return f"{instruction}\n[vision] {visual_summary}"
        if transcript:
            return f"{instruction}\n[user] {transcript}"
        return instruction
