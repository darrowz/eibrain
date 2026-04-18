"""Prompt builder for cognition."""

from eibrain.memory.contracts import MemoryResult
from eibrain.state.embodied import EmbodiedState


class PromptBuilder:
    def build(self, state: EmbodiedState, memory: MemoryResult) -> str:
        transcript = state.world.last_transcript.strip()
        visual_summary = state.world.last_visual_summary.strip()
        summary = memory.summary.strip()
        if summary and transcript:
            return f"[memory] {summary}\n[user] {transcript}"
        if summary and visual_summary:
            return f"[memory] {summary}\n[vision] {visual_summary}"
        if visual_summary:
            return f"[vision] {visual_summary}"
        return transcript
