"""Minimal dialogue manager."""

from eibrain.memory.contracts import MemoryResult
from eibrain.state.embodied import EmbodiedState


class DialogueManager:
    def build_reply_text(self, state: EmbodiedState, memory: MemoryResult, llm_text: str) -> str:
        transcript = state.world.last_transcript.strip()
        if llm_text.strip():
            return llm_text.strip()
        if transcript:
            return f"收到：{transcript}"
        return memory.summary or ""
