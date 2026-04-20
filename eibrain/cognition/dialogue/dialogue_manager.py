"""Minimal dialogue manager."""

from eibrain.memory.contracts import MemoryResult
from eibrain.state.embodied import EmbodiedState


class DialogueManager:
    def build_reply_text(self, state: EmbodiedState, memory: MemoryResult, llm_text: str) -> str:
        transcript = state.world.last_transcript.strip()
        if llm_text.strip():
            return self._prepare_for_speech(llm_text)
        if transcript:
            return f"收到：{transcript}"
        return memory.summary or ""

    @staticmethod
    def _prepare_for_speech(text: str, *, max_chars: int = 140) -> str:
        cleaned = " ".join(text.replace("**", "").replace("__", "").split())
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rstrip("，,。 ") + "。"
