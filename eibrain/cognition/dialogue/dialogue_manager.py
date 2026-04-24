"""Minimal dialogue manager."""

from eibrain.memory.contracts import MemoryResult
from eibrain.state.embodied import EmbodiedState


class DialogueManager:
    def build_reply_text(self, state: EmbodiedState, memory: MemoryResult, llm_text: str) -> str:
        transcript = state.world.last_transcript.strip()
        if llm_text.strip():
            return self._prepare_for_speech(llm_text)
        if transcript:
            return self._fallback_reply(transcript)
        return memory.summary or ""

    @staticmethod
    def _prepare_for_speech(text: str, *, max_chars: int = 28) -> str:
        cleaned = " ".join(text.replace("**", "").replace("__", "").split())
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rstrip("，,。 ") + "。"

    @staticmethod
    def _fallback_reply(transcript: str) -> str:
        normalized = transcript.replace("洪图", "鸿途").replace("宏图", "鸿途").strip()
        if any(marker in normalized for marker in ("你叫", "你是", "你的名字", "叫什么")):
            return "我是鸿途。"
        if normalized.endswith(("吗", "么", "嘛", "?","？")):
            return "我先想一下。"
        return "我听到了，继续说。"
