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
        cleaned = " ".join(_repair_mojibake(text).replace("**", "").replace("__", "").split())
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rstrip("，,。 ") + "。"

    @staticmethod
    def _fallback_reply(transcript: str) -> str:
        normalized = transcript.replace("洪图", "鸿途").replace("宏图", "鸿途").strip()
        if any(marker in normalized for marker in ("你叫", "你是", "你的名字", "叫什么")):
            return "我是鸿途。"
        if any(marker in normalized for marker in ("介绍", "自我介绍", "你能做什么", "有什么用")):
            return "我是鸿途，可以陪你对话、看画面和控制设备。"
        if any(marker in normalized for marker in ("天气", "时间", "几点", "日期")):
            return "这个问题需要联网查询，我先记下来。"
        if normalized.endswith(("吗", "么", "嘛", "?","？")):
            return "我现在接口不稳，先给你简短回答：可以。"
        return f"我理解你说的是：{normalized[:18]}。"


def _repair_mojibake(text: str) -> str:
    if not text:
        return text
    latin1_chars = sum(1 for char in text if "\u00a0" <= char <= "\u00ff")
    if latin1_chars == 0:
        return text
    try:
        repaired = text.encode("latin-1").decode("gbk")
    except UnicodeError:
        return text
    if any("\u4e00" <= char <= "\u9fff" for char in repaired):
        return repaired
    return text
