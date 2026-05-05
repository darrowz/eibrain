"""Low-disturbance proactive activity proposals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class ProactiveActivityManager:
    """Choose silent, visual-only, or spoken proactive activity."""

    def __init__(
        self,
        *,
        min_idle_seconds: float = 30.0,
        visual_idle_seconds: float = 60.0,
        speak_idle_seconds: float = 120.0,
    ) -> None:
        self.min_idle_seconds = min_idle_seconds
        self.visual_idle_seconds = visual_idle_seconds
        self.speak_idle_seconds = speak_idle_seconds

    def propose(
        self,
        *,
        idle_seconds: float,
        emotion_context: Mapping[str, Any] | None = None,
        memory_candidates: Sequence[Mapping[str, Any]] | None = None,
        execution_result: Mapping[str, Any] | None = None,
        round_id: str | None = None,
        cancellation_token: str | None = None,
        allow_speech: bool = True,
        **aliases: Any,
    ) -> dict[str, Any]:
        emotion_context = emotion_context or aliases.get("emotion") or aliases.get("emotion_state") or {}
        memory_candidates = memory_candidates or aliases.get("memories") or aliases.get("memory") or []
        memory_refs = _memory_refs(memory_candidates)
        emotion = dict(emotion_context)
        mood = str(emotion.get("mood") or emotion.get("state") or "neutral").lower()
        execution_failed = execution_result is not None and execution_result.get("ok") is False
        urgency = _urgency(mood=mood, memory_refs=memory_refs, execution_failed=execution_failed)

        if idle_seconds < self.min_idle_seconds:
            return self._proposal(
                channel="silent",
                reason="recent_activity",
                text="",
                urgency=0.0,
                emotion=emotion,
                memory_refs=memory_refs,
                round_id=round_id,
                cancellation_token=cancellation_token,
            )

        if (
            allow_speech
            and idle_seconds >= self.speak_idle_seconds
            and urgency >= 0.65
            and not execution_failed
        ):
            channel = "speak"
        elif idle_seconds >= self.visual_idle_seconds and urgency > 0.0:
            channel = "visual_only"
        else:
            channel = "silent"

        return self._proposal(
            channel=channel,
            reason=_reason(mood=mood, memory_refs=memory_refs, execution_failed=execution_failed),
            text=_text(channel=channel, mood=mood, memory_refs=memory_refs, execution_failed=execution_failed),
            urgency=urgency,
            emotion=emotion,
            memory_refs=memory_refs,
            round_id=round_id,
            cancellation_token=cancellation_token,
        )

    def _proposal(
        self,
        *,
        channel: str,
        reason: str,
        text: str,
        urgency: float,
        emotion: Mapping[str, Any],
        memory_refs: Sequence[Mapping[str, Any]],
        round_id: str | None,
        cancellation_token: str | None,
    ) -> dict[str, Any]:
        return {
            "type": "proactive_activity",
            "round_id": round_id,
            "cancellation_token": cancellation_token,
            "channel": channel,
            "disturbance": "low",
            "should_emit": channel != "silent",
            "requires_user_attention": False,
            "reason": reason,
            "text": text,
            "urgency": round(max(0.0, min(1.0, urgency)), 3),
            "emotion": dict(emotion),
            "memory_refs": [dict(item) for item in memory_refs],
            "source": "proactive_activity_manager",
        }


def _memory_refs(memory_candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for candidate in memory_candidates[:3]:
        ref: dict[str, Any] = {}
        if "id" in candidate:
            ref["id"] = candidate["id"]
        text = candidate.get("text") or candidate.get("summary") or candidate.get("content") or candidate.get("query")
        if text is not None:
            ref["text"] = str(text)
        score = candidate.get("importance", candidate.get("score"))
        if score is not None:
            ref["importance"] = _as_float(score)
        if ref:
            refs.append(ref)
    return refs


def _urgency(*, mood: str, memory_refs: Sequence[Mapping[str, Any]], execution_failed: bool) -> float:
    urgency = 0.0
    if mood in {"sad", "anxious", "stressed", "lonely", "tired"}:
        urgency += 0.35
    if execution_failed:
        urgency += 0.45
    if memory_refs:
        urgency += max(_as_float(item.get("importance", 0.0)) for item in memory_refs) * 0.45
    return min(1.0, urgency)


def _reason(*, mood: str, memory_refs: Sequence[Mapping[str, Any]], execution_failed: bool) -> str:
    if execution_failed:
        return "execution_result_needs_attention"
    if memory_refs and mood in {"sad", "anxious", "stressed", "lonely", "tired"}:
        return "emotion_and_memory_nudge"
    if memory_refs:
        return "memory_nudge"
    if mood in {"sad", "anxious", "stressed", "lonely", "tired"}:
        return "emotion_check_in"
    return "no_low_disturbance_opportunity"


def _text(
    *,
    channel: str,
    mood: str,
    memory_refs: Sequence[Mapping[str, Any]],
    execution_failed: bool,
) -> str:
    if channel == "silent":
        return ""
    if execution_failed:
        return "我刚才的执行没有成功，先在这里留一个低打扰提示。"
    memory_text = memory_refs[0].get("text") if memory_refs else None
    if channel == "speak":
        if memory_text:
            return f"我注意到你可能需要一点提醒：{memory_text}。"
        return "我注意到你可能需要一点帮助，要我继续吗？"
    if memory_text:
        return f"低打扰提示：{memory_text}。"
    if mood in {"sad", "anxious", "stressed", "lonely", "tired"}:
        return "低打扰提示：我可以在你需要时继续陪你。"
    return "低打扰提示已准备。"


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
