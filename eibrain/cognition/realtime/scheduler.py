"""High-level facade for realtime cognitive scheduling."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

from .activity import ProactiveActivityManager
from .arbiter import ResponseArbiter
from .fast import FastThinkEngine
from .interruption import InterruptionController
from .slow import SlowReasoner
from .turn import (
    RealtimeTurnManager,
    TurnBlackboard,
)


Clock = Callable[[], float]


class RealtimeCognitiveScheduler:
    """Coordinate partial observation, final decisions, interruption, and snapshots."""

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        turn_manager: RealtimeTurnManager | None = None,
        fast_engine: FastThinkEngine | None = None,
        slow_reasoner: SlowReasoner | None = None,
        activity_manager: ProactiveActivityManager | None = None,
        arbiter: ResponseArbiter | None = None,
        interruption_controller: InterruptionController | None = None,
    ) -> None:
        self._clock = clock or time.time
        self.turn_manager = turn_manager or RealtimeTurnManager(clock=self._clock)
        self.fast_engine = fast_engine or FastThinkEngine()
        self.slow_reasoner = slow_reasoner or SlowReasoner()
        self.activity_manager = activity_manager or ProactiveActivityManager()
        self.arbiter = arbiter or ResponseArbiter()
        self.interruption_controller = interruption_controller or InterruptionController()

    def observe_partial(
        self,
        asr_text: str | None = None,
        *,
        text: str | None = None,
        partial_text: str | None = None,
        round_id: str | None = None,
        cancellation_token: str | None = None,
        persona_context: Mapping[str, Any] | None = None,
        emotion_context: Mapping[str, Any] | None = None,
        environment_context: Mapping[str, Any] | None = None,
        memory_candidates: Sequence[Mapping[str, Any]] | None = None,
        deadline_ms: int = 500,
    ) -> dict[str, Any]:
        observed_text = _first_text(asr_text, text, partial_text)
        turn = self._active_turn(reason="partial_observation")
        self._guard_if_explicit(turn=turn, round_id=round_id, cancellation_token=cancellation_token)
        self._merge_context(
            turn,
            persona_context=persona_context,
            emotion_context=emotion_context,
            environment_context=environment_context,
        )

        self.turn_manager.observe_partial(
            round_id=turn.round_id,
            cancellation_token=turn.cancellation_token,
            asr_text=observed_text,
        )
        fast_result = self.fast_engine.process_partial(turn, observed_text, deadline_ms=deadline_ms)
        fast_payload = _to_dict(fast_result)
        microfeedback = _first_text(
            fast_payload.get("microfeedback"),
            _mapping_text(fast_payload.get("micro_feedback"), "text"),
        )
        intent_hypotheses = _first_list(
            fast_payload.get("intent_hypotheses"),
            fast_payload.get("intent_hints"),
        )
        stored_hypothesis = self.turn_manager.write_fast_hypothesis(
            round_id=turn.round_id,
            cancellation_token=turn.cancellation_token,
            hypothesis={
                "partial_text": observed_text,
                "microfeedback": microfeedback,
                "intent_hypotheses": intent_hypotheses,
                "deadline_ms": fast_payload.get("deadline_ms", deadline_ms),
                "stable": False,
            },
            source="scheduler_fast_lane",
        )
        prefetch = self._prefetch_memory(turn=turn, text=observed_text)
        if memory_candidates:
            prefetch.extend(_normalize_memory_candidates(memory_candidates, source="caller_memory"))
        turn.memory_candidates = _merge_memory_candidates(turn.memory_candidates, prefetch)

        return {
            "round_id": turn.round_id,
            "cancellation_token": turn.cancellation_token,
            "fast": {
                **fast_payload,
                "microfeedback": microfeedback,
                "intent_hypotheses": intent_hypotheses,
                "stored_hypothesis": stored_hypothesis,
            },
            "memory_prefetch": prefetch,
            "turn": turn.to_dict(),
        }

    def observe_final(
        self,
        final_text: str | None = None,
        *,
        text: str | None = None,
        round_id: str | None = None,
        cancellation_token: str | None = None,
        memory_candidates: Sequence[Mapping[str, Any]] | None = None,
        persona_context: Mapping[str, Any] | None = None,
        emotion_context: Mapping[str, Any] | None = None,
        environment_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        observed_text = _first_text(final_text, text)
        turn = self._active_turn(reason="final_observation")
        self._guard_if_explicit(turn=turn, round_id=round_id, cancellation_token=cancellation_token)
        self._merge_context(
            turn,
            persona_context=persona_context,
            emotion_context=emotion_context,
            environment_context=environment_context,
        )
        if memory_candidates:
            incoming = _normalize_memory_candidates(memory_candidates, source="caller_memory")
            turn.memory_candidates = _merge_memory_candidates(incoming, turn.memory_candidates)

        self.turn_manager.finalize_asr(
            round_id=turn.round_id,
            cancellation_token=turn.cancellation_token,
            asr_text=observed_text,
        )
        return {
            "round_id": turn.round_id,
            "cancellation_token": turn.cancellation_token,
            "final_text": turn.asr_final,
            "memory_candidates": [dict(item) for item in turn.memory_candidates],
            "turn": turn.to_dict(),
        }

    def decide(
        self,
        *,
        round_id: str | None = None,
        cancellation_token: str | None = None,
        final_text: str | None = None,
        fast_hypotheses: Sequence[Mapping[str, Any]] | None = None,
        memory_candidates: Sequence[Mapping[str, Any]] | None = None,
        persona_context: Mapping[str, Any] | None = None,
        emotion_context: Mapping[str, Any] | None = None,
        execution_result: Mapping[str, Any] | None = None,
        idle_seconds: float = 0.0,
    ) -> dict[str, Any]:
        turn = self._active_turn(reason="decide")
        requested_round_id = round_id or turn.round_id
        requested_token = cancellation_token or turn.cancellation_token
        self.turn_manager.reject_if_cancelled(
            round_id=requested_round_id,
            cancellation_token=requested_token,
        )
        self._merge_context(turn, persona_context=persona_context, emotion_context=emotion_context)
        if memory_candidates:
            incoming = _normalize_memory_candidates(memory_candidates, source="caller_memory")
            turn.memory_candidates = _merge_memory_candidates(incoming, turn.memory_candidates)

        decision = self.slow_reasoner.decide(
            turn=turn,
            round_id=requested_round_id,
            cancellation_token=requested_token,
            final_text=final_text,
            fast_hypotheses=fast_hypotheses,
            memory_candidates=turn.memory_candidates,
            persona_context=turn.persona_state,
            emotion_context=turn.emotion_state,
            execution_result=execution_result,
        )
        committed = self.turn_manager.commit_stable_decision(
            round_id=requested_round_id,
            cancellation_token=requested_token,
            decision=decision,
        )
        turn.speech_plan = {
            "stable": True,
            "speech_segments": list(committed.get("speech_segments", [])),
            "action_segments": list(committed.get("action_segments", committed.get("action_plan", []))),
            "action_plan": list(committed.get("action_plan", [])),
            "actions": list(committed.get("actions", committed.get("action_plan", []))),
            "language": committed.get("persona", {}).get("language", "zh-CN"),
            "source": "realtime_cognitive_scheduler",
        }
        turn.action_plan = list(committed.get("action_plan", []))
        can_speak = self.arbiter.allow_speaking(self.turn_manager, turn, committed)
        activity = self.activity_manager.propose(
            idle_seconds=idle_seconds,
            emotion_context=turn.emotion_state,
            memory_candidates=turn.memory_candidates,
            execution_result=execution_result,
            round_id=turn.round_id,
            cancellation_token=turn.cancellation_token,
        )
        return {
            "round_id": turn.round_id,
            "cancellation_token": turn.cancellation_token,
            "decision": committed,
            "can_speak": can_speak,
            "speech_segments": list(committed.get("speech_segments", [])),
            "action_plan": list(committed.get("action_plan", [])),
            "proactive_activity": activity,
        }

    def interrupt(self, *, reason: str = "user_interrupt") -> dict[str, Any]:
        return self.interruption_controller.interrupt_and_start_new_round(
            self.turn_manager,
            reason=reason,
        )

    def snapshot(self) -> dict[str, Any]:
        payload = self.turn_manager.status_payload()
        current = payload.get("current") or {}
        payload["scheduler"] = {
            "lane": "realtime_cognitive_scheduler",
            "current_round_id": payload.get("current_round_id"),
            "fast_hypothesis_count": len(current.get("fast_hypotheses") or []),
            "stable_decision_count": len(current.get("stable_decisions") or []),
            "memory_candidate_count": len(current.get("memory_candidates") or []),
        }
        return payload

    def current_turn(self) -> TurnBlackboard | None:
        return self.turn_manager.current_turn()

    def _active_turn(self, *, reason: str) -> TurnBlackboard:
        turn = self.turn_manager.current_turn()
        if turn is None or turn.state != "active" or (turn.cancellation is not None and turn.cancellation.cancelled):
            turn = self.turn_manager.start_round(reason=reason)
        return turn

    def _guard_if_explicit(
        self,
        *,
        turn: TurnBlackboard,
        round_id: str | None,
        cancellation_token: str | None,
    ) -> None:
        if round_id is None and cancellation_token is None:
            return
        self.turn_manager.reject_if_cancelled(
            round_id=round_id or turn.round_id,
            cancellation_token=cancellation_token or turn.cancellation_token,
        )

    def _merge_context(
        self,
        turn: TurnBlackboard,
        *,
        persona_context: Mapping[str, Any] | None = None,
        emotion_context: Mapping[str, Any] | None = None,
        environment_context: Mapping[str, Any] | None = None,
    ) -> None:
        if persona_context:
            turn.persona_state.update(dict(persona_context))
        if emotion_context:
            turn.emotion_state.update(dict(emotion_context))
        if environment_context:
            environment = dict(turn.emotion_state.get("environment", {}))
            environment.update(dict(environment_context))
            turn.emotion_state["environment"] = environment

    def _prefetch_memory(self, *, turn: TurnBlackboard, text: str) -> list[dict[str, Any]]:
        stripped = text.strip()
        if not stripped:
            return []
        if not any(marker in stripped for marker in ("记", "提醒", "上次", "以前", "喜欢", "妈妈", "爸爸")):
            return []
        return [
            {
                "id": f"{turn.round_id}:prefetch:{len(turn.memory_candidates)}",
                "query": stripped,
                "text": stripped,
                "kind": "recall",
                "score": 0.5,
                "source": "scheduler_prefetch",
            }
        ]


def _first_text(*values: str | None) -> str:
    for value in values:
        if value is not None:
            return value
    return ""


def _to_dict(value: Any) -> dict[str, Any]:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    return dict(getattr(value, "__dict__", {}))


def _mapping_text(value: Any, key: str) -> str:
    if isinstance(value, Mapping):
        return str(value.get(key) or "")
    return ""


def _first_list(*values: Any) -> list[Any]:
    for value in values:
        if isinstance(value, list):
            return list(value)
        if isinstance(value, tuple):
            return list(value)
    return []


def _normalize_memory_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    source: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        item = dict(candidate)
        item.setdefault("id", f"{source}:{index}")
        item.setdefault("source", source)
        normalized.append(item)
    return normalized


def _merge_memory_candidates(
    primary: Sequence[Mapping[str, Any]],
    secondary: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in list(primary or []) + list(secondary or []):
        item = dict(candidate)
        key = str(item.get("id") or item.get("query") or item.get("text") or len(merged))
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged
