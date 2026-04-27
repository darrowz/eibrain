"""Pure active attention ranking for multimodal cognitive candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class AttentionCandidate:
    """A candidate target that may deserve cognitive attention."""

    source: str = "unknown"
    modality: str = "unknown"
    label: str = "unknown"
    confidence: float = 1.0
    recency_s: float = 0.0
    motion_score: float = 0.0
    identity_known: bool = False
    is_speaking: bool = False
    risk_score: float = 0.0
    task_relevance: float = 0.0
    world_change: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


CandidateInput = Mapping[str, Any] | AttentionCandidate


@dataclass(frozen=True)
class RankedAttentionCandidate:
    """A scored candidate with explainable score details."""

    candidate: AttentionCandidate
    score: float
    reason: str
    source_weights: Mapping[str, float]
    breakdown: Mapping[str, float]

    @property
    def source(self) -> str:
        return self.candidate.source

    @property
    def modality(self) -> str:
        return self.candidate.modality

    @property
    def label(self) -> str:
        return self.candidate.label


@dataclass(frozen=True)
class AttentionDecision:
    """The ranked attention result and top-level decision."""

    decision: str
    top: RankedAttentionCandidate | None
    ranked: Sequence[RankedAttentionCandidate]
    reason: str
    source_weights: Mapping[str, float]
    score_breakdown: Mapping[str, float]


DEFAULT_FEATURE_WEIGHTS: Mapping[str, float] = {
    "speaking": 4.0,
    "risk": 3.2,
    "world_change": 2.6,
    "task": 2.2,
    "known_identity": 0.8,
    "motion": 0.7,
    "confidence": 0.5,
}

ATTEND_THRESHOLD = 0.05
RECENCY_HALF_LIFE_S = 5.0


def rank_attention_candidates(
    candidates: Sequence[CandidateInput],
    *,
    feature_weights: Mapping[str, float] | None = None,
) -> AttentionDecision:
    """Rank multimodal candidates without side effects or hardware dependencies."""

    weights = dict(DEFAULT_FEATURE_WEIGHTS)
    if feature_weights:
        weights.update(feature_weights)

    ranked = [_score_candidate(_coerce_candidate(candidate), weights) for candidate in candidates]
    ranked.sort(key=lambda item: item.score, reverse=True)

    if not ranked or ranked[0].score < ATTEND_THRESHOLD:
        return AttentionDecision(
            decision="idle",
            top=None,
            ranked=ranked,
            reason="idle",
            source_weights={},
            score_breakdown={},
        )

    top = ranked[0]
    return AttentionDecision(
        decision="attend",
        top=top,
        ranked=ranked,
        reason=top.reason,
        source_weights=top.source_weights,
        score_breakdown=top.breakdown,
    )


def _score_candidate(
    candidate: AttentionCandidate,
    weights: Mapping[str, float],
) -> RankedAttentionCandidate:
    raw_breakdown = {
        "speaking": weights["speaking"] if candidate.is_speaking else 0.0,
        "risk": weights["risk"] * candidate.risk_score,
        "world_change": weights["world_change"] * candidate.world_change,
        "task": weights["task"] * candidate.task_relevance,
        "known_identity": weights["known_identity"] if candidate.identity_known else 0.0,
        "motion": weights["motion"] * candidate.motion_score,
        "confidence": weights["confidence"] * candidate.confidence,
    }
    recency_multiplier = _recency_multiplier(candidate.recency_s)
    score = sum(raw_breakdown.values()) * recency_multiplier
    breakdown = {
        **{key: round(value * recency_multiplier, 6) for key, value in raw_breakdown.items()},
        "recency_multiplier": round(recency_multiplier, 6),
    }
    reason = _build_reason(breakdown)
    source_key = candidate.source or candidate.modality or "unknown"
    return RankedAttentionCandidate(
        candidate=candidate,
        score=round(score, 6),
        reason=reason,
        source_weights={source_key: round(score, 6)},
        breakdown=breakdown,
    )


def _coerce_candidate(candidate: CandidateInput) -> AttentionCandidate:
    if isinstance(candidate, AttentionCandidate):
        return candidate

    known_fields = {
        "source",
        "modality",
        "label",
        "confidence",
        "recency_s",
        "motion_score",
        "identity_known",
        "is_speaking",
        "risk_score",
        "task_relevance",
        "world_change",
    }
    source = str(candidate.get("source") or candidate.get("modality") or "unknown")
    modality = str(candidate.get("modality") or candidate.get("source") or "unknown")
    metadata = {key: value for key, value in candidate.items() if key not in known_fields}
    return AttentionCandidate(
        source=source,
        modality=modality,
        label=str(candidate.get("label") or "unknown"),
        confidence=_clamp_float(candidate.get("confidence", 1.0)),
        recency_s=max(0.0, _to_float(candidate.get("recency_s", 0.0))),
        motion_score=_clamp_float(candidate.get("motion_score", 0.0)),
        identity_known=bool(candidate.get("identity_known", False)),
        is_speaking=bool(candidate.get("is_speaking", False)),
        risk_score=_clamp_float(candidate.get("risk_score", 0.0)),
        task_relevance=_clamp_float(candidate.get("task_relevance", 0.0)),
        world_change=_clamp_float(candidate.get("world_change", 0.0)),
        metadata=metadata,
    )


def _recency_multiplier(recency_s: float) -> float:
    return 1.0 / (1.0 + (recency_s / RECENCY_HALF_LIFE_S))


def _build_reason(breakdown: Mapping[str, float]) -> str:
    active = [
        key
        for key in (
            "speaking",
            "risk",
            "world_change",
            "task",
            "known_identity",
            "motion",
            "confidence",
        )
        if breakdown[key] > 0
    ]
    return ",".join(active) or "baseline"


def _clamp_float(value: Any) -> float:
    return max(0.0, min(1.0, _to_float(value)))


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
