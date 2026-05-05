from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from apps.body_runtime.voice_chain_benchmark import DEFAULT_THRESHOLDS, summarize_voice_chain


@dataclass(frozen=True, slots=True)
class VoiceScenario:
    name: str
    description: str
    turns: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "turnCount": len(self.turns),
            "turns": [dict(turn) for turn in self.turns],
        }


DEFAULT_SCENARIOS = (
    VoiceScenario(
        name="short_chinese",
        description="single short wake-to-reply turn",
        turns=[
            {
                "roundId": "rnd-short-001",
                "wakeToListenMs": 120.0,
                "asrFinalMs": 520.0,
                "firstTokenMs": 360.0,
                "firstAudioMs": 1280.0,
                "interruptStopMs": 180.0,
                "roundLeak": False,
            }
        ],
    ),
    VoiceScenario(
        name="child_fuzzy",
        description="fuzzy child-like utterance with slower ASR finalization",
        turns=[
            {
                "roundId": "rnd-child-001",
                "wakeToListenMs": 180.0,
                "asrFinalMs": 760.0,
                "firstTokenMs": 520.0,
                "firstAudioMs": 1760.0,
                "interruptStopMs": 240.0,
                "roundLeak": False,
            }
        ],
    ),
    VoiceScenario(
        name="playback_barge_in",
        description="user interrupts while TTS playback is active",
        turns=[
            {
                "roundId": "rnd-barge-001",
                "asrFinalMs": 610.0,
                "firstTokenMs": 440.0,
                "firstAudioMs": 1490.0,
                "interruptStopMs": 210.0,
                "interrupted": True,
                "roundLeak": False,
            }
        ],
    ),
    VoiceScenario(
        name="follow_up_turn",
        description="two consecutive follow-up turns without stale round leakage",
        turns=[
            {
                "roundId": "rnd-follow-001",
                "asrFinalMs": 560.0,
                "firstTokenMs": 390.0,
                "firstAudioMs": 1320.0,
                "interruptStopMs": 190.0,
                "roundLeak": False,
            },
            {
                "roundId": "rnd-follow-002",
                "asrFinalMs": 590.0,
                "firstTokenMs": 410.0,
                "firstAudioMs": 1380.0,
                "interruptStopMs": 205.0,
                "roundLeak": False,
            },
        ],
    ),
    VoiceScenario(
        name="network_jitter",
        description="provider/network jitter while staying inside first-audio target",
        turns=[
            {
                "roundId": "rnd-jitter-001",
                "asrFinalMs": 780.0,
                "firstTokenMs": 690.0,
                "firstAudioMs": 1960.0,
                "interruptStopMs": 280.0,
                "roundLeak": False,
            }
        ],
    ),
)


def run_voice_chain_scenarios(
    *,
    scenarios: Iterable[VoiceScenario] | None = None,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    scenario_list = list(DEFAULT_SCENARIOS if scenarios is None else scenarios)
    threshold_values = dict(DEFAULT_THRESHOLDS)
    if thresholds is not None:
        threshold_values.update(dict(thresholds))
    turns = [dict(turn) for scenario in scenario_list for turn in scenario.turns]
    summary = summarize_voice_chain(turns, thresholds=threshold_values)
    failed_metrics = [
        str(name)
        for name, metric in summary.get("metrics", {}).items()
        if isinstance(metric, Mapping) and metric.get("pass") is False
    ]
    round_leak_free = int(summary.get("roundLeakCount", 0) or 0) == 0
    return {
        "schema": "eibrain.voice_chain_scenarios.v1",
        "scenarioCount": len(scenario_list),
        "turnCount": len(turns),
        "thresholds": threshold_values,
        "summary": summary,
        "failedMetrics": failed_metrics,
        "roundLeakFree": round_leak_free,
        "honjiaReady": not failed_metrics and round_leak_free and bool(turns),
        "scenarios": [scenario.to_dict() for scenario in scenario_list],
    }


__all__ = ["DEFAULT_SCENARIOS", "VoiceScenario", "run_voice_chain_scenarios"]
