from __future__ import annotations

from typing import Any


def _ids(*suffixes: str):
    from eiprotocol.builders import EventIdFactory

    values = iter(suffixes)
    return EventIdFactory(clock=lambda: "2026-05-05T09:40:00.000+08:00", id_factory=lambda: next(values))


def _assert_strict(event: Any) -> dict[str, Any]:
    from eiprotocol.validation import validate_event_strict

    payload = event.to_dict()
    assert validate_event_strict(payload, known_event_required=True) == []
    return payload


def test_realtime_cognition_events_are_catalogued_and_strictly_validated() -> None:
    from eiprotocol import EmotionContext, require_event_definition
    from eiprotocol.builders import (
        build_dialogue_cancellation_applied_event,
        build_emotion_context_event,
        build_memory_prefetch_requested_event,
        build_proactive_activity_proposed_event,
        build_speech_action_plan_event,
    )

    expected = {
        "ei.dialogue.speech_action.plan": ("dialogue", "dialogue", True),
        "ei.observation.emotion.context": ("observation", "observation", True),
        "ei.memory.prefetch.requested": ("memory", "memory", True),
        "ei.activity.proactive.proposed": ("dialogue", "activity", True),
        "ei.dialogue.cancellation.applied": ("dialogue", "dialogue", True),
    }
    for name, (event_type, plane, round_scoped) in expected.items():
        definition = require_event_definition(name)
        assert definition.event_type == event_type
        assert definition.plane == plane
        assert definition.round_scoped is round_scoped
        assert definition.realtime is True

    brain = {"domain": "eibrain", "instanceId": "honxin"}
    head = {"domain": "eihead", "instanceId": "honjia"}
    memory = {"domain": "eimemory", "instanceId": "default"}
    common = {"round_id": "round-1", "session_id": "session-1"}

    events = [
        build_emotion_context_event(
            ids=_ids("emotion-event", "emotion-request"),
            context_id="emotion-1",
            mood="curious",
            confidence=0.72,
            signals={"prosody": "rising"},
            environment={"noise": "low"},
            context_source="fast_lane",
            source=head,
            target=brain,
            **common,
        ),
        build_memory_prefetch_requested_event(
            ids=_ids("prefetch-event", "prefetch-request"),
            prefetch_id="prefetch-1",
            query="user previous snack preference",
            reason="partial_asr_memory_hint",
            candidates=[{"id": "mem-1", "score": 0.8}],
            source=brain,
            target=memory,
            **common,
        ),
        build_speech_action_plan_event(
            ids=_ids("plan-event", "plan-request"),
            plan_id="plan-1",
            stable=True,
            speech_segments=[{"text": "I will reply softly.", "emotion": "warm", "startOffsetMs": 0, "stable": True}],
            action_segments=[{"capabilityId": "neck.pan", "startOffsetMs": 120, "durationMs": 200, "style": "gentle"}],
            source=brain,
            target=head,
            **common,
        ),
        build_proactive_activity_proposed_event(
            ids=_ids("activity-event", "activity-request"),
            proposal_id="activity-1",
            channel="visual_only",
            reason="memory_nudge",
            should_emit=True,
            urgency=0.66,
            source=brain,
            target=head,
            **common,
        ),
        build_dialogue_cancellation_applied_event(
            ids=_ids("cancel-event", "cancel-request"),
            cancellation_id="cancel-evt-1",
            cancelled_round_id="round-0",
            cancellation_token="token-0",
            reason="user_barge_in",
            applied_to=["slow_reasoner", "tts"],
            source=brain,
            target=head,
            **common,
        ),
    ]

    assert [event.name for event in events] == [
        "ei.observation.emotion.context",
        "ei.memory.prefetch.requested",
        "ei.dialogue.speech_action.plan",
        "ei.activity.proactive.proposed",
        "ei.dialogue.cancellation.applied",
    ]
    payloads = [_assert_strict(event) for event in events]
    assert payloads[0]["content"]["contextId"] == "emotion-1"
    emotion_context = EmotionContext.from_content(payloads[0]["content"])
    assert emotion_context.to_content() == payloads[0]["content"]
    assert payloads[1]["content"]["query"] == "user previous snack preference"
    assert payloads[2]["content"]["speechSegments"][0]["text"] == "I will reply softly."
    assert payloads[3]["content"]["shouldEmit"] is True
    assert payloads[4]["content"]["cancelledRoundId"] == "round-0"
    assert payloads[0]["source"]["domain"] == "eihead"
    assert payloads[0]["target"]["domain"] == "eibrain"
    assert payloads[1]["target"]["domain"] == "eimemory"
