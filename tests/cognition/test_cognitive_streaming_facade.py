from __future__ import annotations


def test_cognitive_runtime_batch_path_remains_valid() -> None:
    from apps.cognitive_runtime.app import CognitiveRuntimeApp
    from eibrain.protocol.observations import AudioTranscriptFinal

    runtime = CognitiveRuntimeApp()

    actions = runtime.handle_observation(
        AudioTranscriptFinal(ts=1.0, source="test", text="hello eibrain", session_id="s1", actor_id="user-1")
    )

    assert len(actions) == 1
    assert actions[0].kind == "play_speech_action"
    assert "hello" in actions[0].text


def test_cognitive_runtime_stream_facade_emits_reply_deltas_and_final_actions() -> None:
    from apps.cognitive_runtime.app import CognitiveRuntimeApp
    from eibrain.protocol.observations import AudioTranscriptFinal

    class _SentenceReplyDialogue:
        def build_reply_text(self, state, memory, llm_text: str) -> str:
            return "Alpha is ready. Beta follows? Gamma closes!"

    runtime = CognitiveRuntimeApp()
    runtime.planner.dialogue_manager = _SentenceReplyDialogue()

    events = runtime.handle_observation_stream(
        AudioTranscriptFinal(ts=1.0, source="test", text="start voice loop", session_id="s1", actor_id="user-1"),
        round_id="round-7",
        cancellation_token="cancel-7",
    )

    assert events[0]["type"] == "status"
    assert events[0]["status"] == "started"
    assert all(event["round_id"] == "round-7" for event in events)
    assert all(event["cancellation_token"] == "cancel-7" for event in events)

    deltas = [event for event in events if event["type"] == "reply_delta"]
    assert [event["delta"] for event in deltas] == ["Alpha is ready. ", "Beta follows? ", "Gamma closes!"]
    assert "".join(event["delta"] for event in deltas) == runtime.last_reply

    final = events[-1]
    assert final["type"] == "actions_final"
    assert final["status"] == "planned"
    assert final["actions"][0]["kind"] == "play_speech_action"
    assert final["actions"][0]["text"] == runtime.last_reply
    assert final["stage_latency_ms"]["total"] >= 0.0
    assert final["cognitive_latency_ms"]["total"] >= 0.0
    assert final["llm_status"]["provider"] == "echo"


def test_cognitive_runtime_stream_facade_is_truthful_when_no_reply() -> None:
    from apps.cognitive_runtime.app import CognitiveRuntimeApp
    from eibrain.protocol.observations import AudioTranscriptFinal

    runtime = CognitiveRuntimeApp()

    events = runtime.handle_observation_stream(
        AudioTranscriptFinal(ts=1.0, source="test", text="", session_id="s1", actor_id="user-1"),
        round_id="round-empty",
        cancellation_token="cancel-empty",
    )

    assert [event["type"] for event in events] == ["status", "actions_final"]
    assert not any(event.get("delta") for event in events)
    assert events[-1]["status"] == "no_reply"
    assert events[-1]["actions"] == []
    assert events[-1]["reply_text"] == ""
