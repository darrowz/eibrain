from __future__ import annotations


def test_intent_planner_generates_speak_intent_from_transcript() -> None:
    from eibrain.cognition.planner.intent_planner import IntentPlanner
    from eibrain.memory.contracts import MemoryResult
    from eibrain.state.embodied import EmbodiedState

    planner = IntentPlanner()
    state = EmbodiedState.create_default().with_transcript(
        text="hello",
        session_id="s1",
        actor_id="user-1",
        ts=1.0,
    )
    memory = MemoryResult(summary="user prefers concise answers")

    intents = planner.plan(state=state, memory=memory)

    assert intents[0].kind == "speak_intent"
    assert intents[0].session_id == "s1"
    assert "hello" in intents[0].text


def test_intent_planner_generates_orient_intent_from_visual_focus() -> None:
    from eibrain.cognition.planner.intent_planner import IntentPlanner
    from eibrain.memory.contracts import MemoryResult
    from eibrain.state.embodied import EmbodiedState

    planner = IntentPlanner()
    state = EmbodiedState.create_default().with_visual_focus(
        target_name="speaker",
        actor_id="user-1",
        summary="a person is looking at the camera",
        ts=1.0,
    )

    intents = planner.plan(state=state, memory=MemoryResult())

    assert any(intent.kind == "orient_intent" for intent in intents)


def test_intent_planner_generates_pause_intent_on_interrupt() -> None:
    from eibrain.cognition.planner.intent_planner import IntentPlanner
    from eibrain.memory.contracts import MemoryResult
    from eibrain.state.embodied import EmbodiedState

    planner = IntentPlanner()
    state = EmbodiedState.create_default().with_interrupt(
        session_id="s1",
        actor_id="user-1",
        ts=2.0,
    )

    intents = planner.plan(state=state, memory=MemoryResult())

    assert len(intents) == 1
    assert intents[0].kind == "pause_intent"


def test_dialogue_manager_prepares_llm_text_for_speech() -> None:
    from eibrain.cognition.dialogue.dialogue_manager import DialogueManager
    from eibrain.memory.contracts import MemoryResult
    from eibrain.state.embodied import EmbodiedState

    state = EmbodiedState.create_default().with_transcript(
        text="福建有多少个行政村？",
        session_id="s1",
        actor_id="user-1",
        ts=1.0,
    )

    reply = DialogueManager().build_reply_text(
        state,
        MemoryResult(),
        "**福建省** 大约有 1.5 万个行政村。",
    )

    assert reply == "福建省 大约有 1.5 万个行政村。"


def test_dialogue_manager_does_not_echo_when_llm_is_empty() -> None:
    from eibrain.cognition.dialogue.dialogue_manager import DialogueManager
    from eibrain.memory.contracts import MemoryResult
    from eibrain.state.embodied import EmbodiedState

    state = EmbodiedState.create_default().with_transcript(
        text="你叫洪图",
        session_id="s1",
        actor_id="user-1",
        ts=1.0,
    )

    reply = DialogueManager().build_reply_text(state, MemoryResult(), "")

    assert reply == "我是鸿途。"
