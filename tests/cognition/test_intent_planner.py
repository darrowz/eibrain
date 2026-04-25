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


def test_intent_planner_respects_no_reply_decision() -> None:
    from eibrain.cognition.planner.intent_planner import IntentPlanner
    from eibrain.memory.contracts import MemoryResult
    from eibrain.protocol.events import CognitiveDecision
    from eibrain.state.embodied import EmbodiedState

    planner = IntentPlanner()
    state = EmbodiedState.create_default().with_transcript(
        text="hello",
        session_id="s1",
        actor_id="user-1",
        ts=1.0,
    )
    decision = CognitiveDecision(
        decision_type="ignore",
        reason="low_salience",
        should_reply=False,
        should_orient=False,
        should_writeback=False,
    )

    intents = planner.plan(state=state, memory=MemoryResult(summary="memory fallback"), decision=decision)

    assert not any(intent.kind == "speak_intent" for intent in intents)


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


def test_dialogue_manager_fallback_is_not_generic_acknowledgement() -> None:
    from eibrain.cognition.dialogue.dialogue_manager import DialogueManager
    from eibrain.memory.contracts import MemoryResult
    from eibrain.state.embodied import EmbodiedState

    state = EmbodiedState.create_default().with_transcript(
        text="介绍一下你自己",
        session_id="s1",
        actor_id="user-1",
        ts=1.0,
    )

    reply = DialogueManager().build_reply_text(state, MemoryResult(), "")

    assert "我听到了" not in reply
    assert "鸿途" in reply or "honjia" in reply


def test_dialogue_manager_trims_long_llm_reply_for_fast_tts() -> None:
    from eibrain.cognition.dialogue.dialogue_manager import DialogueManager
    from eibrain.memory.contracts import MemoryResult
    from eibrain.state.embodied import EmbodiedState

    state = EmbodiedState.create_default().with_transcript(
        text="介绍一下你自己",
        session_id="s1",
        actor_id="user-1",
        ts=1.0,
    )

    reply = DialogueManager().build_reply_text(
        state,
        MemoryResult(),
        "我是 honjia 的语音助手，随时帮你查询、提醒和解答问题。",
    )

    assert len(reply) <= 29
    assert reply.endswith("。")
