from __future__ import annotations


def test_memory_stores_support_profiles_and_session_summary() -> None:
    from eibrain.memory.episodic.store import EpisodicMemoryStore
    from eibrain.memory.semantic.store import SemanticMemoryStore
    from eibrain.memory.working.store import WorkingMemoryStore

    working = WorkingMemoryStore()
    episodic = EpisodicMemoryStore()
    semantic = SemanticMemoryStore()

    working.remember_turn(session_id="s1", text="hello")
    episodic.remember_episode(session_id="s1", summary="user greeted system")
    semantic.remember_profile(actor_id="user-1", profile={"tone": "warm"})

    assert working.recent_turns("s1") == ["hello"]
    assert episodic.summarize_session("s1") == "user greeted system"
    assert semantic.load_actor_profile("user-1") == {"tone": "warm"}

