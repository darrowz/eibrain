from __future__ import annotations

from eibrain.cognition.realtime import RealtimeCognitiveScheduler


def _clock():
    values = iter(
        [
            4000.0,
            4000.05,
            4000.1,
            4000.15,
            4000.2,
            4000.25,
            4000.3,
            4000.35,
            4000.4,
            4000.45,
            4000.5,
            4000.55,
            4000.6,
            4000.65,
        ]
    )

    def clock() -> float:
        return next(values)

    return clock


def _assert_lane_shape(lane: dict[str, object], *, round_id: str, cancellation_token: str) -> None:
    assert lane["round_id"] == round_id
    assert lane["cancellation_token"] == cancellation_token
    assert isinstance(lane["status"], str)
    assert isinstance(lane["latency_ms"], (int, float))
    assert lane["latency_ms"] >= 0
    assert isinstance(lane["pending_count"], int)
    assert isinstance(lane["stable_count"], int)
    assert isinstance(lane["cancellable"], bool)


def test_scheduler_snapshot_exposes_parallel_lane_status_counts_and_latency() -> None:
    scheduler = RealtimeCognitiveScheduler(clock=_clock())

    partial = scheduler.observe_partial("帮我记一下明天")
    scheduler.observe_final("帮我记一下明天给妈妈打电话")
    result = scheduler.decide()
    snapshot = scheduler.snapshot()

    assert snapshot["current_round_id"] == result["round_id"]
    assert {"fast", "slow", "arbiter", "speaking"} <= set(snapshot["lanes"])

    fast_lane = snapshot["lanes"]["fast"]
    slow_lane = snapshot["lanes"]["slow"]
    arbiter_lane = snapshot["lanes"]["arbiter"]
    speaking_lane = snapshot["lanes"]["speaking"]

    _assert_lane_shape(fast_lane, round_id=result["round_id"], cancellation_token=result["cancellation_token"])
    _assert_lane_shape(slow_lane, round_id=result["round_id"], cancellation_token=result["cancellation_token"])
    _assert_lane_shape(arbiter_lane, round_id=result["round_id"], cancellation_token=result["cancellation_token"])
    _assert_lane_shape(speaking_lane, round_id=result["round_id"], cancellation_token=result["cancellation_token"])

    assert fast_lane["status"] == "hypothesis_pending"
    assert fast_lane["pending_count"] >= 1
    assert fast_lane["stable_count"] == 0

    assert slow_lane["status"] == "stable_committed"
    assert slow_lane["pending_count"] == 0
    assert slow_lane["stable_count"] == 1

    assert arbiter_lane["status"] == "approved"
    assert arbiter_lane["pending_count"] == 0
    assert arbiter_lane["stable_count"] == 1

    assert speaking_lane["status"] == "ready"
    assert speaking_lane["pending_count"] == 0
    assert speaking_lane["stable_count"] == len(snapshot["current"]["stable_speech_segments"])

    assert partial["fast"]["stored_hypothesis"]["stable"] is False
    assert snapshot["current"]["stable_decisions"] == [result["decision"]]


def test_scheduler_snapshot_clears_old_round_lane_state_after_interrupt() -> None:
    scheduler = RealtimeCognitiveScheduler(clock=_clock())

    initial = scheduler.observe_partial("请打开客厅灯")
    scheduler.observe_final("请打开客厅灯")
    scheduler.decide()
    old_round_id = initial["round_id"]
    old_token = initial["cancellation_token"]

    interrupt = scheduler.interrupt(reason="barge_in")
    snapshot = scheduler.snapshot()

    new_round_id = interrupt["start_new_round"]["round_id"]
    new_token = interrupt["start_new_round"]["cancellation_token"]

    assert snapshot["current_round_id"] == new_round_id
    assert snapshot["history"][old_round_id]["stable_decisions"]
    assert snapshot["history"][old_round_id]["cancellation"]["cancelled"] is True

    for lane_name in ("fast", "slow", "arbiter", "speaking"):
        lane = snapshot["lanes"][lane_name]
        _assert_lane_shape(lane, round_id=new_round_id, cancellation_token=new_token)
        assert lane["round_id"] != old_round_id
        assert lane["cancellation_token"] != old_token

    assert snapshot["lanes"]["fast"]["status"] == "idle"
    assert snapshot["lanes"]["fast"]["pending_count"] == 0
    assert snapshot["lanes"]["slow"]["status"] == "idle"
    assert snapshot["lanes"]["slow"]["stable_count"] == 0
    assert snapshot["lanes"]["arbiter"]["status"] == "idle"
    assert snapshot["lanes"]["arbiter"]["stable_count"] == 0
    assert snapshot["lanes"]["speaking"]["status"] == "idle"
    assert snapshot["lanes"]["speaking"]["stable_count"] == 0
