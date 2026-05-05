from __future__ import annotations


def test_lifecycle_manager_starts_components_and_stops_in_reverse_order() -> None:
    from eibrain.kernel.lifecycle import LifecycleManager

    calls: list[str] = []
    manager = LifecycleManager()
    manager.register("ear", start=lambda: calls.append("start-ear"), stop=lambda: calls.append("stop-ear"))
    manager.register("mouth", start=lambda: calls.append("start-mouth"), stop=lambda: calls.append("stop-mouth"))

    manager.start()
    status_after_start = manager.status()
    manager.stop()

    assert calls == ["start-ear", "start-mouth", "stop-mouth", "stop-ear"]
    assert status_after_start["state"] == "running"
    assert status_after_start["components"]["ear"]["state"] == "running"
    assert manager.status()["state"] == "stopped"


def test_lifecycle_manager_reports_failed_component_without_starting_later_components() -> None:
    from eibrain.kernel.lifecycle import LifecycleManager

    calls: list[str] = []
    manager = LifecycleManager()
    manager.register("ear", start=lambda: calls.append("start-ear"))

    def fail_start() -> None:
        calls.append("start-mouth")
        raise RuntimeError("speaker missing")

    manager.register("mouth", start=fail_start)
    manager.register("neck", start=lambda: calls.append("start-neck"))

    try:
        manager.start()
    except RuntimeError as exc:
        assert "speaker missing" in str(exc)
    else:
        raise AssertionError("LifecycleManager.start() should propagate startup failures")

    status = manager.status()
    assert calls == ["start-ear", "start-mouth"]
    assert status["state"] == "failed"
    assert status["components"]["mouth"]["state"] == "failed"
    assert status["components"]["neck"]["state"] == "registered"


def test_kernel_scheduler_runs_due_once_tasks_in_deadline_order() -> None:
    from eibrain.kernel.scheduler import KernelScheduler

    now = 100.0
    scheduler = KernelScheduler(clock=lambda: now)
    calls: list[str] = []
    scheduler.schedule_once("later", lambda: calls.append("later"), delay_s=2.0)
    scheduler.schedule_once("now", lambda: calls.append("now"), delay_s=0.0)

    assert scheduler.tick() == ["now"]
    assert calls == ["now"]

    now = 102.0
    assert scheduler.tick() == ["later"]
    assert calls == ["now", "later"]
    assert scheduler.pending() == []


def test_kernel_scheduler_reschedules_interval_tasks_until_cancelled() -> None:
    from eibrain.kernel.scheduler import KernelScheduler

    now = 10.0
    scheduler = KernelScheduler(clock=lambda: now)
    calls: list[float] = []
    scheduler.schedule_interval("heartbeat", lambda: calls.append(now), interval_s=1.5, run_immediately=True)

    assert scheduler.tick() == ["heartbeat"]
    now = 11.0
    assert scheduler.tick() == []
    now = 11.5
    assert scheduler.tick() == ["heartbeat"]
    scheduler.cancel("heartbeat")
    now = 13.0

    assert scheduler.tick() == []
    assert calls == [10.0, 11.5]
