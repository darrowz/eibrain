"""Small deterministic scheduler used by runtime loops and tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time


ScheduledCallback = Callable[[], object]


@dataclass(slots=True)
class ScheduledTask:
    name: str
    due_at_ts: float
    callback: ScheduledCallback
    interval_s: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "due_at_ts": self.due_at_ts,
            "interval_s": self.interval_s,
        }


class KernelScheduler:
    """Run due callbacks without owning a thread or event loop."""

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.time
        self._tasks: dict[str, ScheduledTask] = {}

    def schedule_once(self, name: str, callback: ScheduledCallback, *, delay_s: float = 0.0) -> None:
        self._upsert_task(
            ScheduledTask(
                name=name,
                callback=callback,
                due_at_ts=self._clock() + max(0.0, delay_s),
            )
        )

    def schedule_interval(
        self,
        name: str,
        callback: ScheduledCallback,
        *,
        interval_s: float,
        run_immediately: bool = False,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        now = self._clock()
        self._upsert_task(
            ScheduledTask(
                name=name,
                callback=callback,
                due_at_ts=now if run_immediately else now + interval_s,
                interval_s=interval_s,
            )
        )

    def cancel(self, name: str) -> bool:
        return self._tasks.pop(name, None) is not None

    def tick(self, *, now_ts: float | None = None) -> list[str]:
        now = self._clock() if now_ts is None else now_ts
        due_tasks = sorted(
            (task for task in self._tasks.values() if task.due_at_ts <= now),
            key=lambda task: (task.due_at_ts, task.name),
        )
        ran: list[str] = []
        for task in due_tasks:
            current = self._tasks.get(task.name)
            if current is not task:
                continue
            task.callback()
            ran.append(task.name)
            if task.interval_s is None:
                self._tasks.pop(task.name, None)
            elif task.name in self._tasks:
                task.due_at_ts = now + task.interval_s
        return ran

    def pending(self) -> list[dict[str, object]]:
        return [
            task.to_dict()
            for task in sorted(self._tasks.values(), key=lambda item: (item.due_at_ts, item.name))
        ]

    def _upsert_task(self, task: ScheduledTask) -> None:
        self._tasks[task.name] = task
