"""Continuous honjia visual tracking loop."""

from __future__ import annotations

import threading

from apps.body_runtime.app import BodyRuntimeApp


class VisualTrackingLoop:
    def __init__(
        self,
        *,
        body_runtime: BodyRuntimeApp,
        interval_s: float = 1.0,
        session_id: str = "visual-tracking-loop",
        actor_id: str = "vision-runtime",
    ) -> None:
        self.body_runtime = body_runtime
        self.interval_s = max(0.2, float(interval_s))
        self.session_id = session_id
        self.actor_id = actor_id
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="visual-tracking-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.body_runtime.track_visual_target_once(
                    session_id=self.session_id,
                    actor_id=self.actor_id,
                )
            except Exception:
                self._sleep(max(1.0, self.interval_s))
                continue
            self._sleep(self.interval_s)

    def _sleep(self, seconds: float) -> None:
        self._stop_event.wait(max(0.0, seconds))
