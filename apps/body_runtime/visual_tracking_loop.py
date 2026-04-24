"""Continuous visual tracking loop for honjia."""

from __future__ import annotations

import threading
import time

from apps.body_runtime.app import BodyRuntimeApp


class VisualTrackingLoop:
    def __init__(
        self,
        *,
        body_runtime: BodyRuntimeApp,
        interval_s: float = 0.5,
        recenter_after_misses: int = 3,
    ) -> None:
        self.body_runtime = body_runtime
        self.interval_s = max(0.2, float(interval_s))
        self.recenter_after_misses = max(1, int(recenter_after_misses))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="eibrain-visual-tracking", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self.body_runtime.track_visual_target_once(recenter_after_misses=self.recenter_after_misses)
            except Exception as exc:  # pragma: no cover - hardware boundary
                self.body_runtime.record_runtime_event(
                    kind="visual_tracking_error",
                    source="eye.tracking",
                    status="error",
                    details={"error": str(exc)},
                )
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.0, self.interval_s - elapsed))
