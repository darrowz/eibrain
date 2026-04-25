"""Continuous visual tracking loop for honjia."""

from __future__ import annotations

import inspect
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
        self.session_id = "tracking-session"
        self.actor_id = "vision-runtime"
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
                self._track_once()
            except Exception as exc:  # pragma: no cover - hardware boundary
                record = getattr(self.body_runtime, "record_runtime_event", None)
                if callable(record):
                    record(
                        kind="visual_tracking_error",
                        source="eye.tracking",
                        status="error",
                        details={"error": str(exc)},
                    )
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.0, self.interval_s - elapsed))

    def _track_once(self) -> None:
        track = self.body_runtime.track_visual_target_once
        kwargs = {"session_id": self.session_id, "actor_id": self.actor_id}
        signature = inspect.signature(track)
        accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
        if accepts_kwargs or "recenter_after_misses" in signature.parameters:
            kwargs["recenter_after_misses"] = self.recenter_after_misses
        track(**kwargs)
