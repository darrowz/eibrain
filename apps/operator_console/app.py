"""Operator console placeholder."""

from __future__ import annotations

import time


class OperatorConsoleApp:
    """Operator console for status summaries."""

    IMPORTANT_CAPABILITIES = (
        "can_hear_voice",
        "can_transcribe_speech",
        "can_see_people",
        "can_speak",
        "can_orient_head",
    )

    ORGAN_LABELS = {
        "ear": "Ear",
        "eye": "Eye",
        "mouth": "Mouth",
        "neck": "Neck",
    }

    def build_status_report(
        self,
        *,
        body_snapshot: dict[str, object],
        cognitive_snapshot: dict[str, object],
        traces: list[dict[str, object]],
    ) -> dict[str, object]:
        degradation = str(body_snapshot.get("degradation_mode", "unknown"))
        capabilities = dict(body_snapshot.get("capabilities", {}))
        organs = dict(body_snapshot.get("organs", {}))
        warnings = [
            f"{name}=false"
            for name in self.IMPORTANT_CAPABILITIES
            if capabilities.get(name) is False
        ]
        degraded_organs = sorted(
            organ_name
            for organ_name, snapshot in organs.items()
            if isinstance(snapshot, dict) and snapshot.get("health") != "healthy"
        )
        system_health = "healthy" if degradation == "normal" and not warnings and not degraded_organs else "degraded"
        organ_cards = self._build_organ_cards(organs)
        latency_metrics = self._build_latency_metrics(organs)
        summary = self._build_summary(
            capabilities=capabilities,
            warnings=warnings,
            degraded_organs=degraded_organs,
            latency_metrics=latency_metrics,
        )
        return {
            "system_health": system_health,
            "generated_at_ts": time.time(),
            "trace_count": len(traces),
            "warnings": warnings,
            "degraded_organs": degraded_organs,
            "summary": summary,
            "organ_cards": organ_cards,
            "latency_metrics": latency_metrics,
            "event_breakdown": self._build_event_breakdown(traces),
            "body": body_snapshot,
            "cognition": cognitive_snapshot,
            "recent_traces": traces[-5:],
        }

    def _build_organ_cards(self, organs: dict[str, object]) -> list[dict[str, object]]:
        cards: list[dict[str, object]] = []
        for organ_name, snapshot in organs.items():
            if not isinstance(snapshot, dict):
                continue
            subfunctions = snapshot.get("subfunctions", {})
            if not isinstance(subfunctions, dict):
                subfunctions = {}
            entries: list[dict[str, object]] = []
            latencies: list[float] = []
            for sub_name, sub_snapshot in subfunctions.items():
                if not isinstance(sub_snapshot, dict):
                    continue
                details = sub_snapshot.get("details", {})
                if not isinstance(details, dict):
                    details = {}
                elapsed_ms = details.get("elapsed_ms")
                if isinstance(elapsed_ms, (int, float)):
                    latencies.append(float(elapsed_ms))
                entries.append(
                    {
                        "name": sub_name,
                        "health": sub_snapshot.get("health", "unknown"),
                        "driver": details.get("driver", "unknown"),
                        "elapsed_ms": elapsed_ms,
                        "status": details.get("status", sub_snapshot.get("health", "unknown")),
                        "error": details.get("error") or details.get("reason") or details.get("stderr", ""),
                    }
                )
            cards.append(
                {
                    "name": organ_name,
                    "label": self.ORGAN_LABELS.get(organ_name, organ_name.title()),
                    "health": snapshot.get("health", "unknown"),
                    "subfunction_count": len(entries),
                    "healthy_subfunctions": sum(1 for entry in entries if entry["health"] == "healthy"),
                    "degraded_subfunctions": sum(1 for entry in entries if entry["health"] != "healthy"),
                    "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
                    "max_latency_ms": round(max(latencies), 2) if latencies else None,
                    "subfunctions": entries,
                }
            )
        return cards

    def _build_latency_metrics(self, organs: dict[str, object]) -> list[dict[str, object]]:
        metrics: list[dict[str, object]] = []
        for organ_name, snapshot in organs.items():
            if not isinstance(snapshot, dict):
                continue
            subfunctions = snapshot.get("subfunctions", {})
            if not isinstance(subfunctions, dict):
                continue
            for sub_name, sub_snapshot in subfunctions.items():
                if not isinstance(sub_snapshot, dict):
                    continue
                details = sub_snapshot.get("details", {})
                if not isinstance(details, dict):
                    continue
                elapsed_ms = details.get("elapsed_ms")
                if not isinstance(elapsed_ms, (int, float)):
                    continue
                metrics.append(
                    {
                        "id": f"{organ_name}.{sub_name}",
                        "organ": organ_name,
                        "subfunction": sub_name,
                        "driver": details.get("driver", "unknown"),
                        "health": sub_snapshot.get("health", "unknown"),
                        "elapsed_ms": round(float(elapsed_ms), 2),
                    }
                )
        metrics.sort(key=lambda item: float(item["elapsed_ms"]), reverse=True)
        return metrics

    def _build_summary(
        self,
        *,
        capabilities: dict[str, object],
        warnings: list[str],
        degraded_organs: list[str],
        latency_metrics: list[dict[str, object]],
    ) -> dict[str, object]:
        enabled_capabilities = sum(1 for value in capabilities.values() if value is True)
        avg_latency = round(
            sum(float(metric["elapsed_ms"]) for metric in latency_metrics) / len(latency_metrics),
            2,
        ) if latency_metrics else None
        p95_latency = None
        if latency_metrics:
            ordered = sorted(float(metric["elapsed_ms"]) for metric in latency_metrics)
            index = min(len(ordered) - 1, max(0, int(len(ordered) * 0.95) - 1))
            p95_latency = round(ordered[index], 2)
        return {
            "enabled_capability_count": enabled_capabilities,
            "capability_count": len(capabilities),
            "warning_count": len(warnings),
            "degraded_organ_count": len(degraded_organs),
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
        }

    @staticmethod
    def _build_event_breakdown(traces: list[dict[str, object]]) -> list[dict[str, object]]:
        counts: dict[str, int] = {}
        for trace in traces:
            kind = str(trace.get("kind", "unknown"))
            counts[kind] = counts.get(kind, 0) + 1
        return [
            {"kind": kind, "count": count}
            for kind, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ]
