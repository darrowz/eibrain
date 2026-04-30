from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from eibrain.learning.exploration import plan_exploration_tasks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eibrain-plan-exploration")
    parser.add_argument("--experiences")
    parser.add_argument("--registry")
    parser.add_argument("--max-tasks", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    tasks = plan_exploration_tasks(
        experiences=_read_items(args.experiences),
        registry_assets=_read_items(args.registry),
        max_tasks=args.max_tasks,
    )
    payload = {"ok": True, "tasks": tasks, "task_count": len(tasks)}
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


def _read_items(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    text = Path(path).read_text(encoding="utf-8")
    if not text.strip():
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("items", "experiences", "records", "skills", "candidates", "tasks"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [payload]
    return []


if __name__ == "__main__":
    raise SystemExit(main())
