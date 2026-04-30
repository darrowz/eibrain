from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from eibrain.learning.execution import execute_exploration_plan, write_outcomes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eibrain-execute-exploration")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--replay-results")
    parser.add_argument("--registry")
    parser.add_argument("--max-tasks", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    plan = _read_json(args.plan)
    outcomes = execute_exploration_plan(
        plan,
        replay_results_path=args.replay_results,
        registry_path=args.registry,
        max_tasks=args.max_tasks,
    )
    write_outcomes(args.output, outcomes)
    payload = {"ok": True, "outcome_count": len(outcomes), "outcomes": outcomes, "output": args.output}
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


def _read_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
