"""CLI entrypoint for cognitive runtime."""

from __future__ import annotations

import argparse
import json

from eibrain.protocol.observations import AudioTranscriptFinal

from .app import CognitiveRuntimeApp


def main() -> None:
    parser = argparse.ArgumentParser(description="Start eibrain cognitive runtime")
    parser.add_argument("--config", default="config/eibrain.yaml")
    parser.add_argument("--text", required=True)
    parser.add_argument("--session-id", default="session-cli")
    parser.add_argument("--actor-id", default="user-cli")
    args = parser.parse_args()

    runtime = CognitiveRuntimeApp.from_config_path(args.config)
    actions = runtime.handle_observation(
        AudioTranscriptFinal(
            ts=1.0,
            source="ear.asr",
            text=args.text,
            session_id=args.session_id,
            actor_id=args.actor_id,
        )
    )
    print(json.dumps([action.to_dict() for action in actions], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
