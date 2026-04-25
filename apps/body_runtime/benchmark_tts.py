"""Small TTS backend benchmark helper for honjia experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Callable, Iterable

from eibrain.body.runtime_linux import speak_text


DEFAULT_TEXTS = [
    "\u6211\u5728\u3002",
    "\u597d\u7684\uff0c\u5148\u4f11\u606f\u3002",
    "\u4f60\u597d\uff0c\u6211\u662f eibrain\uff0c\u6b63\u5728\u8fdb\u884c\u8bed\u97f3\u94fe\u8def\u6027\u80fd\u6d4b\u8bd5\u3002",
]


def benchmark_backend(
    *,
    backend: str,
    texts: Iterable[str],
    output_device: str,
    speak_fn: Callable[..., dict[str, object]] = speak_text,
    perf_counter: Callable[[], float] = time.perf_counter,
    **speak_kwargs: object,
) -> dict[str, object]:
    items: list[dict[str, object]] = []
    for index, text in enumerate(texts, start=1):
        started = perf_counter()
        result = speak_fn(
            text=text,
            output_device=output_device,
            backend=backend,
            **speak_kwargs,
        )
        elapsed_ms = round((perf_counter() - started) * 1000, 2)
        items.append(
            {
                "index": index,
                "text": text,
                "status": result.get("status", "unknown"),
                "elapsed_ms": elapsed_ms,
                "details": dict(result.get("details", {}) or {}),
            }
        )
    elapsed_values = [float(item["elapsed_ms"]) for item in items]
    ok_count = sum(1 for item in items if item.get("status") == "ok")
    return {
        "backend": backend,
        "output_device": output_device,
        "summary": {
            "count": len(items),
            "ok_count": ok_count,
            "error_count": len(items) - ok_count,
            "avg_elapsed_ms": round(sum(elapsed_values) / len(elapsed_values), 2) if elapsed_values else None,
            "min_elapsed_ms": round(min(elapsed_values), 2) if elapsed_values else None,
            "max_elapsed_ms": round(max(elapsed_values), 2) if elapsed_values else None,
        },
        "items": items,
    }


def _load_texts(args: argparse.Namespace) -> list[str]:
    texts = list(args.text or [])
    if args.text_file:
        texts.extend(
            line.strip()
            for line in Path(args.text_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return texts or list(DEFAULT_TEXTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark eibrain TTS backends with repeated short utterances.")
    parser.add_argument("--backend", default="moss_onnx", choices=("minimax", "moss_onnx", "espeak"))
    parser.add_argument("--output-device", default="plughw:2,0")
    parser.add_argument("--text", action="append", help="Text to synthesize. Repeat for multiple samples.")
    parser.add_argument("--text-file", default="", help="UTF-8 file containing one utterance per line.")
    parser.add_argument("--json-output", default="", help="Optional path to write the benchmark JSON report.")
    parser.add_argument("--api-key-env", default="MINIMAX_API_KEY")
    parser.add_argument("--api-base-url", default="https://api.minimaxi.com")
    parser.add_argument("--model", default="")
    parser.add_argument("--voice-id", default="Junhao")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--moss-command", default="moss-tts-nano")
    parser.add_argument("--moss-prompt-audio-path", default="")
    parser.add_argument("--moss-cpu-threads", type=int, default=4)
    parser.add_argument("--moss-max-new-frames", type=int, default=375)
    parser.add_argument("--moss-voice-clone-max-text-tokens", type=int, default=75)
    parser.add_argument("--moss-sample-mode", default="fixed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = benchmark_backend(
        backend=args.backend,
        texts=_load_texts(args),
        output_device=args.output_device,
        api_key=os.environ.get(args.api_key_env, ""),
        api_base_url=args.api_base_url,
        model=args.model,
        voice_id=args.voice_id,
        cache_dir=args.cache_dir or None,
        timeout_s=args.timeout_s,
        moss_command=args.moss_command,
        moss_prompt_audio_path=args.moss_prompt_audio_path,
        moss_cpu_threads=args.moss_cpu_threads,
        moss_max_new_frames=args.moss_max_new_frames,
        moss_voice_clone_max_text_tokens=args.moss_voice_clone_max_text_tokens,
        moss_sample_mode=args.moss_sample_mode,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json_output:
        Path(args.json_output).write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
