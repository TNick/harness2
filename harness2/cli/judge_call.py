"""Call the configured judge transport with a JSON request on stdin.

Input is {"prompt": "...", "system_prompt": "...", "response_schema": {...}}.
Output is a single JSON JudgeResult. No benchmark artifacts are read.
"""

from __future__ import annotations

import argparse
import json
import sys

from .. import config
from ..judging import JudgeError, JudgeSettings, create_judge


def settings(args) -> JudgeSettings:
    return JudgeSettings(
        backend=args.backend or config.JUDGE_BACKEND,
        model=args.model or config.JUDGE_MODEL,
        effort=args.effort or config.JUDGE_REASONING_EFFORT,
        timeout=args.timeout if args.timeout is not None else config.JUDGE_TIMEOUT,
        api_base_url=args.base_url or config.JB_JUDGE_API_BASE,
        api_key=config.JB_JUDGE_API_KEY,
        codex_bin=args.codex_bin or config.JUDGE_CODEX_BIN,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="harness2-judge-call", description=__doc__)
    p.add_argument("--backend", choices=("api", "codex_cli"))
    p.add_argument("--model")
    p.add_argument("--effort")
    p.add_argument("--timeout", type=int)
    p.add_argument("--base-url")
    p.add_argument("--codex-bin")
    p.add_argument(
        "--check", action="store_true", help="validate local configuration without a model call"
    )
    args = p.parse_args(argv)
    try:
        adapter = create_judge(settings(args))
        if args.check:
            adapter.check()
            return 0
        raw = json.load(sys.stdin)
        if not isinstance(raw, dict) or not isinstance(raw.get("prompt"), str):
            raise JudgeError("configuration", "stdin JSON must contain a string 'prompt'")
        from ..judging import JudgeRequest

        result = adapter.judge(
            JudgeRequest(
                prompt=raw["prompt"],
                system_prompt=raw.get("system_prompt", ""),
                response_schema=raw.get("response_schema"),
            )
        )
        print(json.dumps(result.__dict__))
        return 0
    except (JudgeError, json.JSONDecodeError, OSError) as exc:
        print(f"harness2-judge-call: {exc}", file=sys.stderr)
        return 2 if isinstance(exc, JudgeError) and exc.code == "configuration" else 1


if __name__ == "__main__":
    raise SystemExit(main())
