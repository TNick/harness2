"""Transport adapters for judge model requests.

Adapters receive only the supplied prompt. They do not inspect benchmark artifacts or
workspace files. The API transport remains the default; Codex CLI is opt-in.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class JudgeRequest:
    prompt: str
    system_prompt: str = ""
    response_schema: dict | None = None


@dataclass(frozen=True)
class JudgeResult:
    text: str
    parsed: dict | None
    backend: str
    model: str
    elapsed_seconds: float
    usage: dict[str, int] | None = None


@dataclass(frozen=True)
class JudgeSettings:
    backend: str
    model: str
    effort: str = "medium"
    timeout: int = 300
    api_base_url: str = ""
    api_key: str = field(default="", repr=False)
    codex_bin: str = "codex"


class JudgeError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class JudgeAdapter(Protocol):
    def check(self) -> None: ...

    def judge(self, request: JudgeRequest) -> JudgeResult: ...


def _validate_json(text: str, schema: dict) -> dict:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JudgeError("invalid_output", f"judge returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise JudgeError("invalid_output", "structured judge output must be a JSON object")
    try:
        import jsonschema
    except ImportError as exc:
        raise JudgeError("configuration", "install harness2[judge] for schema validation") from exc
    try:
        jsonschema.validate(value, schema)
    except jsonschema.ValidationError as exc:
        raise JudgeError(
            "invalid_output", f"judge output failed schema validation: {exc.message}"
        ) from exc
    return value


def _result(
    text: str, settings: JudgeSettings, started: float, *, backend: str, usage=None, schema=None
):
    if not isinstance(text, str):
        raise JudgeError("invalid_output", "judge returned non-text content")
    text = text.strip()
    if not text:
        raise JudgeError("invalid_output", "judge returned an empty response")
    parsed = _validate_json(text, schema) if schema else None
    return JudgeResult(text, parsed, backend, settings.model, time.monotonic() - started, usage)


class ApiJudge:
    def __init__(self, settings: JudgeSettings):
        self.settings = settings

    def check(self) -> None:
        s = self.settings
        if not s.model:
            raise JudgeError("configuration", "HARNESS2_JUDGE_MODEL is required")
        if not s.api_base_url:
            raise JudgeError("configuration", "HARNESS2_JUDGE_BASE_URL is required for API judging")
        if not s.api_key:
            raise JudgeError(
                "configuration", "HARNESS2_JUDGE_API_KEY or OPENAI_API_KEY is required"
            )
        if s.timeout <= 0:
            raise JudgeError("configuration", "HARNESS2_JUDGE_TIMEOUT must be positive")

    def judge(self, request: JudgeRequest) -> JudgeResult:
        self.check()
        s = self.settings
        started = time.monotonic()
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        body = {"model": s.model, "messages": messages, "reasoning_effort": s.effort}
        if request.response_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_response",
                    "strict": True,
                    "schema": request.response_schema,
                },
            }
        url = s.api_base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        req = Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {s.api_key}", "Content-Type": "application/json"},
        )
        try:
            with urlopen(req, timeout=s.timeout) as response:
                payload = json.loads(response.read())
        except HTTPError as exc:
            code = (
                "rate_limit"
                if exc.code == 429
                else ("authentication" if exc.code in (401, 403) else "provider")
            )
            raise JudgeError(code, f"judge API returned HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise JudgeError("provider", f"judge API request failed: {type(exc).__name__}") from exc
        except json.JSONDecodeError as exc:
            raise JudgeError("invalid_output", "judge API returned malformed JSON") from exc
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise JudgeError("invalid_output", "judge API response has no message content") from exc
        usage = payload.get("usage")
        usage = (
            {k: int(v) for k, v in usage.items() if isinstance(v, (int, float))}
            if isinstance(usage, dict)
            else None
        )
        return _result(text, s, started, backend="api", usage=usage, schema=request.response_schema)


class CodexCliJudge:
    def __init__(self, settings: JudgeSettings):
        self.settings = settings

    def _binary(self) -> str:
        binary = shutil.which(self.settings.codex_bin)
        if not binary:
            raise JudgeError("configuration", f"Codex CLI not found: {self.settings.codex_bin}")
        return binary

    def check(self) -> None:
        s = self.settings
        binary = self._binary()
        if not s.model:
            raise JudgeError("configuration", "HARNESS2_JUDGE_MODEL is required")
        if s.timeout <= 0:
            raise JudgeError("configuration", "HARNESS2_JUDGE_TIMEOUT must be positive")
        try:
            r = subprocess.run(
                [binary, "exec", "--help"], capture_output=True, text=True, timeout=15
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise JudgeError(
                "configuration", f"could not check Codex CLI: {type(exc).__name__}"
            ) from exc
        required_flags = ("--output-last-message", "--json", "--disable")
        if r.returncode or any(flag not in r.stdout for flag in required_flags):
            raise JudgeError(
                "configuration",
                "Codex CLI lacks required exec flags (--json, --disable, --output-last-message)",
            )
        # Login check is local and must never print tokens.
        try:
            status = subprocess.run(
                [binary, "login", "status"], capture_output=True, text=True, timeout=15
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise JudgeError(
                "authentication", f"could not check Codex login: {type(exc).__name__}"
            ) from exc
        if status.returncode:
            raise JudgeError("authentication", "Codex CLI is not logged in; run `codex login`")

    def judge(self, request: JudgeRequest) -> JudgeResult:
        self.check()
        s = self.settings
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="harness2-judge-") as tmp:
            root = Path(tmp)
            instructions = root / "instructions.md"
            instructions.write_text(
                "You are a judge. Use only the supplied prompt. Do not inspect files, run tools, "
                "or investigate the environment. Return only the requested answer.\n\n"
                + request.system_prompt,
                encoding="utf-8",
            )
            out = root / "answer.txt"
            schema_path = root / "schema.json"
            cmd = [
                self._binary(),
                "exec",
                "--ephemeral",
                "--json",
                "--skip-git-repo-check",
                "--ignore-user-config",
                "--ignore-rules",
                "--sandbox",
                "read-only",
                "--disable",
                "shell_tool",
                "--disable",
                "apps",
                "--disable",
                "hooks",
                "--disable",
                "multi_agent",
                "--disable",
                "view_image",
                "--disable",
                "image_generation",
                "--model",
                s.model,
                "-c",
                f'model_reasoning_effort="{s.effort}"',
                "-c",
                f'model_instructions_file="{instructions}"',
                "-c",
                "project_doc_max_bytes=0",
                "-c",
                'approval_policy="never"',
                "-c",
                'web_search="disabled"',
                "-C",
                str(root),
                "--output-last-message",
                str(out),
                "-",
            ]
            if request.response_schema:
                schema_path.write_text(json.dumps(request.response_schema), encoding="utf-8")
                cmd.extend(["--output-schema", str(schema_path)])
            # Preserve Codex's ChatGPT login while removing API endpoint overrides and the solver's
            # custom CODEX_HOME. User-global config is ignored; only the authenticated default home remains.
            env = {
                k: v
                for k, v in os.environ.items()
                if k
                not in {
                    "OPENAI_BASE_URL",
                    "OPENAI_API_BASE",
                    "CODEX_HOME",
                    "HARNESS2_CODEX_BASE_URL",
                    "HARNESS2_CODEX_API_KEY",
                    "OPENAI_API_KEY",
                }
            }
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=root,
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                stdout, stderr = proc.communicate(request.prompt, timeout=s.timeout)
            except subprocess.TimeoutExpired as exc:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.communicate()
                raise JudgeError("timeout", f"Codex judge timed out after {s.timeout}s") from exc
            except OSError as exc:
                raise JudgeError(
                    "provider", f"Codex CLI failed to start: {type(exc).__name__}"
                ) from exc
            if proc.returncode:
                tail = (stderr or stdout)[-500:].strip()
                if any(
                    token in tail.lower()
                    for token in ("not logged in", "unauthorized", "authentication")
                ):
                    code = "authentication"
                elif "429" in tail or "rate limit" in tail.lower():
                    code = "rate_limit"
                else:
                    code = "provider"
                raise JudgeError(code, f"Codex judge exited {proc.returncode}: {tail}")
            events = []
            for line in stdout.splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    raise JudgeError("invalid_output", "Codex CLI emitted malformed JSONL")
            for event in events:
                item = event.get("item") if isinstance(event.get("item"), dict) else {}
                if event.get("type") == "error" or item.get("type") == "error":
                    detail = event.get("error") or item.get("error") or item.get("message")
                    if isinstance(detail, dict):
                        detail = detail.get("message") or detail.get("code")
                    if not isinstance(detail, str) or not detail.strip():
                        detail = "Codex reported an unspecified error"
                    detail = detail.strip().replace("\n", " ")[:240]
                    raise JudgeError("provider", f"Codex judge error: {detail}")
            if any(e.get("type") == "turn.failed" for e in events):
                raise JudgeError("provider", "Codex judge turn failed")
            if not any(e.get("type") == "turn.completed" for e in events):
                raise JudgeError("invalid_output", "Codex CLI did not report a completed turn")
            tool_types = sorted(
                {
                    e["item"].get("type", "unknown")
                    for e in events
                    if e.get("type", "").startswith("item.")
                    and isinstance(e.get("item"), dict)
                    and e["item"].get("type") not in {"agent_message", "reasoning"}
                }
            )
            if tool_types:
                raise JudgeError(
                    "provider",
                    "Codex judge emitted non-answer item(s): " + ", ".join(tool_types),
                )
            try:
                text = out.read_text(encoding="utf-8")
            except OSError as exc:
                raise JudgeError(
                    "invalid_output", "Codex CLI did not write its final answer"
                ) from exc
            return _result(text, s, started, backend="codex_cli", schema=request.response_schema)


def create_judge(settings: JudgeSettings) -> JudgeAdapter:
    if settings.backend == "api":
        return ApiJudge(settings)
    if settings.backend == "codex_cli":
        return CodexCliJudge(settings)
    raise JudgeError(
        "configuration", f"unsupported judge backend {settings.backend!r}; use api or codex_cli"
    )
