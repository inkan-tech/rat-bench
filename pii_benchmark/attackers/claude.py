import datetime as dt
import json
import os
import re
import subprocess
import tempfile
import threading
import time
from typing import List

from pii_benchmark.prompts import get_staab_prompt
from pii_benchmark.utils import parse_output_gpt, retry_with_backoff

SYSTEM_PROMPT = "You are an AI Assistant that specializes in generating synthetic data. Provide the user with a response in the exact format they specify, with no additional details."

# attack.py runs 10 threads; each call is a separate claude process, so cap
# them to keep the machine usable and stay under the subscription rate limits.
_CONCURRENCY = threading.Semaphore(int(os.environ.get("CLAUDE_CLI_CONCURRENCY", "8")))

# One line per call, for counting refusals and truncated answers afterwards.
LOG_PATH = os.environ.get("CLAUDE_CLI_LOG", os.path.join("benchmark", "claude_cli_calls.jsonl"))
_LOG_LOCK = threading.Lock()

# Subscription limit messages, e.g. "You've hit your session limit · resets 3:45pm"
# or "... · resets Mon 12:00am" (code.claude.com/docs/en/errors.md).
_LIMIT_RE = re.compile(r"hit your .*limit")
_RESET_RE = re.compile(r"resets\s+(?:(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+)?(\d{1,2}):(\d{2})\s*([ap]m)", re.I)
_DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
_LIMIT_FALLBACK_WAIT_S = 15 * 60

_ENV = {
    **os.environ,
    # Match the API call this replaces: no thinking, max_tokens 4096.
    "MAX_THINKING_TOKENS": "0",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "4096",
    # Keep the CLI version fixed during a run and skip telemetry/update checks;
    # neither touches the model request.
    "DISABLE_AUTOUPDATER": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
}


def seconds_until_reset(message: str, now: dt.datetime | None = None) -> float:
    """Seconds until the reset time named in a usage-limit message (local time)."""
    now = now or dt.datetime.now()
    m = _RESET_RE.search(message)
    if not m:
        return _LIMIT_FALLBACK_WAIT_S
    day, hour, minute, ampm = m.groups()
    hour = int(hour) % 12 + (12 if ampm.lower() == "pm" else 0)
    reset = now.replace(hour=hour, minute=int(minute), second=0, microsecond=0)
    if day:
        reset += dt.timedelta(days=(_DAYS.index(day.lower()) - now.weekday()) % 7)
    if reset <= now:
        reset += dt.timedelta(days=7 if day else 1)
    return (reset - now).total_seconds()


class ClaudeAttacker:
    """Claude attacker through the claude CLI (subscription), not the paid API.

    Each call runs `claude -p` with the attack system prompt, no tools, no
    settings or MCP servers, from an empty directory so no CLAUDE.md is read.
    Claude Code still attaches the account context (the user's email).

    Thinking is off and output is capped at 4096 tokens, to match the API
    call this replaces (claude-opus-4-8, no thinking parameter, max_tokens
    4096); effort stays at the model's default, high. Without this the CLI
    thinks on attack prompts (checked with stream-json).
    """

    def __init__(self, model_version: str = "claude-opus-4-8"):
        self.model_version = model_version
        self.workdir = tempfile.mkdtemp(prefix="rat-bench-claude-")
        self.cli_version = subprocess.run(
            ["claude", "--version"], capture_output=True, text=True, env=_ENV
        ).stdout.strip()
        print(f"[ClaudeAttacker] claude CLI {self.cli_version}, log: {LOG_PATH}")

    def _run(self, prompt: str) -> tuple[subprocess.CompletedProcess, dict | None]:
        cmd = [
            "claude", "-p",
            "--model", self.model_version,
            "--effort", "high",
            "--settings", '{"alwaysThinkingEnabled": false}',
            "--system-prompt", SYSTEM_PROMPT,
            "--tools", "",
            "--setting-sources", "",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--output-format", "json",
        ]
        with _CONCURRENCY:
            proc = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True,
                cwd=self.workdir, timeout=900, env=_ENV,
            )
        try:
            result = json.loads(proc.stdout)
        except json.JSONDecodeError:
            result = None
        return proc, result

    def _log(self, result: dict) -> None:
        usage = result.get("usage") or {}
        line = {
            "time": dt.datetime.now().isoformat(timespec="seconds"),
            "model": self.model_version,
            "cli_version": self.cli_version,
            "is_error": result.get("is_error"),
            "stop_reason": result.get("stop_reason"),
            "input_tokens": usage.get("input_tokens"),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "duration_ms": result.get("duration_ms"),
            "duration_api_ms": result.get("duration_api_ms"),
        }
        with _LOG_LOCK:
            os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
            with open(LOG_PATH, "a") as f:
                f.write(json.dumps(line) + "\n")

    def _call(self, prompt: str) -> dict:
        while True:
            proc, result = self._run(prompt)
            message = f"{(result or {}).get('result') or ''} {proc.stdout[-500:]} {proc.stderr[-500:]}"
            if _LIMIT_RE.search(message):
                # A subscription limit is not a failure of this prompt: wait for
                # the reset and send the same call again.
                wait = seconds_until_reset(message)
                print(f"[ClaudeAttacker] usage limit, waiting {wait / 60:.0f} min: {message.strip()[:200]}")
                time.sleep(wait + 60)
                continue
            break
        if result is None:
            raise RuntimeError(f"claude CLI exit {proc.returncode}: {(proc.stderr or proc.stdout)[-500:]}")
        self._log(result)
        if proc.returncode != 0 or result.get("is_error"):
            raise RuntimeError(f"claude CLI error: {result.get('subtype')}: {str(result.get('result'))[:500]}")
        if result.get("stop_reason") == "max_tokens":
            print(f"[ClaudeAttacker] answer cut at the 4096-token limit on {self.model_version}")
        return result

    def infer(
        self, text: str, attributes: List[str] | None = None, scenario: str = "medical", language: str = "English",
        interactive: bool = False
    ):
        prompt = get_staab_prompt(attributes=attributes, text=text, scenario=scenario, language=language)

        result = retry_with_backoff(self._call, prompt)
        if result.get("stop_reason") == "refusal":
            # A refusal yields no guesses, which score as wrong and lower the
            # re-identification rate; log it so it can be counted.
            print(f"[ClaudeAttacker] refusal on {self.model_version}")
        model_guesses = parse_output_gpt(result.get("result") or "", interactive=interactive)
        return model_guesses, prompt
