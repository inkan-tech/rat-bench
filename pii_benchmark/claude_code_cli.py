import json
import os
import shutil
import subprocess
import tempfile

# Claude Code CLI binary; override with CLAUDE_CODE_BIN if it is not on PATH.
CLAUDE_BIN = os.environ.get("CLAUDE_CODE_BIN", "claude")

# With an API key, run in --bare mode: no account context reaches the model.
# Without one (logged-in account, OAuth), Claude Code adds the account email and
# environment details to the context, and --bare cannot be used.
BARE = bool(os.environ.get("ANTHROPIC_API_KEY"))
_warned = False


class ClaudeCodeError(RuntimeError):
    pass


def run_claude_code(prompt: str, system_prompt: str, model: str = "haiku", timeout: int = 600) -> str:
    """Run one headless `claude -p` call and return the text of the final answer.

    The call is isolated so the benchmark measures the model and not the local setup:
    no tools, no MCP servers, no user/project settings (CLAUDE.md, hooks, plugins),
    no saved session. Auth comes from ANTHROPIC_API_KEY (--bare mode) if set,
    else from the logged-in Claude Code account (see BARE).

    Raises ClaudeCodeError on any failure, so an error string is never saved
    as an anonymized text or an attacker answer.
    """
    global _warned
    if shutil.which(CLAUDE_BIN) is None:
        raise ClaudeCodeError(f"Claude Code CLI not found: {CLAUDE_BIN!r}")
    if not BARE and not _warned:
        print("WARNING: ANTHROPIC_API_KEY not set. Claude Code runs on the logged-in account and adds "
              "the account email and environment details to the model context. Set ANTHROPIC_API_KEY "
              "for clean (--bare) runs.")
        _warned = True

    cmd = [
        CLAUDE_BIN, "-p",
        *(["--bare"] if BARE else []),
        "--model", model,
        "--system-prompt", system_prompt,
        "--tools", "",
        "--setting-sources", "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--output-format", "json",
    ]
    # Run from an empty directory so no project CLAUDE.md is picked up.
    with tempfile.TemporaryDirectory() as cwd:
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, cwd=cwd)

    if proc.returncode != 0:
        raise ClaudeCodeError(f"claude exited with {proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()[:500]}")

    try:
        events = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise ClaudeCodeError(f"Unparsable claude output: {proc.stdout[:500]}") from e
    if isinstance(events, dict):
        events = [events]

    result = next((e for e in reversed(events) if e.get("type") == "result"), None)
    if result is None:
        raise ClaudeCodeError("No result event in claude output")
    if result.get("is_error") or result.get("subtype") != "success":
        raise ClaudeCodeError(f"claude returned an error: {result.get('subtype')}: {result.get('result')}")

    return result.get("result") or ""
