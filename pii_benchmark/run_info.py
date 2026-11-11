"""Record what produced a run: model repo and revision, mlx-lm version,
claude CLI version and git commit. Every field is best effort (None when it
cannot be read), so recording never stops a run."""
import datetime
import importlib.metadata
import os
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _hub_cache() -> str:
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(var):
            return os.environ[var]
    hf_home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return os.path.join(hf_home, "hub")


def model_revision(model_repo, ref="main"):
    """Commit SHA of `model_repo` in the local Hugging Face cache (refs/<ref>)."""
    if not model_repo:
        return None
    path = os.path.join(_hub_cache(), "models--" + model_repo.replace("/", "--"), "refs", ref)
    try:
        with open(path) as f:
            return f.read().strip() or None
    except OSError:
        return None


def _run(cmd, cwd=None):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


def _pkg_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def mlx_repo_for(model_version):
    """MLX checkpoint behind a llama/mlx model version (mlx_engine.resolve_repo),
    or the version itself when it is not an MLX one (e.g. a Claude model)."""
    if not model_version:
        return None
    try:
        from pii_benchmark.mlx_engine import resolve_repo
        return resolve_repo(model_version)
    except Exception:
        return model_version


def collect_run_info(model_repo=None):
    return {
        "model_repo": model_repo,
        "model_revision": model_revision(model_repo),
        "mlx_lm_version": _pkg_version("mlx-lm"),
        "claude_cli_version": _run(["claude", "--version"]),
        "git_commit": _run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT),
        "python_version": sys.version.split()[0],
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
