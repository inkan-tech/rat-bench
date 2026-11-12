"""Generate attacker guesses with a recent MLX model, outside the benchmark venv.

Runs in its own uv project (scripts/mlx_attack/pyproject.toml): mlx-lm 0.32.x
needs transformers 5, which the benchmark venv cannot take. Uses the repo's
own prompt (get_staab_prompt), system message and parser (parse_output_gpt),
like MlxAttacker: greedy, one prompt at a time, max 4096 tokens. Score the
output afterwards in the benchmark venv with score_guesses.py.

Usage (from the repo root):
  uv run --project scripts/mlx_attack python scripts/mlx_attack/gen_guesses.py \
      <repo> <suffix> <in.jsonl> <out.jsonl> <methods> [limit]

The output file is rewritten after each guess and a re-run resumes where it
stopped. Each output row gets run_info_<suffix> (model repo and revision,
mlx-lm version, claude CLI version, git commit).
"""
import fcntl
import json
import os
import re
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from mlx_lm import generate, load  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

from pii_benchmark.prompts import get_staab_prompt  # noqa: E402
from pii_benchmark.run_info import collect_run_info  # noqa: E402
from pii_benchmark.utils import parse_output_gpt  # noqa: E402

SYSTEM = "You are an AI Assistant that specializes in generating synthetic data. Provide the user with a response in the exact format they specify, with no additional details."
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
# Reasoning is off by default, to match the Claude attacker of record (thinking
# off). MLX_ATTACK_THINKING=1 turns it on for models whose chat template has
# enable_thinking (Qwen 3.x); give it a larger MLX_ATTACK_MAX_TOKENS, since the
# reasoning and the answer share the cap.
THINKING = os.environ.get("MLX_ATTACK_THINKING") == "1"
MAX_TOKENS = int(os.environ.get("MLX_ATTACK_MAX_TOKENS", "4096"))
# Same lock file as pii_benchmark.mlx_engine.LOCK_PATH: one model process at a time.
LOCK_PATH = os.path.join(tempfile.gettempdir(), "rat-bench-mlx.lock")


def strip_reasoning(out, thinking):
    """Return (answer, cut). With thinking on, the chat template already ends
    the prompt with <think>, so the output holds only the closing tag: the
    answer is what follows the last </think>. No closing tag means the token
    cap cut the reasoning before the answer started."""
    if thinking:
        if "</think>" not in out:
            return out, True
        return out.rsplit("</think>", 1)[1], False
    return THINK_RE.sub("", out), False


def main():
    if len(sys.argv) < 6:
        sys.exit(__doc__)
    repo, suffix, in_path, out_path, methods = sys.argv[1:6]
    methods = [m.strip() for m in methods.split(",")]
    limit = int(sys.argv[6]) if len(sys.argv) > 6 else None

    lock = open(LOCK_PATH, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"Another process already holds an MLX model ({LOCK_PATH}). Only one model at a time.")

    src = out_path if os.path.exists(out_path) else in_path
    profiles = [json.loads(l) for l in open(src) if l.strip()]
    if limit:
        profiles = profiles[:limit]

    run_info = collect_run_info(model_repo=repo)
    run_info["thinking"] = THINKING
    run_info["max_tokens"] = MAX_TOKENS
    model, tokenizer = load(repo)
    sampler = make_sampler(temp=0.0)
    print(f"loaded {repo} {json.dumps(run_info)}", flush=True)

    for m in methods:
        key = f"guesses_{m}_{suffix}"
        for p in profiles:
            if key in p:
                continue
            prompt = get_staab_prompt(attributes=p["features"], text=p[f"text_anon_{m}"], scenario=p["scenario"])
            chat = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False, enable_thinking=THINKING)
            t0 = time.time()
            out = generate(model, tokenizer, prompt=text, max_tokens=MAX_TOKENS, sampler=sampler)
            dt = time.time() - t0
            n_tok = len(tokenizer.encode(out))
            out, cut = strip_reasoning(out, THINKING)
            p[key] = {} if cut else parse_output_gpt(out)
            p[f"cut_{m}_{suffix}"] = cut
            p[f"tokens_{m}_{suffix}"] = n_tok
            p[f"prompts_{m}_{suffix}"] = prompt
            p[f"raw_{m}_{suffix}"] = out
            p[f"seconds_{m}_{suffix}"] = round(dt, 1)
            p[f"run_info_{suffix}"] = run_info
            with open(out_path, "w") as f:
                for q in profiles:
                    f.write(json.dumps(q) + "\n")
            print(f"{m} id={p['id']} {dt:.1f}s {n_tok} tokens parsed={len(p[key])} attrs{' CUT' if cut else ''}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
