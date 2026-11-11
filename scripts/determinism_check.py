"""Check that MLX generation on the bf16 Llama is deterministic.

Anonymizes the same level 1 profiles twice with the llama_basic prompt, one
prompt per generate call, and writes both texts per profile to
docs/results/determinism_bf16.json. Exit 0 when every pair is byte-identical.

Usage (repo root):
    uv run --no-sync python scripts/determinism_check.py [--n 5]

Loads ~16GB of weights; the mlx_engine lock refuses to start while another
process holds a model.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["RAT_BACKEND"] = "mlx"

MODEL_VERSION = "3.1-8B-Instruct-bf16"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--data", default=os.path.join(ROOT, "benchmark", "level_1.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "results", "determinism_bf16.json"))
    args = ap.parse_args()

    import mlx_lm
    from pii_benchmark import mlx_engine
    from pii_benchmark.anonymizers.llama import LlamaAnonymizer

    with open(args.data) as f:
        profiles = [json.loads(line) for line in f if line.strip()][: args.n]

    # Same settings as the llama_basic method in get_anonymizers.py.
    anon = LlamaAnonymizer(attributes=None, prompt_type="anthropic", model_version=MODEL_VERSION)

    texts = {}
    for run in (1, 2):
        for p in profiles:
            texts[(p["id"], run)] = anon.anonymize(p["text"])
            print(f"run {run} id {p['id']}: {len(texts[(p['id'], run)])} chars", flush=True)

    runs = [
        {
            "id": p["id"],
            "text_run1": texts[(p["id"], 1)],
            "text_run2": texts[(p["id"], 2)],
            "identical": texts[(p["id"], 1)].encode() == texts[(p["id"], 2)].encode(),
        }
        for p in profiles
    ]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    result = {
        "model_repo": mlx_engine.resolve_repo(MODEL_VERSION),
        "level": 1,
        "prompt": "llama_basic (prompt_type=anthropic)",
        "temperature": 0.0,
        "one_prompt_per_call": True,
        "mlx_lm_version": getattr(mlx_lm, "__version__", "unknown"),
        "git_commit": commit,
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "all_identical": all(r["identical"] for r in runs),
        "runs": runs,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"{sum(r['identical'] for r in runs)}/{len(runs)} byte-identical -> {args.out}")
    return 0 if result["all_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
