"""Score attacker guesses made by gen_guesses.py (no LLM call).

Runs in the BENCHMARK venv (repo root), not in scripts/mlx_attack: scoring
needs CorrectMatch (Julia) and the benchmark dependencies. It wraps
pii_benchmark.attack.only_check_correctness, which recomputes correctness
and the re-identification rate from the guesses already in the file.

Usage (from the repo root):
  uv run --no-sync python scripts/mlx_attack/score_guesses.py \
      --in benchmark/attacker_qwen35/level_1.jsonl --attacker qwen35 \
      --out benchmark/attacker_qwen35_rescore --level 1

Writes <out>/level_<level>.jsonl and
<out>/results/<scenario>/level_<level>_attacker_<attacker>.pickle, and prints
the re-identification rate of each method.
"""
import argparse
import json
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from pii_benchmark.attack import only_check_correctness  # noqa: E402
from pii_benchmark.run_info import collect_run_info  # noqa: E402

DEFAULT_METHODS = "presidio,gliner,scrubadub,llama_basic"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True, help="jsonl with guesses_<method>_<attacker> keys")
    ap.add_argument("--attacker", required=True, help="attacker suffix used by gen_guesses.py")
    ap.add_argument("--out", required=True, help="new results folder (the input file is not modified)")
    ap.add_argument("--methods", default=DEFAULT_METHODS)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--scenario", default="medical")
    ap.add_argument("--dataset", default="PUMS")
    ap.add_argument("--language", default=None)
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    profiles = [json.loads(l) for l in open(args.in_path) if l.strip()]
    os.makedirs(args.out, exist_ok=True)
    uniq = os.path.join(args.out, "results")
    os.makedirs(os.path.join(uniq, args.scenario), exist_ok=True)

    _, pickle_path = only_check_correctness(
        profiles=profiles, anon_methods=methods, attacker_name=args.attacker, scenario=args.scenario,
        results_path=args.out, uniqueness_results_path=uniq, level=args.level,
        language=args.language, dataset=args.dataset)

    with open(pickle_path, "rb") as f:
        rates = {m: float(v) for m, v in pickle.load(f)["reidentification_rate"].items()}
    info = {"input": args.in_path, "attacker": args.attacker, "results_pickle": pickle_path,
            "rates": rates, "run_info": collect_run_info()}
    with open(os.path.join(args.out, f"score_level_{args.level}_{args.attacker}.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps({"results_pickle": pickle_path, "rates": rates}, indent=2))


if __name__ == "__main__":
    main()
