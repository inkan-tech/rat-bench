"""Compare the vLLM and bf16 MLX llama_basic runs on level 1 (issue #7).

Reads the two attack pickles and the two text sets, writes
docs/results/mlx_vs_vllm.json. The MLX anonymizer is "valid" when the
re-identification rates differ by less than the half-width of the bf16 MLX
95% bootstrap interval.

    uv run --no-sync python scripts/compare_mlx_vllm.py [--next-step "..."]
"""
import argparse
import json
import pickle
from pathlib import Path

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

ROOT = Path(__file__).resolve().parents[1]
PICKLE = "results/medical/level_1_attacker_claude.pickle"
METHOD = "llama_basic"


def load_rows(path):
    with open(path) as f:
        return {r["id"]: r for r in map(json.loads, f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--next-step", default="")
    ap.add_argument("--out", default=str(ROOT / "docs/results/mlx_vs_vllm.json"))
    args = ap.parse_args()

    vp = pickle.load(open(ROOT / "benchmark/vllm" / PICKLE, "rb"))
    mp = pickle.load(open(ROOT / "benchmark/bf16" / PICKLE, "rb"))
    v = float(vp["reidentification_rate"][METHOD])
    m = float(mp["reidentification_rate"][METHOD])
    vb = vp["reidentification_rate_bootstrap"][METHOD]
    mb = mp["reidentification_rate_bootstrap"][METHOD]
    hw = (float(mb["ci_upper"]) - float(mb["ci_lower"])) / 2

    vt = load_rows(ROOT / "benchmark/vllm/level_1.jsonl")
    mt = load_rows(ROOT / "benchmark/bf16/level_1.jsonl")
    ids = sorted(set(vt) & set(mt), key=str)
    sm = SmoothingFunction().method1
    key = f"text_anon_{METHOD}"
    bleus = [sentence_bleu([vt[i][key].split()], mt[i][key].split(), smoothing_function=sm) for i in ids]
    identical = sum(vt[i][key] == mt[i][key] for i in ids)

    diff = abs(v - m)
    valid = diff < hw
    out = {
        "level": 1,
        "method": METHOD,
        "attacker": "claude",
        "rate_vllm": v,
        "ci_vllm": [float(vb["ci_lower"]), float(vb["ci_upper"])],
        "rate_mlx_bf16": m,
        "ci_mlx_bf16": [float(mb["ci_lower"]), float(mb["ci_upper"])],
        "abs_diff": diff,
        "ci_half_width": hw,
        "paper_rate": 0.37,
        "n_profiles": len(ids),
        "mean_bleu": sum(bleus) / len(bleus),
        "min_bleu": min(bleus),
        "identical_texts": identical,
        "bleu_definition": "mean over ids of nltk sentence_bleu([vllm.split()], mlx.split(), SmoothingFunction().method1)",
        "decision": "valid" if valid else "not valid",
        "next_step": "" if valid else args.next_step,
    }
    if not valid and len(args.next_step.strip()) < 20:
        raise SystemExit("not valid: pass --next-step with the next investigation step")
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
