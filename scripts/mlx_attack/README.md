# MLX attacker

Generates attacker guesses with a recent MLX model (Qwen 3.5, Gemma 4) and scores them.
Two environments:

- `gen_guesses.py` runs in this folder's uv project. It pins `mlx-lm==0.32.2`, which needs
  transformers 5, so it cannot run in the benchmark venv. It takes the MLX lock file
  (`rat-bench-mlx.lock`, same as `pii_benchmark/mlx_engine.py`): one model at a time.
- `score_guesses.py` runs in the benchmark venv. It calls
  `pii_benchmark.attack.only_check_correctness` (CorrectMatch, no LLM).

Run both from the repo root.

Generate guesses (arguments: repo, suffix, input, output, methods, optional limit):

```bash
uv run --project scripts/mlx_attack python scripts/mlx_attack/gen_guesses.py mlx-community/Qwen3.5-9B-8bit qwen35 benchmark/level_1.jsonl benchmark/attacker_qwen35/level_1.jsonl presidio,gliner,scrubadub,llama_basic
```

Score them (use `--no-sync`: a plain `uv run` removes the spaCy model):

```bash
uv run --no-sync python scripts/mlx_attack/score_guesses.py --in benchmark/attacker_qwen35/level_1.jsonl --attacker qwen35 --out benchmark/attacker_qwen35_rescore --level 1
```

Each output row records `run_info_<suffix>` (model repo and revision from the Hugging Face
cache, mlx-lm version, claude CLI version, git commit).

`repro/qwen35_level1.json` is the scoring reproduction of the Qwen 3.5 9B level 1 guesses:
0.33, 0.63, 0.53, 0.42 for presidio, gliner, scrubadub, llama_basic.
