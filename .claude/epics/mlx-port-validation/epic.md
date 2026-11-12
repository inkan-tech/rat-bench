---
name: mlx-port-validation
status: backlog
created: 2026-11-11T18:39:10Z
updated: 2026-11-11T19:51:51Z
progress: 0%
prd: .claude/prds/mlx-port-validation.md
github: https://github.com/inkan-tech/rat-bench/issues/3
---

# Epic: mlx-port-validation

## Overview

Validate the macOS port of RAT-Bench: the MLX anonymizer must match a vLLM reference, every result-altering code change since the paper-era commit `b8cface` must be decided, and levels 1 to 3 must match the paper's Table 1 within the 95% interval or have a documented cause. Then package the macOS path so others can reproduce it.

## Architecture Decisions

- **Full precision on MLX for validation.** Use `mlx-community/Meta-Llama-3.1-8B-Instruct-bf16` (16 GB) for the anonymizer, so the only difference from the paper is the inference engine. 8-bit stays available for fast development.
- **One prompt at a time by default.** Batched generation changed 4 of 20 answers; it stays opt-in (`MLX_ATTACK_BATCH=1`), and the anonymizer gets the same rule.
- **vLLM as the reference engine.** The paper's engine, run once on RunPod through the existing `scripts/runpod/launch.sh` (cost cap 5 USD).
- **Claude CLI as attacker of record.** Already matches GPT-4.1 on Presidio and GLiNER at level 1. Thinking off, 4096 output tokens.
- **Separate environment for recent MLX models.** mlx-lm 0.32 needs transformers 5, which breaks `flair`; the Qwen/Gemma generator gets its own `uv` project under `scripts/`, and scoring stays in the benchmark environment.
- **Restore paper behavior on result-altering drift.** Same rule as the Scrubadub date-of-birth fix (commit `b0d4550`).

## Technical Approach

### Frontend Components

None. Results are reported in the existing private claude.ai page (RAT-Bench Level 1 Attackers), extended with levels 2 and 3.

### Backend Services

- `pii_benchmark/mlx_engine.py`: add the bf16 repo to `MLX_REPOS`; select it through `--llama_version` or an environment variable; make the anonymizer path one-at-a-time unless batching is asked for.
- `pii_benchmark/attack.py`: already resumes per call; use `attack_repeated` for 3 repeats.
- Drift fixes land in the files the audit names (anonymizers, `prompts.py`, `evaluation.py`, `uniqueness.py`, `utils.py`).
- `scripts/mlx_attack/`: guess generator (`gen_guesses.py`) and scoring wrapper (`only_check_correctness`) moved from the job temp folder.

### Infrastructure

- Apple M1 Max, 32 GB, one model at a time (file lock).
- RunPod RTX 4090 pod for the single vLLM reference run.
- claude CLI subscription for attack calls; per-call log in `benchmark/claude_cli_calls.jsonl`.

## Implementation Strategy

1. Audit first: it is free and can explain the `llama_basic` gap before any spend.
2. bf16 MLX run and determinism check next, still free.
3. vLLM reference only after 1 and 2, to decide whether MLX itself causes the gap.
4. Levels 2 and 3 after the anonymizer is validated, so their MLX texts are trusted.
5. Packaging last, from the final commands.

Risk: if bf16 MLX still differs from vLLM, the port is not valid for LLM anonymizers; stop and investigate sampling, chat template and tokenizer before step 4.

## Task Breakdown Preview

| # | Task | Depends on | Parallel |
|---|---|---|---|
| #4 | Audit code drift since `b8cface` and decide each result-altering change | – | yes |
| #5 | Add bf16 MLX anonymizer, one-at-a-time default, determinism check | – | yes |
| #6 | Re-run level 1 `llama_basic` on bf16 MLX and attack with Claude | #5 | no |
| #7 | vLLM reference run on RunPod and MLX vs vLLM comparison | #6 | no |
| #8 | Apply the audit's restore decisions with tests | #4 | yes |
| #9 | Levels 2 and 3: anonymize (5 methods) and attack with Claude, 3 repeats | #7, #8 | no |
| #10 | Move the MLX guess generator and scoring into `scripts/` with their own uv project; record versions in results | – | yes |
| #11 | README "Run on macOS", results page update with levels 2-3, final comparison with Table 1 | #9, #10 | no |

## Dependencies

- `HF_TOKEN` in the root `.env` for task #7.
- Paper Table 1 (Easy, Hard, Implicit columns).
- claude CLI quota: about 1,500 attack calls for task #9.

## Success Criteria (Technical)

- Task #5: the same 5 prompts generated twice give byte-identical text.
- Task #7: Claude rate on MLX bf16 texts and on vLLM texts differ by less than the 95% interval half-width; mean BLEU between the two text sets reported.
- Task #8: each restored behavior has a test that fails on the drifted code and passes after.
- Task #9: per anonymizer and level, rate inside the paper value ± interval, or a documented cause.
- Task #11: a clean checkout following the README reproduces the level 1 rates within their intervals.

## Estimated Effort

| Task | Effort | Machine time | Spend |
|---|---|---|---|
| #4 | 2 h | – | 0 |
| #5 | 1 h | 15 min | 0 |
| #6 | 30 min | about 3 h | 0 |
| #7 | 1 h | about 30 min on pod | under 5 USD |
| #8 | 2 h | – | 0 |
| #9 | 2 h | about 12 h MLX + 1 h Claude | plan quota |
| #10 | 2 h | – | 0 |
| #11 | 2 h | 30 min | 0 |

## Tasks Created
- [ ] #4 - Audit code drift since the paper-era commit (parallel: true)
- [ ] #5 - Add the bf16 MLX anonymizer with a determinism check (parallel: true)
- [ ] #6 - Re-run level 1 llama_basic on bf16 MLX with the Claude attacker (parallel: false)
- [ ] #7 - vLLM reference run on RunPod and MLX comparison (parallel: false)
- [ ] #8 - Apply the drift audit's restore decisions with tests (parallel: true)
- [ ] #9 - Run levels 2 and 3 with the Claude attacker, 3 repeats (parallel: false)
- [ ] #10 - Package the MLX guess generator and record run versions (parallel: true)
- [ ] #11 - Document the macOS run and publish the final comparison (parallel: false)

Total tasks: 8
Parallel tasks: 4
Sequential tasks: 4
Estimated total effort: 12.5 hours of work, plus about 16 h of machine time

## Results

- #6 level 1 `llama_basic`, bf16 MLX anonymizer, Claude attacker (claude-opus-4-8): re-identification rate 0.53 (95% bootstrap interval 0.42 to 0.63), paper 0.37, 8-bit MLX 0.52. 100 calls, 0 refusals, 0 cut answers. The 8-bit value used the pre-restore attacker prompt; this run uses the restored paper-era prompt (#8, D44/D45). Details: `docs/results/level1_llama_basic_bf16.json`.
