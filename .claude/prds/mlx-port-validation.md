---
name: mlx-port-validation
description: Make the macOS (MLX) port of RAT-Bench produce results that match the paper, or explain every gap
status: backlog
created: 2026-11-11T18:39:10Z
---

# PRD: mlx-port-validation

## Executive Summary

RAT-Bench (arXiv 2602.12806) now runs end to end on an Apple M1 Max: local LLM parts on MLX, the attacker of record through the claude CLI. At level 1, two of four anonymizers match the paper (Presidio 44% vs 41%, GLiNER 68% vs 67%). Scrubadub matched after restoring a detector that a later upstream commit disabled. The MLX Llama anonymizer does not match (52% vs 37%) and is not validated. This work validates the port so that its results can be used as benchmark results.

## Problem Statement

The port gives numbers, but they cannot be trusted yet:

- The MLX `llama_basic` anonymizer differs from the paper by 15 points. It ran in 8-bit, partly batched, while the paper used full precision on vLLM.
- The repo changed a lot after the paper-era commit `b8cface` (about 1,800 lines added in anonymizers, prompts, evaluation and re-identification code). One change (Scrubadub date-of-birth detector) already moved a result from 37% to 66%. Others may do the same.
- Batched MLX generation changed 4 of 20 attack answers against one-at-a-time generation.
- Only level 1, English, medical scenario, 4 anonymizers and 100 profiles were tested.
- The local tools that produced the Qwen and Gemma results live outside the repo.

## User Stories

1. As the benchmark owner, I want the MLX anonymizer to give the same rate as the vLLM reference, so that I can report MLX results as RAT-Bench results.
   - Acceptance: on level 1, the Claude rate for `llama_basic` anonymized on MLX (bf16) and on vLLM (full precision) differ by less than the 95% bootstrap interval half-width.
2. As the benchmark owner, I want every repo change since the paper-era code that alters results to be found and decided, so that no hidden drift biases the numbers.
   - Acceptance: a written audit lists each such change with a decision (restore or keep) and a reason; restored behaviors are committed with a test.
3. As the benchmark owner, I want levels 2 and 3 to be checked against the paper, so that the port is valid beyond the easy case.
   - Acceptance: for each in-scope anonymizer at levels 2 and 3, the rate is inside the 95% interval of the paper value, or the gap has a documented cause.
4. As a developer, I want a local attacker with a known offset to Claude, so that I can iterate without spending the subscription quota.
   - Acceptance: the offset of the development attacker per anonymizer and level is recorded next to the results.
5. As a new user on macOS, I want one documented command per level, so that I can reproduce the results.
   - Acceptance: a "Run on macOS" README section; a clean run from it reproduces the level 1 rates within their intervals.

## Functional Requirements

- FR1: An audit of `git diff b8cface HEAD` over `pii_benchmark/anonymizers`, `prompts.py`, `evaluation.py`, `uniqueness.py`, `utils.py`, classifying each change as bug fix, new feature, or result-altering behavior.
- FR2: The MLX anonymizer can run the full-precision model `mlx-community/Meta-Llama-3.1-8B-Instruct-bf16`, one prompt at a time by default.
- FR3: A determinism check: the same prompts generated twice on MLX give identical text.
- FR4: A vLLM reference run of `llama_basic` on RunPod (`scripts/runpod/launch.sh`, cost cap 5 USD) on the original level 1 data.
- FR5: Claude CLI attack of levels 2 and 3 for Presidio, GLiNER, Scrubadub, `llama_basic` and `llama_rescriber`, with `n_attack_repeats=3`.
- FR6: The MLX guess generator and its scoring move into `scripts/`, with their own `uv` project for mlx-lm 0.32 (transformers 5).
- FR7: Every result records the model revision, mlx-lm version, claude CLI version and git commit.

## Non-Functional Requirements

- Only one local model loaded at a time; the Mac must stay usable (lock in `pii_benchmark/mlx_engine.py`).
- Claude calls go through the claude CLI on the subscription, never the paid API.
- No paid spend without approval; RunPod capped at 5 USD per run.
- Secrets only in the git-ignored root `.env`; never committed.
- Runs resume after a crash without losing finished calls.

## Success Criteria

- `llama_basic` (MLX bf16 vs vLLM): Claude rate difference inside the 95% interval half-width.
- Level 1: all 5 in-scope anonymizers inside the paper's value ± interval, or explained.
- Levels 2 and 3: at least 4 of 5 anonymizers inside the interval, the rest explained.
- Drift audit: 100% of result-altering changes decided and documented.
- Reproduction from the README on a clean checkout gives level 1 rates within their intervals.

## Constraints & Assumptions

- Hardware: Apple M1 Max, 32 GB; bf16 Llama 3.1 8B needs 16 GB.
- Assumption: in-scope anonymizers are Presidio, GLiNER, Scrubadub, `llama_basic`, `llama_rescriber`.
- Assumption: where a change since `b8cface` alters results, the paper behavior is restored (as done for Scrubadub).
- Assumption: Claude Opus 4.8 through the CLI is the attacker of record in place of GPT-4.1.
- Assumption: work stays on branch `worktree-mlx-backend`.
- The vLLM reference needs a Hugging Face token with access to `meta-llama/Llama-3.1-8B-Instruct`.

## Out of Scope

- Spanish, Serbian, Dutch and Chinese benchmarks.
- API-based anonymizers (Azure, GPT-4.1, Gemini, Anthropic) and the iterative anonymizer.
- An exact GPT-4.1 replication of the attacker.
- Upstreaming changes to the authors' repository.
- Batched MLX generation as a default (stays opt-in).

## Dependencies

- `HF_TOKEN` in the root `.env` (step FR4).
- RunPod account and `RUNPOD_API_KEY` (present).
- claude CLI subscription quota for about 1,500 attack calls (levels 2-3, 3 repeats).
- Paper Table 1 values (levels Easy, Hard, Implicit).
