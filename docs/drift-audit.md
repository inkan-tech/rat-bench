# Code drift audit since the paper-era commit

Issue: #4. Paper: arXiv 2602.12806v1.

## Scope and method

- Base: `b8cface` (29 January 2026), the code version closest to the paper.
- Head: `4cd8462`.
- Command: `git diff b8cface 4cd8462 -- pii_benchmark/anonymizers/ pii_benchmark/prompts.py pii_benchmark/evaluation.py pii_benchmark/uniqueness.py pii_benchmark/utils.py`.
- 33 files changed (1842 lines added, 435 removed). Each file has one or more rows below.
- Commits in scope: `46d64dd`, `54721b4`, `994ff7c`, `1a80fe2`, `6f7d65e`, `81fa9ff`, `9502f26`, `0b29628`, `91eb502`, `39b9cb6`, `29bbff7`, `b0d4550`.
- In-scope anonymizers for the port: Presidio, GLiNER, Scrubadub, `llama_basic`, `llama_rescriber`. Rows on other anonymizers are "keep" because the port does not run them. The reason cell says what to check before a rerun.
- Classes: bug fix (old code crashed or read a wrong path), new feature (new code path that the English PUMS run does not use), result-altering (the English PUMS benchmark output can change).
- Decision applies to result-altering rows only. "restore" = task #8 brings back the paper-era behavior (or makes it the default for the parity run).

## Audit table

| ID | File | Change | Class | Decision | Reason | Follow-up |
| --- | --- | --- | --- | --- | --- | --- |
| D1 | pii_benchmark/anonymizers/anon_prompts/clio_summarization_concert.txt | New Clio summary prompt for the concert scenario | new feature | n/a | Not used by the PUMS levels 1-3 (scenarios are medical, chatbot, meeting) | - |
| D2 | pii_benchmark/anonymizers/anon_prompts/clio_summarization_meeting.txt | New Clio summary prompt for the meeting scenario | new feature | n/a | Used only by Clio anonymizers, which are out of port scope | - |
| D3 | pii_benchmark/anonymizers/anon_prompts/clio_summarization_topic.txt | New Clio summary prompt for the topic scenario | new feature | n/a | Not used by the PUMS levels 1-3 | - |
| D4 | pii_benchmark/anonymizers/anon_prompts/clio_summarization_tourist.txt | New Clio summary prompt for the tourist scenario | new feature | n/a | Not used by the PUMS levels 1-3 | - |
| D5 | pii_benchmark/anonymizers/anon_prompts/rescriber_meeting.txt | New Rescriber prompt for meeting transcripts | new feature | n/a | Selected only through the scenario routing of D27; the decision is in D27 | - |
| D6 | pii_benchmark/anonymizers/anonymizer.py | Abstract `anonymize` gets a `scenario` parameter | new feature | n/a | Interface change only; no output change | - |
| D7 | pii_benchmark/anonymizers/anthropic.py | Rewrite: default model `claude-haiku-4-5-20251001` (was `claude-opus-4-1-20250805`), `temperature=0` removed, `max_tokens` 1000 to 3000, Rescriber and language prompts | result-altering | keep | Out of port scope (API anonymizer, not run). Before any rerun: set `temperature=0` again and pin the paper model | - |
| D8 | pii_benchmark/anonymizers/azure.py | `language` parameter passed to the Azure recognizer | new feature | n/a | Default stays English; out of port scope | - |
| D9 | pii_benchmark/anonymizers/clio.py | Import from `pii_benchmark.credentials`; return `content[0].text` instead of the content list | bug fix | n/a | Old code returned a list object, not text | - |
| D10 | pii_benchmark/anonymizers/euguardrail.py | New EU guardrail anonymizer | new feature | n/a | New method key `eu_guardrail`; not in the paper | - |
| D11 | pii_benchmark/anonymizers/gemini.py | Passes `scenario`; empty answer is retried; after all retries it returns the input text (was `None`) | result-altering | keep | Out of port scope (API anonymizer). Note: a failed call now leaks the original text into the attack | - |
| D12 | pii_benchmark/anonymizers/get_anonymizers.py | New method keys (presidio_es/srb/nl, azure_*, anthropic_*, *_public, scrubadub_es/nl, textwash_nl, eu_guardrail, privacy_filter); `llama_rescriber` now gets `llama_version` | new feature | n/a | Existing English keys build the same classes; default version stays 3.1-8B-Instruct | - |
| D13 | pii_benchmark/anonymizers/get_anonymizers.py | Unknown method name raises `ValueError` (was silent `None`) | bug fix | n/a | Old code failed later with an unclear error | - |
| D14 | pii_benchmark/anonymizers/gliner.py | `anonymize` signature gets `scenario`; body unchanged | new feature | n/a | Same entities, same masking, same chunking; no output change | - |
| D15 | pii_benchmark/anonymizers/gpt_anon.py | `self.attributes` no longer passed to the prompt; `scenario` and `language` passed; Rescriber `results` list unwrapped | result-altering | keep | Out of port scope (API anonymizer). Regression to fix before a GPT rerun: attribute lists are dropped | - |
| D16 | pii_benchmark/anonymizers/iterative.py | Pipeline loads with bf16/fp16 dtype and `device_map=auto` | result-altering | keep | Out of port scope; precision change only affects the iterative anonymizer | - |
| D17 | pii_benchmark/anonymizers/languages_config.yml | New spaCy model config for es, hr, nl, de | new feature | n/a | Used only by non-English Presidio paths | - |
| D18 | pii_benchmark/anonymizers/llama.py | HF `pipeline` replaced by `get_engine` from `pii_benchmark.backend` (vLLM, MLX, RunPod); new `anonymize_batch` | new feature | n/a | Backend swap; the decoding and precision effects are rows D19 and D20 | - |
| D19 | pii_benchmark/anonymizers/llama.py | Decoding is greedy (`temperature=0.0`); paper-era HF pipeline used the model generation config (sampling, Llama 3.1 8B Instruct: do_sample, T 0.6, top_p 0.9) | result-altering | keep | Greedy output is needed for the byte-identical check of #5; sampled paper outputs cannot be reproduced token for token anyway | - |
| D20 | pii_benchmark/anonymizers/llama.py | Model runs in bf16 (engine) instead of the pipeline default dtype | result-altering | keep | bf16 is the reference precision for the MLX comparison in #6; the change is small next to sampling noise | - |
| D21 | pii_benchmark/anonymizers/llama.py | `anonymize_rescriber` passes the profile scenario, so chatbot and meeting profiles get their own Rescriber prompt; paper-era always used `rescriber_medical.txt` | result-altering | restore | Paper numbers used the medical Rescriber prompt for every profile; 73 of 100 level-1 profiles are chatbot or meeting | #8 |
| D22 | pii_benchmark/anonymizers/llama.py | Rescriber output cap 4096 to 2048 tokens (`MAX_OUTPUT_TOKENS_RESCRIBER`) | result-altering | keep | Measured worst case is about 1600 tokens, so the cap only cuts runaway generations | - |
| D23 | pii_benchmark/anonymizers/llama.py | `anonymize_clio` is one step (prompt plus summary instructions in one system prompt); paper-era asked a second turn | result-altering | keep | Clio is out of port scope; must be revisited before any Clio rerun | - |
| D24 | pii_benchmark/anonymizers/llamaclio.py | Same one-step Clio change; takes `scenario` per call | result-altering | keep | Clio is out of port scope; must be revisited before any Clio rerun | - |
| D25 | pii_benchmark/anonymizers/llamarescriber.py | HF `pipeline` replaced by the shared engine; new `anonymize_batch`; debug prints removed | new feature | n/a | Backend swap; decoding effects are the same as D19 and D20 | - |
| D26 | pii_benchmark/anonymizers/llamarescriber.py | Output cap 2048 tokens (paper-era class used 4096); greedy decoding | result-altering | keep | Same reasons as D19 and D22: determinism for #5, worst case about 1600 tokens | - |
| D27 | pii_benchmark/anonymizers/llamarescriber.py | Prompt built with `scenario or self.scenario`, so chatbot and meeting profiles get their own Rescriber prompt; paper-era `llama_rescriber` (through `LlamaAnonymizer.anonymize_rescriber`) always used the medical prompt | result-altering | restore | Same as D21: the paper run used the medical prompt for all scenarios; keep scenario prompts behind an option | #8 |
| D28 | pii_benchmark/anonymizers/llamarescriber.py | Fallback parser loop stops at `len(lines)-1`; `except` narrowed to `JSONDecodeError`; redaction uses `str.replace` | bug fix | n/a | Old loop could read past the last line (IndexError); `replace` masks the same spans as the old while loop | - |
| D29 | pii_benchmark/anonymizers/madlib.py | `anonymize` signature gets `scenario` | new feature | n/a | Interface change only | - |
| D30 | pii_benchmark/anonymizers/openaifilter.py | New OpenAI filter anonymizer | new feature | n/a | Not in the paper | - |
| D31 | pii_benchmark/anonymizers/presidio.py | English analyzer now really uses `en_core_web_sm`; paper-era code built that engine, then replaced it with `AnalyzerEngine()`, whose default NLP engine is `en_core_web_lg` | result-altering | restore | The paper Presidio numbers came from the large spaCy model; the small model finds fewer PERSON, LOCATION and NRP spans | #8 |
| D32 | pii_benchmark/anonymizers/presidio.py | New es, sr and nl paths with their own registry | new feature | n/a | Not used by English runs. They read a hardcoded `/data/natasa/...` config path and fail on other machines | - |
| D33 | pii_benchmark/anonymizers/privacyfilter.py | New privacy filter anonymizer | new feature | n/a | Not in the paper | - |
| D34 | pii_benchmark/anonymizers/scrubadub.py | `DateOfBirthDetector` was removed for all locales in `1a80fe2`; `b0d4550` adds it back for English locales only | result-altering | restore | Without it, level-1 re-identification rose from about 0.37-0.44 to 0.66; already restored in `b0d4550` | b0d4550 |
| D35 | pii_benchmark/anonymizers/tem.py | `anonymize` signature gets `scenario` | new feature | n/a | Interface change only | - |
| D36 | pii_benchmark/anonymizers/textwash.py | `anonymize` signature gets `scenario`; tokenizer `local_files_only=False` (library default) | new feature | n/a | No output change for English | - |
| D37 | pii_benchmark/anonymizers/textwash_config.py | Data paths resolved from the module folder instead of the working directory | bug fix | n/a | Old relative paths failed outside the repo root | - |
| D38 | pii_benchmark/anonymizers/uniner.py | `anonymize` signature gets `scenario`; per-entity print removed | new feature | n/a | No output change | - |
| D39 | pii_benchmark/anonymizers/vllm_engine.py | New shared vLLM engine factory | new feature | n/a | Backend code; decoding effects are D19 and D20 | - |
| D40 | pii_benchmark/evaluation.py | `check_age` returns 0 when the ground truth is not a number | bug fix | n/a | Old code raised ValueError on missing age | - |
| D41 | pii_benchmark/evaluation.py | New checker dispatch for EDAD, age keys, JMBG, RRN, CURP, dob and dom date parts | new feature | n/a | PUMS keys (AGEP, OCCP, SCHL, ST, SSN, DOB parts, name, email) still reach the same checkers with the same thresholds (1.0 and 0.75) | - |
| D42 | pii_benchmark/evaluation.py | Ground truth read only from `indirect_identifiers`; fallback to `ground_truth` key removed | result-altering | keep | Levels 1-3 have `indirect_identifiers` in 100 of 100 profiles and no `ground_truth` key, so no effect | - |
| D43 | pii_benchmark/evaluation.py | Guess key resolution rewritten with aliases for SRB and NL labels | new feature | n/a | For PUMS keys the lookup gives the same key as before (`DOB` for the date parts) | - |
| D44 | pii_benchmark/prompts.py | English attacker header changed: question mark replaced by "Remember to make your guesses in English, from the given options..."; medical header adds "in {language}" | result-altering | restore | The attacker prompt must match paper Appendix B for the parity run; keep the new text only for non-English runs | #8 |
| D45 | pii_benchmark/prompts.py | Attacker JSON footer quotes attribute keys (`"sex" :{` instead of `sex :{`) | result-altering | restore | Changes the attacker output format and the parse rate compared to the paper run | #8 |
| D46 | pii_benchmark/prompts.py | New prompts: language redactor prompt, `public_info` prompt, Spanish attacker prompts, concert, tourist and topic scenarios | new feature | n/a | English `anthropic` prompt text is unchanged | - |
| D47 | pii_benchmark/prompts.py | Clio prompt returns one string (intro plus summary prompt) instead of two prompts | result-altering | keep | Clio is out of port scope; linked to D23 and D24 | - |
| D48 | pii_benchmark/prompts.py | Rescriber prompt routing accepts long scenario names and a meeting prompt; returns empty string for unknown methods | new feature | n/a | Has an effect only through the callers in D21 and D27, where the decision is restore | - |
| D49 | pii_benchmark/uniqueness.py | Map paths `data/pums/maps/` to `data/maps/` | bug fix | n/a | `b8cface` already stored the maps in `data/maps/`; old path did not exist | - |
| D50 | pii_benchmark/uniqueness.py | Occupation "CON-EARTH DRILLERS, EXCEPT OIL AND GAS" mapped to code 411 | bug fix | n/a | Old code raised KeyError for this label | - |
| D51 | pii_benchmark/uniqueness.py | MEX, SRB and NL datasets with their own encoders; SRB and NL CSV files load at import time | new feature | n/a | PUMS path unchanged; import now needs `data/srb/data/sample_enc.csv` and `data/nl/data/sample_enc.csv` (present) | - |
| D52 | pii_benchmark/uniqueness.py | Re-identification rule: CorrectMatch correctness `== 1` changed to `>= 0.2` (`1a80fe2`, constant `REID_CORRECTNESS_THRESHOLD` in `9502f26`) | result-altering | keep | Matches the paper Section 4 rule (threshold 0.2); the paper-era `== 1` did not | - |
| D53 | pii_benchmark/uniqueness.py | Bootstrap confidence interval for the re-identification rate | new feature | n/a | Adds a key to the results; the point rate is unchanged | - |
| D54 | pii_benchmark/uniqueness.py | `fit_model_and_compute` applies `dropna().astype(int)` to the fit data | result-altering | keep | PUMS sample (2,022,378 rows) has no missing value in the checked columns, so no effect on English | - |
| D55 | pii_benchmark/utils.py | `PUMS_MAPS_PATH` from `data/pums/maps/` to `data/maps/` | bug fix | n/a | Same as D49 | - |
| D56 | pii_benchmark/utils.py | MEX, SRB, NL label maps; `att_names` extended (adds "age", "ethnicity" and others); `type_to_options` gets `language` | new feature | n/a | No PUMS feature uses the new labels; English options text unchanged | - |
| D57 | pii_benchmark/utils.py | `parse_output_gpt` catches parse errors and keeps the partial result; optional interactive fallback (off by default) | bug fix | n/a | Old code crashed the whole run on one bad answer | - |
| D58 | pii_benchmark/utils.py | Bare `except` narrowed to `IndexError` and `JSONDecodeError`; dead `raise` removed | bug fix | n/a | Only those errors can occur in the guarded lines | - |
| D59 | pii_benchmark/utils.py | `load_data` no longer falls back to the Hugging Face dataset when `data_path` is None | result-altering | keep | Local `benchmark/level_*.jsonl` is the data source of the port; confirm once that it equals the published dataset | - |
| D60 | pii_benchmark/utils.py | `write_output` overwrites the file; the old merge into an existing file is removed | result-altering | keep | Each run writes a clean file; the merge mixed columns of earlier runs. Callers must write full records | - |
| D61 | pii_benchmark/utils.py | New `retry_with_backoff` helper | new feature | n/a | Used by API callers only | - |

Summary: 61 rows. 22 result-altering: 6 restore (D21, D27, D31, D44, D45 for #8; D34 already done in `b0d4550`) and 16 keep.

## Re-identification rule

Paper reference: arXiv 2602.12806v1, Section 4. A profile counts as re-identified when:

1. the attacker recovers at least one direct identifier (name, SSN, email, phone number, credit card number, address): risk 1; otherwise
2. the CorrectMatch correctness of the record built from the correctly guessed indirect attributes is at or above the threshold 0.2.

Code at `4cd8462` (`pii_benchmark/uniqueness.py`, `_is_reided`):

```python
REID_CORRECTNESS_THRESHOLD = 0.2
return (n_direct_ids > 0) | (correctness >= REID_CORRECTNESS_THRESHOLD)
```

Result: the code matches the Section 4 rule (direct identifier, then CorrectMatch correctness >= 0.2).

Findings:

- Paper-era code (`b8cface`) did not match the paper: it counted a profile only when correctness `== 1`. Commit `1a80fe2` changed the rule to `>= 0.2`. The same commit removed the Scrubadub `DateOfBirthDetector` (D34). So the jump of level-1 re-identification from about 37% to 66% has two causes in one commit. Task #8 must measure the Scrubadub effect with the 0.2 rule fixed, not against the paper-era `== 1` numbers.
- Population size for PUMS in `compute_reid_risk`: `n = 306169200`, same as `b8cface`. The module constant `N = 309349689` is used only when no `n` is given.
- CorrectMatch correctness is the mean of 10 calls of `individual_correctness` on one fitted model, same as `b8cface`.
- The task brief also gives "k=5". This audit did not find a parameter that maps to k=5 in the scope files. Open question for the benchmark owner.
- The rule text above comes from the task brief. The paper PDF was not read again during this audit.

## Follow-up items for task #8

- D21, D27: use the medical Rescriber prompt for every scenario in the parity run (option to keep scenario prompts).
- D31: English Presidio analyzer back to the default engine (`en_core_web_lg`).
- D34: done in `b0d4550`; re-measure level 1 with the 0.2 rule.
- D44, D45: English attacker header and footer back to the `b8cface` text.
