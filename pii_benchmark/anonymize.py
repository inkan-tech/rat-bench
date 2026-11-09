import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List
from tqdm import tqdm

from pii_benchmark.anonymizers.get_anonymizers import get_anonymizer
import argparse
import time

from pii_benchmark.utility import utility_scores
from synthetic_data_generation.utils import write_output_async as write_output

# Most anonymizers here are I/O-bound (an API call per profile); running them
# one at a time serializes purely on network latency for no benefit. This
# matches attack.py's existing max_workers=10 pattern for attacker calls.
MAX_WORKERS = 10


def _run_anonymizer_loop(anonymizer, profiles, output_key, output_file, timing_flag, utility_flag, call_fn):
    """Run anonymization over all profiles concurrently, recording timing and results."""
    to_process = [p for p in profiles if f"text_anon_{output_key}" not in p]

    def _process(profile):
        start_time = time.perf_counter()
        anon_text = call_fn(anonymizer, profile)
        return profile, anon_text, time.perf_counter() - start_time

    if to_process:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_process, p): p for p in to_process}
            for future in tqdm(as_completed(futures), total=len(to_process)):
                profile, anon_text, elapsed = future.result()
                profile[f"text_anon_{output_key}"] = anon_text
                if timing_flag:
                    profile[f"runtime_{output_key}"] = elapsed
                if utility_flag:
                    scores = utility_scores(anon_text, profile["text"])
                    profile[f"rouge_score_{output_key}"] = scores[0]
                    profile[f"bleu_score_{output_key}"] = scores[1]

    write_output(output_file, profiles)

def _run_anonymizer_loop_sequential(anonymizer, profiles, output_key, output_file, timing_flag, utility_flag, call_fn):
    """Run anonymization over all profiles concurrently, recording timing and results."""
    to_process = [p for p in profiles if f"text_anon_{output_key}" not in p]

    def _process(profile):
        start_time = time.perf_counter()
        anon_text = call_fn(anonymizer, profile)
        return profile, anon_text, time.perf_counter() - start_time

    if to_process:

        for profile in tqdm(to_process, total=len(to_process)):
            profile, anon_text, elapsed = _process(profile)
            profile[f"text_anon_{output_key}"] = anon_text
            if timing_flag:
                profile[f"runtime_{output_key}"] = elapsed
            if utility_flag:
                scores = utility_scores(anon_text, profile["text"])
                profile[f"rouge_score_{output_key}"] = scores[0]
                profile[f"bleu_score_{output_key}"] = scores[1]

    write_output(output_file, profiles)


def _run_anonymizer_batched(anonymizer, profiles, output_key, output_file, timing_flag, utility_flag, **batch_kwargs):
    """Run a batch-capable anonymizer over all profiles in one call.

    Timing is per-profile elsewhere, but batching makes only the wall time of
    the whole batch meaningful, so that total is split evenly across profiles.
    """
    to_process = [p for p in profiles if f"text_anon_{output_key}" not in p]

    if to_process:
        start_time = time.perf_counter()
        anon_texts = anonymizer.anonymize_batch(
            [p["text"] for p in to_process],
            scenarios=[p["scenario"] for p in to_process],
            **batch_kwargs,
        )
        elapsed = time.perf_counter() - start_time

        for profile, anon_text in zip(to_process, anon_texts):
            profile[f"text_anon_{output_key}"] = anon_text
            if timing_flag:
                profile[f"runtime_{output_key}"] = elapsed / len(to_process)
            if utility_flag:
                scores = utility_scores(anon_text, profile["text"])
                profile[f"rouge_score_{output_key}"] = scores[0]
                profile[f"bleu_score_{output_key}"] = scores[1]

        print(f"Batched {len(to_process)} profiles in {elapsed:.1f}s")

    write_output(output_file, profiles)


# Anonymization main function
def run_anonymization(profiles: List[dict], anon_methods:List[str], results_path:str, scenario:str, language:str, level:int,
                      gemini_version:str|None=None, llama_version:str|None=None, gpt_version:str|None=None, anthropic_version:str|None=None,
                      claude_code_version:str|None=None, epsilon:int|None=None, temperature:int|None=None, attribute_list_iterative:str|None=None,
                      timing_flag:bool=True, utility_flag:bool=True):
    print("Anonymizing")
    print(f"Will save results to {results_path}")
    llama_idx = -1

    anonymizers = [
        get_anonymizer(
            method=anon_method,
            gemini_version=gemini_version,
            llama_version=llama_version,
            gpt_version=gpt_version,
            scenario=scenario,
            epsilon=epsilon,
            temperature=temperature,
            attribute_list_iterative=attribute_list_iterative,
            anthropic_version=anthropic_version,
            claude_code_version=claude_code_version,
            language=language
        )
        if anon_method not in ["llama_basic", "llama_full", "llama", "llama_clio"]
        else None
        for anon_method in anon_methods
    ]

    if (
        "llama" in anon_methods
        or "llama_basic" in anon_methods
        or "llama_full" in anon_methods
        or "llama_clio" in anon_methods
    ):
        anonymizers.append(
            get_anonymizer(method="llama", llama_version=llama_version, scenario=scenario)
        )
        llama_idx = len(anonymizers) - 1

    llama_attributes = {
        "llama": [
            "SSN",
            "phone number",
            "credit card number",
            "email",
            "name",
            "address",
        ],
        "llama_full": [
            "SSN",
            "phone number",
            "credit card number",
            "email",
            "name",
            "address",
            "race",
            "citizenship status",
            "educational attainment",
            "state of residence",
            "occupation",
            "marital status",
            "employment status",
            "date of birth",
            "age",
        ],
    }

    output_base = Path(results_path)
    output_dir = output_base.parent if output_base.suffix == ".jsonl" else output_base
    output_file = output_base if output_base.suffix == ".jsonl" else output_base / f"level_{level}.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, method in enumerate(anon_methods):
        print(f"Anonymizing with {method}")

        # if f"text_anon_{method}" in profiles[0]:
        #     print(f"Skipping {method} because it already exists in the profiles")
        #     continue

        if method == "llama":
            _run_anonymizer_loop(
                anonymizers[llama_idx], profiles, "llama", output_file, timing_flag, utility_flag,
                lambda a, p: a.anonymize(p["text"], prompt_type="anthropic_attributes",
                                         attributes=llama_attributes["llama"], scenario=p["scenario"])
            )
        elif method == "llama_full":
            _run_anonymizer_loop(
                anonymizers[llama_idx], profiles, "llama_full", output_file, timing_flag, utility_flag,
                lambda a, p: a.anonymize(p["text"], prompt_type="anthropic_attributes",
                                         attributes=llama_attributes["llama_full"], scenario=p["scenario"])
            )
        elif method == "llama_basic":
            # Uses LlamaAnonymizer's vLLM-backed anonymize_batch, which
            # batches all profiles into a single continuous-batching call
            # (same speedup as llama_rescriber).
            _run_anonymizer_batched(
                anonymizers[llama_idx], profiles, "llama_basic", output_file, timing_flag, utility_flag,
                prompt_type="anthropic",
            )
        elif method == "llama_rescriber":
            # Uses the vLLM-backed LlamaRescriberAnonymizer, which batches.
            _run_anonymizer_batched(
                anonymizers[i], profiles, "llama_rescriber", output_file, timing_flag, utility_flag,
            )
        elif method == "llama_clio":
            _run_anonymizer_loop(
                anonymizers[llama_idx], profiles, "llama_clio", output_file, timing_flag, utility_flag,
                lambda a, p: a.anonymize(p["text"], prompt_type="clio", scenario=p["scenario"])
            )
        elif method == "iterative":
            print(f"Iterative anonymizer, attribute list = {attribute_list_iterative}")
            _run_anonymizer_loop(
                anonymizers[i], profiles, f"iterative_{attribute_list_iterative}", output_file, timing_flag, utility_flag,
                lambda a, p: a.anonymize(p)
            )
        elif method in ("madlib", "tem"):
            _run_anonymizer_loop(
                anonymizers[i], profiles, f"{method}_eps{epsilon}", output_file, timing_flag, utility_flag,
                lambda a, p: a.anonymize(p["text"], scenario=p["scenario"])
            )
        elif method == "dp_prompt_gpt":
            _run_anonymizer_loop(
                anonymizers[i], profiles, f"dp_prompt_gpt_temp{temperature}", output_file, timing_flag, utility_flag,
                lambda a, p: a.anonymize(p["text"], scenario=p["scenario"])
            )
        elif method == "privacy_filter":
            _run_anonymizer_loop_sequential(
                anonymizers[i], profiles, f"privacy_filter", output_file, timing_flag, utility_flag,
                lambda a, p: a.anonymize(p["text"], scenario=p["scenario"])
            )
        else:
            # Covers most methods (presidio, azure, anthropic_basic, gpt_basic,
            # gemini_basic, scrubadub, gliner, uniner, textwash, etc.) -- kept
            # as its own branch rather than routed through _run_anonymizer_loop
            # to avoid changing behavior beyond concurrency: unlike that
            # function, this path has never computed utility (rouge/bleu)
            # scores, and still doesn't here.
            anonymizer = anonymizers[i]
            method_name = anon_methods[i]
            to_process = [p for p in profiles if f"text_anon_{method_name}" not in p]

            def _process(profile):
                start_time = time.perf_counter()
                anon_text = anonymizer.anonymize(profile["text"], scenario=profile["scenario"])
                return profile, anon_text, time.perf_counter() - start_time

            if to_process:
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = {executor.submit(_process, p): p for p in to_process}
                    for future in tqdm(as_completed(futures), total=len(to_process)):
                        profile, anon_text, elapsed = future.result()
                        if timing_flag:
                            profile[f"runtime_{method_name}"] = elapsed
                        profile[f"text_anon_{method_name}"] = anon_text
            write_output(output_file, profiles)

    write_output(output_file, profiles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--gemini_version", type=str, default="2.5-flash")
    parser.add_argument("--llama_version", type=str, default="3.1-8B-Instruct")
    parser.add_argument("--gpt_version", type=str, default="gpt-4o-mini")
    parser.add_argument("--anthropic_version", type=str, default="claude-haiku-4-5-20251001")
    parser.add_argument("--claude_code_version", type=str, default="haiku")
    parser.add_argument("--epsilon", type=float, default=10.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--anon_methods", type=str, default=None)
    parser.add_argument("--results_path", type=str, default="")
    parser.add_argument("--scenario", type=str, default="medical_data")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--timing", type=int, default=0)
    parser.add_argument("--attribute_list", type=str, default="ours")
    parser.add_argument("--language", type=str, default="English")
    args = parser.parse_args()

    DATA_PATH = args.data_path
    ANON_METHODS = args.anon_methods
    ANON_METHODS = [s.strip() for s in ANON_METHODS.split(",")]
    print(f"Anonymization methods: {ANON_METHODS}")
    GEMINI_VERSION = args.gemini_version
    LLAMA_VERSION = args.llama_version
    GPT_VERSION = args.gpt_version
    ANTHROPIC_VERSION = args.anthropic_version
    EPSILON = args.epsilon
    TEMPERATURE = args.temperature
    RESULTS_PATH = args.results_path
    SCENARIO = args.scenario
    LEVEL = args.level
    TIMING_FLAG = args.timing
    LANGUAGE = args.language

    profiles = []
    with open(DATA_PATH, "r") as f:
        for line in f:
            profiles.append(json.loads(line))

    run_anonymization(
        profiles,
        ANON_METHODS,
        RESULTS_PATH,
        SCENARIO,
        LANGUAGE,
        LEVEL,
        gemini_version=GEMINI_VERSION,
        llama_version=LLAMA_VERSION,
        gpt_version=GPT_VERSION,
        anthropic_version=ANTHROPIC_VERSION,
        claude_code_version=args.claude_code_version,
        epsilon=EPSILON,
        temperature=TEMPERATURE,
        attribute_list_iterative=args.attribute_list,
        timing_flag=TIMING_FLAG
    )
