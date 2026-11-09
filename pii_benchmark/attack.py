import argparse
import json
import pickle
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import stats
from tqdm import tqdm

from pii_benchmark.attackers.get_attacker import get_attacker
from pii_benchmark.evaluation import check_guess_correctness
from synthetic_data_generation.utils import write_output_async as write_output
from pii_benchmark.uniqueness import compute_reid_risk

def _drop_never_attacked_methods(profiles, anon_methods, attacker_name):
    """Drop any method with zero guesses for this attacker at all (never
    attacked yet), since there's no profile subset that could make it usable.
    """
    kept, dropped = [], []
    for m in anon_methods:
        guess_key = f"guesses_{m}_{attacker_name}"
        if any(guess_key in p for p in profiles):
            kept.append(m)
        else:
            dropped.append(m)
    return kept, dropped


def _profiles_with_full_coverage(profiles, anon_methods, attacker_name):
    """The subset of profiles that have a guess for every method in
    anon_methods/attacker_name, plus the ids left out.

    `check_guess_correctness`/`compute_reid_risk` index guesses by
    `profile[f"guesses_{method}_{attacker_name}"]` unconditionally for every
    profile they're given, so a single partially-attacked method poisons the
    whole run with a KeyError. Restricting to the common subset (rather than
    dropping the incomplete methods) keeps every requested method in the
    output, computed on the same, directly comparable set of profiles.
    """
    guess_keys = [f"guesses_{m}_{attacker_name}" for m in anon_methods]
    covered = [p for p in profiles if all(k in p for k in guess_keys)]
    covered_ids = {p["id"] for p in covered}
    excluded_ids = sorted(p["id"] for p in profiles if p["id"] not in covered_ids)
    return covered, excluded_ids


def only_check_correctness(profiles, anon_methods, attacker_name, scenario,
           results_path, uniqueness_results_path, level, language, dataset, available_only=False):
    """Recompute correctness and reidentification risk from existing guesses
    (no attacker calls).

    available_only=True restricts the computation to the subset of profiles
    that have a guess for every requested method, so this can be run against
    a partially completed/partially re-attacked dataset (e.g. while some
    profiles are still waiting on a re-attack) instead of failing on the
    first gap. Methods never attacked at all are dropped outright (there's no
    profile subset that would make them usable); methods that are merely
    incomplete are kept, just computed over fewer profiles. The full profile
    list is still written back to disk -- only the reid-risk computation
    itself is restricted to the covered subset.
    """
    if available_only:
        anon_methods, never_attacked = _drop_never_attacked_methods(profiles, anon_methods, attacker_name)
        for m in never_attacked:
            print(f"Dropping {m!r} for attacker {attacker_name!r}: never attacked for any profile.")
        if not anon_methods:
            raise ValueError(f"No anon_methods have ever been attacked by {attacker_name!r} -- nothing to compute.")

        covered, excluded_ids = _profiles_with_full_coverage(profiles, anon_methods, attacker_name)
        if not covered:
            raise ValueError(
                f"No single profile has guesses for all of {anon_methods} under attacker {attacker_name!r} -- nothing to compute."
            )
        if excluded_ids:
            print(f"Computing reidentification risk for attacker {attacker_name!r} on {len(covered)}/{len(profiles)} "
                  f"profiles (excluding ids missing at least one method's guess: "
                  f"{excluded_ids[:5]}{'...' if len(excluded_ids) > 5 else ''})")
        print(f"Methods included: {anon_methods}")
        correctness_target = covered
    else:
        correctness_target = profiles

    correctness_target = check_guess_correctness(correctness_target, anon_methods, attacker_name=attacker_name)
    write_output(f"{results_path}/level_{level}.jsonl", profiles)
    reid_results_path = f"{uniqueness_results_path}/{scenario}/level_{level}_attacker_{attacker_name}.pickle"
    compute_reid_risk(profiles=correctness_target, methods=anon_methods, attacker=attacker_name,
                      results_path=reid_results_path,
                      dataset=dataset)
    return anon_methods, reid_results_path

def attack(profiles, anon_methods, attacker_name, model_version, scenario,
           results_path, uniqueness_results_path, level, language=None, dataset="PUMS", force_rerun_attack=False,
           interactive=False, key_suffix=None):
    """Run the attacker once over all profiles/methods.

    `attacker_name` selects the attacker implementation (passed to get_attacker).
    `key_suffix` (defaults to `attacker_name`) is what's used to namespace the
    guesses/correctness/results keys and filenames. Repeated runs (see
    `attack_repeated`) pass a distinct `key_suffix` per repeat so each repeat's
    guesses and CorrectMatch results land under their own keys/files instead of
    overwriting each other.
    """
    id_counts = Counter(profile["id"] for profile in profiles)
    if any(count > 1 for count in id_counts.values()):
        dupes = sorted(i for i, count in id_counts.items() if count > 1)
        raise ValueError(
            f"Duplicate profile ids found: {dupes[:10]}{'...' if len(dupes) > 10 else ''}. "
            "attack() keys guesses/prompts by profile id, so profiles sharing an id would "
            "silently overwrite each other's results -- fix the id collision upstream first."
        )

    attacker = get_attacker(attacker_name, model_version)
    suff = key_suffix if key_suffix is not None else attacker_name

    # print(f"Starting attack, language = {language}")
    # print(f"language or English = {language or 'English'}")

    print("Initialized attacker")

    for anon_method in anon_methods:

        print(f"Anon method {anon_method}")
        results_list = list()

        if anon_method=="pre_anon":
            ff = "text"
        else:
            ff = f"text_anon_{anon_method}"

        guess_key = f"guesses_{anon_method}_{suff}"
        prompt_key = f"prompts_{anon_method}_{suff}"

        # Only (re)attack profiles actually missing this method's guess, so a
        # partial/incremental re-attack (e.g. fixing up a handful of corrupted
        # profiles) doesn't burn API calls re-querying everything else.
        to_attack = profiles if force_rerun_attack else [p for p in profiles if guess_key not in p]

        if to_attack:

            # print(f"Running attack for {attacker_name} on {anon_method} anonymization method in language {language}")

            inputs = [
                (
                    profile["id"],
                    profile[ff],
                    attacker,
                    profile["scenario"],
                    profile["features"],
                    language or "English",
                    interactive,
                )
                for profile in to_attack
                # profile_id, text, attacker, scenario, attributes, language, interactive
            ]

            # Each result is written as soon as it arrives, and a profile whose
            # call still fails after the attacker's own retries is left without
            # a guess instead of aborting the method: the finished calls are kept
            # and the next run only re-attacks the missing profiles (see
            # to_attack above).
            by_id = {profile["id"]: profile for profile in to_attack}
            failed_ids = []
            n_workers = 1 if interactive else 10
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(attack_one_profile, inp): inp for inp in inputs}
                for future in tqdm(as_completed(futures), total=len(inputs)):
                    try:
                        profile_id, guess, prompt = future.result()
                    except Exception as e:
                        failed_ids.append(futures[future][0])
                        print(f"[attack] {anon_method}: profile {futures[future][0]} failed, left for the next run: {e}")
                        continue
                    by_id[profile_id][guess_key] = guess
                    by_id[profile_id][prompt_key] = prompt
                    write_output(f"{results_path}/level_{level}.jsonl", profiles)

            if failed_ids:
                raise RuntimeError(
                    f"{len(failed_ids)} profile(s) failed for {anon_method} ({failed_ids[:10]}). "
                    "Results so far are saved; run the same command again to attack only the missing ones."
                )

        if scenario in ["Concert ticket purchase", "Tourist information chatbot", "Topic history", "public info"]:
            print(f"Inferring public info for {scenario}")

            pub_guess_key = f"guesses_public_info_{anon_method}_{suff}"
            pub_correctness_key = f"correctness_public_info_{anon_method}_{suff}"
            pub_prompt_key = f"prompts_public_info_{anon_method}_{suff}"

            to_infer = profiles if force_rerun_attack else [p for p in profiles if pub_guess_key not in p]

            if to_infer:
                inputs = [
                    (
                        profile["id"],
                        profile[ff],
                        attacker,
                        profile["public_info"],
                        profile["scenario"],
                        profile["language"] if "language" in profile else "English"
                    ) for profile in to_infer
                ]
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = {executor.submit(infer_public_info_one_profile, inp): inp for inp in inputs}
                    results_list = []
                    for future in tqdm(as_completed(futures), total=len(inputs)):
                        results_list.append(future.result())
                guesses = {profile_id: guess for profile_id, guess, correctness, prompt in results_list}
                correctness = {profile_id: correctness for profile_id, guess, correctness, prompt in results_list}
                prompts = {profile_id: prompt for profile_id, guess, correctness, prompt in results_list}

                for profile in to_infer:
                    profile[pub_guess_key] = guesses[profile["id"]]
                    profile[pub_correctness_key] = correctness[profile["id"]]
                    profile[pub_prompt_key] = prompts[profile["id"]]
                write_output(f"{results_path}/level_{level}.jsonl", profiles)
        else:
            print(f"Not inferring public info since scenario does not contain public info, it's {scenario}")

    profiles = check_guess_correctness(profiles, anon_methods, attacker_name=suff)

    write_output(f"{results_path}/level_{level}.jsonl", profiles)

    reid_results_path = f"{uniqueness_results_path}/{scenario}/level_{level}_attacker_{suff}.pickle"
    compute_reid_risk(profiles=profiles, methods=anon_methods, attacker=suff,
                      results_path=reid_results_path,
                      dataset=dataset)

    return reid_results_path


def _summarize_repeat_rates(per_repeat_rates, anon_methods, n_repeats, confidence=0.95):
    """Turn per-method lists of per-repeat reidentification rates into
    mean/std/t-interval CI. Shared by `attack_repeated` (rates collected live)
    and `recompute_repeated_summary` (rates loaded back from disk).
    """
    alpha = 1.0 - confidence
    summary = {}
    for m in anon_methods:
        rates = np.array(per_repeat_rates[m], dtype=float)
        n = len(rates)
        mean = float(rates.mean())
        std = float(rates.std(ddof=1)) if n > 1 else float("nan")

        if n > 1:
            se = std / np.sqrt(n)
            t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
            ci_lower = float(mean - t_crit * se)
            ci_upper = float(mean + t_crit * se)
        else:
            ci_lower = ci_upper = float("nan")

        summary[m] = {
            "rates_per_repeat": rates.tolist(),
            "mean": mean,
            "std": std,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_repeats": n_repeats,
            "confidence": confidence,
        }
    return summary


def recompute_repeated_summary(anon_methods, attacker_name, scenario,
                                uniqueness_results_path, level, confidence=0.95):
    """Rebuild the repeated-attack summary purely from per-repeat result
    pickles already on disk -- no attacker calls, no CorrectMatch refitting.

    Discovers every `level_{level}_attacker_{attacker_name}_rep*.pickle` file
    in the results folder (rather than assuming a contiguous 0..n_repeats-1
    range), so it's robust to gaps left by a broken-up/resumed
    `attack_repeated` run, and just re-derives mean/std/CI from whatever
    per-repeat results actually exist.
    """
    result_dir = Path(uniqueness_results_path) / scenario
    pattern = re.compile(rf"^level_{level}_attacker_{re.escape(attacker_name)}_rep(\d+)\.pickle$")

    rep_paths = {}
    for p in result_dir.glob(f"level_{level}_attacker_{attacker_name}_rep*.pickle"):
        match = pattern.match(p.name)
        if match:
            rep_paths[int(match.group(1))] = p

    if not rep_paths:
        raise FileNotFoundError(
            f"No per-repeat result pickles found matching "
            f"'level_{level}_attacker_{attacker_name}_rep*.pickle' in {result_dir}"
        )

    reps = sorted(rep_paths)
    print(f"Found {len(reps)} existing per-repeat result(s) on disk: reps {reps}")

    per_repeat_rates = {m: [] for m in anon_methods}
    for rep in reps:
        with open(rep_paths[rep], "rb") as f:
            rep_results = pickle.load(f)
        for m in anon_methods:
            per_repeat_rates[m].append(rep_results["reidentification_rate"][m])

    summary = _summarize_repeat_rates(per_repeat_rates, anon_methods, n_repeats=len(reps), confidence=confidence)

    summary_path = f"{uniqueness_results_path}/{scenario}/level_{level}_attacker_{attacker_name}_repeated_summary.pickle"
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "wb") as f:
        pickle.dump(summary, f)

    print(f"Recomputed repeated-attack summary from {len(reps)} existing repeat(s) -> {summary_path}")

    return summary


def attack_repeated(profiles, anon_methods, attacker_name, model_version, scenario,
                     results_path, uniqueness_results_path, level, n_repeats,
                     language=None, dataset="PUMS", interactive=False, confidence=0.95, n_start=0):
    """Estimate reidentification-rate uncertainty by actually rerunning the
    attacker `n_repeats` independent times, instead of bootstrap-resampling a
    single run's outputs. This captures real LLM sampling variance (different
    guesses on each call), not just resampling noise over a fixed set of guesses.

    Each repeat gets its own key_suffix (`{attacker_name}_rep{i}`) so guesses,
    correctness fields, and per-repeat CorrectMatch results never overwrite
    each other and can be inspected individually. Since n_repeats is typically
    small (each repeat re-runs the attacker LLM over every profile), the CI
    uses a t-interval on the per-repeat rates rather than a percentile
    bootstrap, which would be unreliable with so few resamples.

    `n_start` resumes a previously interrupted run: repeats `0..n_start-1` are
    assumed already completed (their per-repeat results pickles already saved
    to disk by an earlier call) and are loaded from disk instead of rerun,
    while repeats `n_start..n_repeats-1` are actually attacked. The aggregate
    still covers all `n_repeats` repeats.
    """
    per_repeat_rates = {m: [] for m in anon_methods}

    for rep in range(n_start):
        suff = f"{attacker_name}_rep{rep}"
        cached_path = f"{uniqueness_results_path}/{scenario}/level_{level}_attacker_{suff}.pickle"
        print(f"Repeat {rep + 1}/{n_repeats} is before n_start={n_start}; loading cached result from {cached_path}")

        with open(cached_path, "rb") as f:
            rep_results = pickle.load(f)
        for m in anon_methods:
            per_repeat_rates[m].append(rep_results["reidentification_rate"][m])

    for rep in range(n_start, n_repeats):
        print(f"=== Attacker repeat {rep + 1}/{n_repeats} ===")
        suff = f"{attacker_name}_rep{rep}"

        reid_results_path = attack(
            profiles=profiles, anon_methods=anon_methods, attacker_name=attacker_name,
            model_version=model_version, scenario=scenario, results_path=results_path,
            uniqueness_results_path=uniqueness_results_path, level=level, language=language,
            dataset=dataset, force_rerun_attack=False, interactive=interactive, key_suffix=suff,
        )

        with open(reid_results_path, "rb") as f:
            rep_results = pickle.load(f)
        for m in anon_methods:
            per_repeat_rates[m].append(rep_results["reidentification_rate"][m])

    summary = _summarize_repeat_rates(per_repeat_rates, anon_methods, n_repeats=n_repeats, confidence=confidence)

    summary_path = f"{uniqueness_results_path}/{scenario}/level_{level}_attacker_{attacker_name}_repeated_summary.pickle"
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "wb") as f:
        pickle.dump(summary, f)

    print(f"Saved repeated-attack reidentification rate summary to {summary_path}")

    return summary


def attack_one_profile(args):
    profile_id, text, attacker, scenario, attributes, language, interactive = args
    guess, prompt = attacker.infer(text=text, attributes=attributes, scenario=scenario, language=language,
                                   interactive=interactive)
    return profile_id, guess, prompt

def infer_public_info_one_profile(args):
    profile_id, text, attacker, public_info, scenario, language = args
    guess, correctness, prompt = attacker.infer_public_info(text=text, public_info=public_info, scenario=scenario, language=language)
    return profile_id, guess, correctness, prompt

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--attacker", type=str)
    parser.add_argument("--model_version", type=str)
    parser.add_argument("--anon_methods", type=str)
    parser.add_argument("--scenario", type=str)
    parser.add_argument("--uniqueness_results_folder", type=str)
    parser.add_argument("--only_correctness", type=str, default="False")
    parser.add_argument("--interactive", action="store_true", default=False,
                        help="Prompt for manual input when automatic parsing fails")
    args = parser.parse_args()

    DATA_PATH = args.data_path
    ATTACKER = args.attacker
    MODEL_VERSION = args.model_version
    ANON_METHODS = args.anon_methods
    ANON_METHODS = [s.strip() for s in ANON_METHODS.split(",")]
    SCENARIO = args.scenario
    RESULTS_PATH = args.results_path
    UNIQUENESS_RESULTS_FOLDER = args.uniqueness_results_folder
    ONLY_CORRECTNESS = args.only_correctness
    LEVEL = args.level
    INTERACTIVE = args.interactive

    profiles = []
    with open(f"{DATA_PATH}/level_{LEVEL}.jsonl", "r") as f:
        for line in f:
            profiles.append(json.loads(line))

    print(f"loaded data, {len(profiles)} profiles")

    PATH_TO_SAVE = RESULTS_PATH if RESULTS_PATH is not None else DATA_PATH

    if ONLY_CORRECTNESS=="True":
        print("Only checking correctness, not re-doing attack")
        only_check_correctness(profiles, ANON_METHODS, ATTACKER, SCENARIO, PATH_TO_SAVE, UNIQUENESS_RESULTS_FOLDER, LEVEL)
    else:
        print("Running attack from scratch")
        attack(profiles=profiles, anon_methods=ANON_METHODS, attacker_name=ATTACKER, model_version=MODEL_VERSION,
               scenario=SCENARIO, results_path=PATH_TO_SAVE, uniqueness_results_path=UNIQUENESS_RESULTS_FOLDER,
               level=LEVEL, interactive=INTERACTIVE)