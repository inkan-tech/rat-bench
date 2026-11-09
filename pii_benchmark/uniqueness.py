import argparse
from pathlib import Path
import pickle
import json
from typing import List
import correctmatch
import pandas as pd
from tqdm import tqdm
import numpy as np

PUMS_MAPS_PATH = "./data/maps/{col}_map.pickle"
PUMS_DOB_MAP_PATH = "./data/maps/{col}_map.pickle"

MEX_MAPS_PATH = "./data/es/maps/{col}_map.json"
# Columns whose ground-truth value is already a raw integer (not a categorical label)
MEX_NUMERIC_COLS = {"EDAD", "HIJOS_NAC_VIVOS"}

mex_cols = [
    "CLASE_VIV", "SEXO", "EDAD", "ENT_PAIS_NAC", "DHSERSAL1", "RELIGION",
    "HLENGUA", "HESPANOL", "ASISTEN", "NIVACAD", "SITUA_CONYUGAL", "HIJOS_NAC_VIVOS",
]

MEX_N = 126_014_024  # Mexico 2020 census population

# ---- Serbian (MICS) dataset ----
SRB_N = 6_900_000  # Serbia population

srb_cols = [
    "urban", "age", "marital_status", "given_birth", "ethnicity",
    "language", "dob", "dom", "age_mar", "partner_age",
]

# Columns whose ground-truth is a raw integer (actual value, not a map code)
SRB_NUMERIC_COLS = {"age", "age_mar", "partner_age"}

# Maps column name -> map filename (without .json) for categorical SRB columns
SRB_COL_TO_MAPFILE = {
    "urban":          "urban_status_map",
    "marital_status": "marital_status_map",
    "given_birth":    "ever_given_birth_map",
    "dob":            "dob_map",
    "dom":            "dom_map",
    "ethnicity":      "ethnicity_map",
    "language":       "language_map",
}

# Load Serbian encoded dataset and build per-column 1-indexed encoders for CorrectMatch.
# sample_enc.csv has raw codes; some columns (age, age_mar, partner_age) hold actual values
# that can exceed the number of unique values K, which CorrectMatch requires to be in {1..K}.
SRB_DF_RAW = pd.read_csv("data/srb/data/sample_enc.csv")
SRB_DF_RAW["age_mar"] = pd.to_numeric(SRB_DF_RAW["age_mar"], errors="coerce")

SRB_ENCODERS: dict = {}
_srb_encoded_cols = {}
for _col in srb_cols:
    _vals = sorted(int(v) for v in SRB_DF_RAW[_col].dropna().unique().tolist())
    _enc = {v: idx + 1 for idx, v in enumerate(_vals)}
    SRB_ENCODERS[_col] = _enc
    _srb_encoded_cols[_col] = SRB_DF_RAW[_col].apply(
        lambda x, e=_enc: e[int(x)] if pd.notna(x) else np.nan
    )

SRB_DF_ENCODED = pd.DataFrame(_srb_encoded_cols)

SRB_DF_ENCODED["language"] = SRB_DF_ENCODED["language"].fillna(6).astype(int)
SRB_DF_ENCODED["dom"] = SRB_DF_ENCODED["dom"].fillna(2016).astype(int)
SRB_DF_ENCODED["age_mar"] = SRB_DF_ENCODED["age_mar"].fillna(35).astype(int)
SRB_DF_ENCODED["partner_age"] = SRB_DF_ENCODED["partner_age"].fillna(48).astype(int)

# ---- NL (Belgian/Flemish IPUMS) dataset ----
NL_N = 16_655_799  # Netherlands 2011 census population

nl_cols = [
    "age", "sex", "marstd", "nativity", "bplcountry", "nation",
    "educnl", "empstatd", "labforce", "occisco", "indgen", "dob",
    # "dob-Month", "dob-Year", "dob-Day",
]

# Maps column name -> map file stem (without _map.json)
NL_COL_TO_MAPFILE_UNIQUENESS = {
    "age":        "age",
    "sex":        "sex",
    "marstd":     "marital_status",
    "nativity":   "nativity",
    "bplcountry": "country",
    "nation":     "nation",
    "educnl":     "education",
    "empstatd":   "employment_status",
    "labforce":   "labor_force",
    "occisco":    "occupation",
    "indgen":     "industry",
    "dob":        "dob",
}

# sample_enc.csv encodes all values as {1,...,K} -- no re-encoding needed.
NL_DF = pd.read_csv("data/nl/data/sample_enc.csv", usecols=nl_cols)
NL_DF = NL_DF.dropna().astype(int)


### special maps for occupation (type and description) ###
with open("data/maps/OCCP_type_map.pickle", "rb") as f:
    PUMS_OCCUPATION_TYPES = pickle.load(f)

PUMS_OCCUPATION_TYPE_TO_CODE = dict()

for k, v in PUMS_OCCUPATION_TYPES.items():
    PUMS_OCCUPATION_TYPE_TO_CODE[v] = k

############################################################
    
pums_cols = [
    "AGEP",
    "RAC2P",
    "CIT",
    "SCHL",
    "POWSP",
    "JWDP",
    "OCCP",
    "COW",
    "MAR",
    "MIL",
    "SEX",
    "JWTRNS",
    "ESR",
    "ST",
    "DRIVESP",
    "DOB",
    "DOB-Day",
    "DOB-Month",
    "DOB-Year",
    "zip code"
]

PUMS_DF = pd.read_csv("data/pums/pums_pwgtp_sample_.csv")
N = 309349689 
print("data loaded")

def process_col(c, guess_correctness, ground_truth, cols_to_fit):

    if c not in guess_correctness:
        # if there is no field corresponding to the column in the guesses dict, return None
        print("no field for column ", c)
        return None, cols_to_fit
    
    if guess_correctness[c] is None or len(guess_correctness[c]) == 0:
        # if the correctness is None (i.e. the guess was None), return None
        print("none guess for column ", c)
        return None, cols_to_fit
    
    if guess_correctness[c][0] == 1 and c in pums_cols:
        # if the guess is correct, process it

        # load map
        if "DOB" in c:
            with open(PUMS_DOB_MAP_PATH.format(col=c), "rb") as f:
                map = pickle.load(f)
        elif c == "zip code":
            with open("data/maps/zip_to_puma.pickle", "rb") as f:
                zip_to_puma_map = pickle.load(f)
            with open("data/maps/PUMA_FULL_map.pickle", "rb") as f:
                puma_map = pickle.load(f)
            inverse_map = {v:k for k,v in puma_map.items()}
            zip_code = ground_truth["zip code"]
            puma_full = zip_to_puma_map[int(zip_code)]
            puma = inverse_map[puma_full]
            cols_to_fit.append("PUMA_FULL")
            print(f"{cols_to_fit=}")
            return int(puma), cols_to_fit
        else:
            with open(PUMS_MAPS_PATH.format(col=c), "rb") as f:
                map = pickle.load(f)
        inverse_map = dict()
        for k, v in map.items():
            inverse_map[v] = k

        if c == "OCCP":
            cols_to_fit.append(c)
            dloc = ground_truth[c].find("DESCRIPTION")
            occp_type = ground_truth[c][:dloc].split(":")[-1].strip().strip(",")
            occp_desc = ground_truth[c][dloc:].split(":")[-1].strip()

            if occp_type=="N/A":
                val = "1"
            else:

                f = PUMS_OCCUPATION_TYPE_TO_CODE[occp_type] + "-" + occp_desc
                if f == "OFF-FIRST":
                    val = "25"
                elif f == "SAL-FIRST":
                    val = "32"
                elif f == "RPR-FIRST":
                    val = "117"
                elif f == "SAL-DOOR":
                    val = "158"
                elif f == "FLIGHT ATTENDANTS":
                    val = "260"
                elif (
                    f == "CON-MISCELLANEOUS EXTRACTION WORKERS, INCLUDING ROOF BOLTERS AND HELPERS"
                ):
                    val = "346"
                elif (
                    f == "CON-EARTH DRILLERS, EXCEPT OIL AND GAS"
                ):
                    val = "411"  
                else:
                    val = inverse_map[f]
        elif c == "AGEP":
            cols_to_fit.append(c)
            val = int(ground_truth[c])
        elif (c == "DOB-Day" or c == "DOB-Month" or c == "DOB-Year"):
            if (guess_correctness["DOB"][0]!=1):
                cols_to_fit.append(c)
                date_of_birth = ground_truth["DOB"]
                components = date_of_birth.split(" ")
                if c == "DOB-Day":
                    val = inverse_map[int(components[0])]
                elif c == "DOB-Month":
                    val = inverse_map[components[1]]
                else:
                    val = inverse_map[int(components[2])]
            else:
                return None, cols_to_fit
        elif c == "zip code":
            # handled above
            pass
        else:
            cols_to_fit.append(c)
            val = inverse_map[ground_truth[c]]
        return int(val), cols_to_fit
    else:
        return None, cols_to_fit


def process_col_mex(c, guess_correctness, ground_truth, cols_to_fit):
    """MEX equivalent of process_col: maps a correctly-guessed census attribute to its encoded integer."""
    if c not in guess_correctness:
        print("no field for column ", c)
        return None, cols_to_fit

    if guess_correctness[c] is None or len(guess_correctness[c]) == 0:
        print("none guess for column ", c)
        return None, cols_to_fit

    if guess_correctness[c][0] == 1 and c in mex_cols:
        cols_to_fit.append(c)

        if c in MEX_NUMERIC_COLS:
            try:
                val = int(float(ground_truth[c]))
            except (ValueError, TypeError):
                return None, cols_to_fit
        else:
            with open(MEX_MAPS_PATH.format(col=c), "r", encoding="utf-8") as f:
                raw_map = json.load(f)  # {human_label: encoded_int}
            gt_val = ground_truth[c]
            if gt_val not in raw_map:
                print(f"Ground truth value '{gt_val}' not found in map for column '{c}'")
                return None, cols_to_fit
            val = raw_map[gt_val]

        return int(val), cols_to_fit
    else:
        print(f"Incorrect guess for column {c} or column not in dataset columns, {guess_correctness[c][0]=}, {c in mex_cols=}")
        return None, cols_to_fit


def process_col_srb(c, guess_correctness, ground_truth, cols_to_fit):
    """Map a correctly-guessed Serbian attribute to its re-encoded integer for CorrectMatch."""
    if c not in guess_correctness:
        print("no field for column", c)
        return None, cols_to_fit

    if guess_correctness[c] is None or len(guess_correctness[c]) == 0:
        print("none guess for column", c)
        return None, cols_to_fit

    if guess_correctness[c][0] == 1 and c in srb_cols:
        cols_to_fit.append(c)

        if c in SRB_NUMERIC_COLS:
            try:
                raw_val = int(float(ground_truth[c]))
            except (ValueError, TypeError):
                return None, cols_to_fit
        else:
            mapfile = SRB_COL_TO_MAPFILE[c]
            with open(f"data/srb/maps/{mapfile}.json", "r", encoding="utf-8") as f:
                raw_map = json.load(f)  # {human_label: encoded_int}
            gt_val = ground_truth[c]
            if gt_val not in raw_map:
                print(f"Ground truth value '{gt_val}' not found in map for column '{c}'")
                return None, cols_to_fit
            raw_val = int(raw_map[gt_val])

        encoder = SRB_ENCODERS[c]
        if raw_val not in encoder:
            print(f"Raw value {raw_val} not found in SRB encoder for column '{c}'")
            return None, cols_to_fit
        return encoder[raw_val], cols_to_fit
    else:
        return None, cols_to_fit


def process_col_nl(c, guess_correctness, ground_truth, cols_to_fit):
    """Map a correctly-guessed NL attribute to its encoded integer for CorrectMatch.

    NL sample_enc.csv already stores values as {1,...,K}, so the map file code
    can be used directly without a re-encoder.
    """
    if c not in guess_correctness:
        print("no field for column", c)
        return None, cols_to_fit

    if guess_correctness[c] is None or len(guess_correctness[c]) == 0:
        print("none guess for column", c)
        return None, cols_to_fit

    if guess_correctness[c][0] == 1 and c in nl_cols:
        cols_to_fit.append(c)

        stem = NL_COL_TO_MAPFILE_UNIQUENESS[c]
        with open(f"data/nl/maps/{stem}_map.json", "r", encoding="utf-8") as f:
            raw_map = json.load(f)  # {human_label: encoded_int}

        gt_val = str(ground_truth[c])
        # age_map keys are float strings ("47.0") but ground truth is cleaned to "47"
        if gt_val not in raw_map:
            try:
                gt_val = str(float(gt_val))
            except ValueError:
                pass

        if gt_val not in raw_map:
            print(f"Ground truth value '{ground_truth[c]}' not found in NL map for column '{c}'")
            return None, cols_to_fit

        return int(raw_map[gt_val]), cols_to_fit
    else:
        return None, cols_to_fit


def compute_correctness(model, record_to_analyze: List[int], n: int | None):
    if n is None:
        n = N

    correctness_list = []
    uniqueness_list = []

    for i in range(10):
        correctness = correctmatch.individual_correctness(
            model,
            np.array(record_to_analyze),
            n=n
        )
        iu = correctmatch.individual_uniqueness(
            model,
            np.array(record_to_analyze),
            n=n
        )
        correctness_list.append(correctness)
        uniqueness_list.append(iu)
    correctness = np.mean([c for c in correctness_list if not np.isnan(c)])
    iu = np.mean([u for u in uniqueness_list if not np.isnan(u)])
    
    return correctness, iu

def fit_model_and_compute(cols_to_fit: List[str], record_to_analyze: List[int], df: pd.DataFrame | None, n: int | None):
    if df is None:
        dff = PUMS_DF[cols_to_fit]
    else:
        # dropna handles conditional SRB columns (e.g. dom is NaN for never-married)
        dff = df[cols_to_fit].dropna().astype(int)

    if n is None:
        n = N

    model = correctmatch.fit_model(dff.to_numpy())

    correctness_list = []
    uniqueness_list = []

    for i in range(10):
        
        correctness = correctmatch.individual_correctness(
            model,
            np.array(record_to_analyze),
            n=n
        )
        iu = correctmatch.individual_uniqueness(
            model,
            np.array(record_to_analyze),
            n=n
        )

        correctness_list.append(correctness)
        uniqueness_list.append(iu)

    
    correctness = np.mean([c for c in correctness_list if not np.isnan(c)])
    iu = np.mean([u for u in uniqueness_list if not np.isnan(u)])
    
    return correctness, iu, model


REID_CORRECTNESS_THRESHOLD = 0.2


def _is_reided(n_direct_ids: np.ndarray, correctness: np.ndarray) -> np.ndarray:
    """A profile counts as reidentified if a direct identifier was guessed,
    or if the CorrectMatch correctness score clears the threshold."""
    return (n_direct_ids > 0) | (correctness >= REID_CORRECTNESS_THRESHOLD)


def bootstrap_reidentification_rate(
    correctness_dict,
    correct_direct_ids,
    methods,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
):
    """Bootstrap CI/std for the reidentification rate.

    Resamples profiles with replacement and recomputes the reidentification
    rate from the *already computed* per-profile CorrectMatch correctness
    scores and direct-identifier hits. This does NOT re-run the attacker LLM
    and does NOT refit any CorrectMatch model -- it only resamples the
    per-profile outputs that `compute_reid_risk` already produced, so running
    many bootstrap iterations is cheap.

    Returns a dict keyed by method, each holding mean/std/CI of the
    reidentification rate across bootstrap resamples plus the raw bootstrap
    distribution (useful for plotting/inspection).
    """
    rng = np.random.default_rng(seed)
    alpha = 1.0 - confidence

    bootstrap_results = {}

    for m in methods:
        correctness = np.array(correctness_dict[m], dtype=float)
        n_direct_ids = np.array([len(ids) for ids in correct_direct_ids[m]], dtype=int)
        n_profiles = len(correctness)

        if n_profiles == 0:
            bootstrap_results[m] = {
                "mean": float("nan"),
                "std": float("nan"),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "n_bootstrap": n_bootstrap,
                "confidence": confidence,
                "bootstrap_rates": [],
            }
            continue

        boot_rates = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n_profiles, size=n_profiles)
            boot_rates[b] = _is_reided(n_direct_ids[idx], correctness[idx]).mean()

        bootstrap_results[m] = {
            "mean": float(boot_rates.mean()),
            "std": float(boot_rates.std(ddof=1)),
            "ci_lower": float(np.percentile(boot_rates, 100 * alpha / 2)),
            "ci_upper": float(np.percentile(boot_rates, 100 * (1 - alpha / 2))),
            "n_bootstrap": n_bootstrap,
            "confidence": confidence,
            "bootstrap_rates": boot_rates.tolist(),
        }

    return bootstrap_results


def compute_reid_risk(
    profiles,
    methods,
    attacker,
    results_path,
    dataset="PUMS",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    bootstrap_seed: int | None = None,
):
    if dataset == "MEX":
        df = pd.read_csv("data/es/personas_sample_enc.csv", usecols=mex_cols)
        df = df.dropna().astype(int)
        n = MEX_N
        active_cols = mex_cols
    elif dataset == "SRB":
        df = SRB_DF_ENCODED  # already re-encoded to {1,...,K}; NaN preserved for conditional cols
        n = SRB_N
        active_cols = srb_cols
    elif dataset == "NL":
        df = NL_DF  # already encoded as {1,...,K}
        n = NL_N
        active_cols = nl_cols
    else:
        df = PUMS_DF
        n = 306169200
        active_cols = pums_cols


    n_correct_atts = {k: [] for k in methods}
    correctness_dict = {k: [] for k in methods}
    uniqueness_dict = {k: [] for k in methods}

    correct_direct_ids = {k: [] for k in methods}

    trained_models = {}

    for ii, profile in tqdm(enumerate(profiles)):
        print(f"Analyzing profile {ii}")

        for anon_method in methods:
            print(f"Analyzing anon method {anon_method}")
            cols_to_fit = []
            record_to_analyze = {}

            if attacker != "False":
                crctnss = profile[f"correctness_{anon_method}_{attacker}"]
            else:
                crctnss = profile[f"correctness_{anon_method}"]
            ids_list = []

            srb_date_parts = {"dob-Day", "dob-Month", "dob-Year", "dom-Day", "dom-Month", "dom-Year"}
            nl_date_parts = {"dob-Day", "dob-Month", "dob-Year"}

            for c in crctnss.keys():
                if dataset == "SRB" and c in srb_date_parts:
                    # These are evaluation-only sub-keys derived from dob/dom;
                    # the full date columns are already in srb_cols and handled above.
                    continue
                if dataset == "NL" and c in nl_date_parts:
                    # These are evaluation-only sub-keys derived from dob; the full dob column is already in nl_cols and handled above.
                    continue

                if c in active_cols:
                    # Indirect identifiers
                    if dataset == "MEX":
                        val, cols_to_fit = process_col_mex(
                            c, crctnss, profile["indirect_identifiers"], cols_to_fit
                        )
                    elif dataset == "SRB":
                        val, cols_to_fit = process_col_srb(
                            c, crctnss, profile["indirect_identifiers"], cols_to_fit
                        )
                    elif dataset == "NL":
                        val, cols_to_fit = process_col_nl(
                            c, crctnss, profile["indirect_identifiers"], cols_to_fit
                        )
                    else:
                        val, cols_to_fit = process_col(
                            c, crctnss, profile["indirect_identifiers"], cols_to_fit
                        )
                    if val is not None:
                        if c == "zip code" and dataset not in ("MEX", "NL"):
                            record_to_analyze["PUMA_FULL"] = val
                        else:
                            record_to_analyze[c] = int(val)

                else:
                    # Direct identifiers
                    c_guess = crctnss[c]
                    if c_guess is not None and len(c_guess) > 0:
                        if crctnss[c][0] == 1:
                            ids_list.append(c)

            ## creating record
            record = []
            ordered_cols = []

            # sort columns preserving dataset-appropriate order
            if dataset == "MEX":
                col_order = mex_cols
            elif dataset == "SRB":
                col_order = srb_cols
            elif dataset == "NL":
                col_order = nl_cols
            else:
                col_order = pums_cols
            for c in col_order:
                if c == "zip code" and dataset not in ("MEX", "SRB", "NL"):
                    if "PUMA_FULL" in record_to_analyze:
                        record.append(record_to_analyze["PUMA_FULL"])
                        ordered_cols.append("PUMA_FULL")
                else:
                    if c in record_to_analyze:
                        record.append(record_to_analyze[c])
                        ordered_cols.append(c)

            if len(cols_to_fit) > 0 and len(record) > 0:

                cols_for_model = []

                for oc in ordered_cols:
                    cols_for_model.append(f"{oc}")

                print(f"record = {record}")
                print(f"cols_for_model = {cols_for_model}")

                if "/".join(cols_for_model) in trained_models:
                    model = trained_models["/".join(cols_for_model)]

                    correctness, iu = compute_correctness(model, record, n)
                else:
                    correctness, iu, fitted_model = fit_model_and_compute(
                        cols_for_model, record, df, n
                    )
                    trained_models["/".join(cols_for_model)] = fitted_model
            else:
                print(f"No cols to fit or empty record: {len(cols_to_fit)=}, {len(record)=}")
                correctness, iu = 0.0, 0.0
            n_correct_atts[anon_method].append(len(ordered_cols))
            correctness_dict[anon_method].append(correctness)
            uniqueness_dict[anon_method].append(iu)

            correct_direct_ids[anon_method].append(ids_list)

    results_dict = {
        "correctness": correctness_dict,
        "uniqueness": uniqueness_dict,
        "n_correct_atts": n_correct_atts,
        "correct_direct_ids": correct_direct_ids,
    }

    n_reided_per_method = dict()

    for m in results_dict["correct_direct_ids"].keys():
        correctness_arr = np.array(results_dict["correctness"][m], dtype=float)
        n_direct_ids_arr = np.array(
            [len(ids) for ids in results_dict["correct_direct_ids"][m]], dtype=int
        )
        n_reided_per_method[m] = _is_reided(n_direct_ids_arr, correctness_arr).mean() if len(profiles) else float("nan")

    results_dict["reidentification_rate"] = n_reided_per_method

    results_dict["reidentification_rate_bootstrap"] = bootstrap_reidentification_rate(
        correctness_dict=results_dict["correctness"],
        correct_direct_ids=results_dict["correct_direct_ids"],
        methods=results_dict["correct_direct_ids"].keys(),
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=bootstrap_seed,
    )

    Path(results_path).parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "wb") as f:
        pickle.dump(results_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--results_path", type=str)
    parser.add_argument("--methods", type=str, default=None)
    parser.add_argument("--full_dataset", type=str, default="False")
    parser.add_argument("--attacker", type=str, default="False")
    parser.add_argument("--dataset", type=str, default="PUMS")
    parser.add_argument("--n_bootstrap", type=int, default=1000,
                        help="Number of bootstrap resamples for the reidentification rate CI")
    parser.add_argument("--confidence", type=float, default=0.95,
                        help="Confidence level for the bootstrap CI (e.g. 0.95)")
    parser.add_argument("--bootstrap_seed", type=int, default=None,
                        help="Seed for the bootstrap RNG (for reproducibility)")
    args = parser.parse_args()

    DATA_PATH = args.data_path
    RESULTS_PATH = args.results_path
    METHODS = args.methods
    FULL_DATASET = args.full_dataset
    ATTACKER = args.attacker
    DATASET = args.dataset

    profiles = []
    with open(DATA_PATH, "r") as f:
        for line in f:
            profiles.append(json.loads(line))

    compute_reid_risk(
        profiles, METHODS, ATTACKER, RESULTS_PATH, dataset=DATASET,
        n_bootstrap=args.n_bootstrap, confidence=args.confidence,
        bootstrap_seed=args.bootstrap_seed,
    )
