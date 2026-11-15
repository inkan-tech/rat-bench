"""CorrectMatch reads a record value as a category number 1..k, so columns
whose codes have gaps (PUMS DOB) must be rank-encoded, data and record alike.
Without it a level 2 record with a correctly guessed full DOB crashed with
MethodError: no method matching pdf(::Categorical, ::Nothing)."""
import numpy as np
import pandas as pd
import pytest

U = pytest.importorskip("pii_benchmark.uniqueness")


def _data(gapped):
    rng = np.random.default_rng(0)
    a = rng.integers(1, 4, 2000)
    b = rng.choice([1, 2, 3, 4], 2000)
    df = pd.DataFrame({"A": a, "B": b})
    if gapped:
        df["B"] = df["B"].map({1: 10, 2: 20, 3: 30, 4: 400})
    return df


def test_rank_encode_is_identity_on_contiguous_columns():
    df = _data(gapped=False)
    encoded, _ = U._rank_encode(df)
    assert (encoded == df.to_numpy()).all()


def test_record_value_absent_from_data_is_refused():
    _, maps = U._rank_encode(_data(gapped=True))
    with pytest.raises(ValueError):
        U._encode_record([1, 25], maps)


def test_gapped_column_scores_like_its_contiguous_twin():
    c_gap, _, model = U.fit_model_and_compute(["A", "B"], [2, 400], _data(gapped=True), 10_000)
    c_ref, _, _ = U.fit_model_and_compute(["A", "B"], [2, 4], _data(gapped=False), 10_000)
    assert c_gap == pytest.approx(c_ref)
    c_cached, _ = U.compute_correctness(model, [2, 400], 10_000)
    assert 0.0 <= c_cached <= 1.0
