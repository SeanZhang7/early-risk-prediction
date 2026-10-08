"""Checks for features_history.build_features. Run: python tests/test_features_history.py (or pytest)."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_cols import (FEATURE_COLS_V2, FEATURE_COLS_V3, LSM_BASE_FEATURES, LSMX_COLS, OWN_LIWC_COLS,
                          PHQ_COLS)
from features_history import WINDOW, build_features

PARENT_ONLY = ["sim"] + LSM_BASE_FEATURES
SIM_LSM_FEATURES = [c for c in FEATURE_COLS_V2 if "PHQ" not in c and any(c == b or b in c for b in PARENT_ONLY)] + \
                   ["max_sim_so_far", "min_sim_so_far", "max_gap_sim_so_far", "delta_sim", "rolling_std_sim"]
SIM_LSM_FEATURES = sorted(set(SIM_LSM_FEATURES))
CARRIED = ("max_sim_so_far", "min_sim_so_far", "max_gap_sim_so_far", "rolling_std_sim",
           "LSM_mean_max_so_far", "LSM_mean_min_so_far", "LSM_mean_rolling_std")


def make_df(seed=0, n_users=6, max_posts=40):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        n = int(rng.integers(8, max_posts))
        secs = np.sort(rng.choice(10**6, size=n, replace=False))
        times = pd.Timestamp("2020-01-01", tz="UTC") + pd.to_timedelta(secs, unit="s")
        types = rng.choice(["comment", "submission"], size=n, p=[0.7, 0.3])
        for t, ty in zip(times, types):
            has_parent = bool(ty == "comment" and rng.uniform() < 0.75)  # some comments have no parent text
            r = {"user_id": f"u{u}", "created_utc": t, "type": ty, "Depression": u % 2, "has_parent": has_parent,
                 "comment_id": f"c{u}_{int(t.timestamp())}", "submission_id": None}
            for c in PHQ_COLS:
                r[c] = rng.normal()
            for c in PARENT_ONLY:
                r[c] = rng.uniform(0, 1) if has_parent else np.nan
            for c in OWN_LIWC_COLS:
                r[c] = rng.uniform(0, 0.2)
            r["log_tokens"] = rng.uniform(0, 6)
            r["loss_dep0"], r["loss_dep1"] = rng.uniform(0, 1, 2)
            r["re"] = r["loss_dep1"] - r["loss_dep0"]
            rows.append(r)
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


def _assert_features_equal(a, b, cols=FEATURE_COLS_V2):
    pd.testing.assert_frame_equal(a[cols].reset_index(drop=True), b[cols].reset_index(drop=True),
                                  check_exact=False, rtol=1e-9, atol=1e-12)


def test_output_has_every_v2_column_and_does_not_mutate_input():
    df = make_df()
    before = df.copy()
    out = build_features(df)
    assert set(FEATURE_COLS_V2) <= set(out.columns)
    pd.testing.assert_frame_equal(df, before)


def test_input_order_does_not_matter():
    df = make_df()
    shuffled = df.sample(frac=1, random_state=123)
    _assert_features_equal(build_features(df), build_features(shuffled))


def test_prefix_invariance():
    df = make_df(seed=1)
    full = build_features(df)
    for uid, g in full.groupby("user_id"):
        for t in (2, len(g) // 2, len(g) - 2):
            cut = g["created_utc"].iloc[t]
            prefix_in = df[(df["user_id"] == uid) & (pd.to_datetime(df["created_utc"], utc=True) <= cut)]
            prefix_out = build_features(prefix_in)
            _assert_features_equal(prefix_out, g.iloc[: t + 1])


def test_so_far_columns_are_monotone():
    out = build_features(make_df(seed=2))
    # Before a user's first row with a parent the sim/LSM history is a 0 placeholder, so only check those rows there.
    for _, g_all in out.groupby("user_id"):
        assert (g_all["post_index"].to_numpy() == np.arange(len(g_all))).all()
        g_parent = g_all[g_all["has_parent"] == 1]
        for c in FEATURE_COLS_V2:
            g = g_parent if c in SIM_LSM_FEATURES else g_all
            step = g[c].diff().fillna(0)
            if c.startswith("max_") and c.endswith("_so_far") and "gap" not in c or c.endswith("_max_so_far"):
                assert (step >= -1e-12).all(), c
            elif c.startswith("min_") and c.endswith("_so_far") or c.endswith("_min_so_far"):
                assert (step <= 1e-12).all(), c
            elif "max_gap" in c:
                assert (step >= -1e-12).all(), c


def test_rows_without_parent_carry_forward_history():
    out = build_features(make_df(seed=3))
    assert ((out["has_parent"] == 0) & (out["is_comment"] == 1)).any(), "fixture needs parentless comments"
    for _, g in out.groupby("user_id"):
        seen_parent = False
        last = None
        for _, row in g.iterrows():
            if row["has_parent"] == 1:
                seen_parent, last = True, row
                continue
            assert row["sim"] == 0 and all(row[c] == 0 for c in LSM_BASE_FEATURES)
            assert row["delta_sim"] == 0
            for c in CARRIED:
                assert row[c] == (last[c] if seen_parent else 0.0), c


def test_sim_and_lsm_history_ignore_rows_without_parent():
    df = make_df(seed=4)
    with_all = build_features(df)
    only_parent = build_features(df[df["has_parent"]])
    a = with_all[with_all["has_parent"] == 1]
    _assert_features_equal(a, only_parent, cols=SIM_LSM_FEATURES)


def test_rolling_std_matches_numpy_on_rows_with_parent():
    out = build_features(make_df(seed=5))
    for _, g in out[out["has_parent"] == 1].groupby("user_id"):
        vals = g["sim"].to_numpy()
        for t in range(len(vals)):
            tail = vals[max(0, t - WINDOW + 1): t + 1]
            expected = np.std(tail, ddof=1) if len(tail) > 1 else 0.0
            assert abs(g["rolling_std_sim"].iloc[t] - expected) < 1e-9


def test_nan_in_required_columns_raises():
    df = make_df()
    df.loc[df["has_parent"], "sim"] = np.nan
    try:
        build_features(df)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_running_means_and_night_flag():
    out = build_features(make_df(seed=7))
    hours = out["created_utc"].dt.hour
    assert ((hours < 6).astype(float) == out["night"]).all()
    for _, g in out.groupby("user_id"):
        for col in ["phq_score", "re", "liwc_I", "liwc_SAD", "log_tokens", "night"]:
            np.testing.assert_allclose(g[f"{col}_cummean"], g[col].expanding().mean(), atol=1e-12)
        seen = g[g["has_parent"] == 1]
        for col in ["sim", "LSM_mean", "negative_mean"]:
            expected = seen[col].expanding().mean().reindex(g.index).ffill().fillna(0.0)
            np.testing.assert_allclose(g[f"{col}_cummean"], expected, atol=1e-12)


def test_lsmx_is_parent_only_with_running_mean_and_prefix_invariant():
    df = make_df(seed=9)
    rng = np.random.default_rng(9)
    for c in LSMX_COLS:
        df[c] = np.where(df["has_parent"], rng.normal(size=len(df)), np.nan)
    out = build_features(df)
    assert set(FEATURE_COLS_V3) <= set(out.columns)
    assert (out.loc[out["has_parent"] == 0, LSMX_COLS] == 0).all().all()
    for _, g in out.groupby("user_id"):
        seen = g[g["has_parent"] == 1]
        for col in ("lsmr_LSM_mean", "par_log_tokens"):
            expected = seen[col].expanding().mean().reindex(g.index).ffill().fillna(0.0)
            np.testing.assert_allclose(g[f"{col}_cummean"], expected, atol=1e-12)
    _assert_features_equal(out, build_features(out), cols=FEATURE_COLS_V3)  # re-feeding is a no-op
    for uid, g in out.groupby("user_id"):
        cut = len(g) // 2
        head = df[df["user_id"] == uid].sort_values("created_utc").iloc[:cut]
        part = build_features(head)
        _assert_features_equal(part, g.iloc[:cut], cols=FEATURE_COLS_V3)


def test_log_gap_is_time_since_previous_post():
    out = build_features(make_df(seed=8))
    for _, g in out.groupby("user_id"):
        secs = g["created_utc"].diff().dt.total_seconds().fillna(0.0)
        np.testing.assert_allclose(g["log_gap"], np.log1p(secs), atol=1e-12)
        assert g["log_gap"].iloc[0] == 0.0


def test_rebuilding_from_its_own_output_is_a_no_op():
    out = build_features(make_df(seed=6))
    again = build_features(out)
    assert not again.columns.duplicated().any()
    _assert_features_equal(out, again)


def test_has_parent_on_a_submission_raises():
    df = make_df()
    df.loc[df["type"] == "submission", "has_parent"] = True
    try:
        build_features(df)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
