"""Time-ordered "so-far" features shared by every offline stage.

Input rows hold post-level features only; this module sorts each user's posts by time and derives
all history columns, so train/val/test and the baselines cannot disagree on their semantics.

Rules
- Features at row t depend only on rows <= t of the same user (see tests/test_features_history.py).
- PHQ and RE history run over every post (submissions and comments).
- sim and the 10 LSM features need the parent text, so they exist only for rows with `has_parent` == 1
  (comments whose parent text is known) and their history runs over those rows only. Every other row
  (submissions, comments without parent text) gets sim/LSM = 0, delta = 0, and the other history values
  carried forward from the user's last row with a parent (0 before the first one).
  `is_comment` and `has_parent` tell the model which rows carry real values.
- Every `<feature>_cummean` is the mean of that feature over the user's rows up to and including this one
  (over rows with a parent for sim and LSM, carried forward like the other history values).
- The optional LSMX columns (feature_cols.LSMX_COLS) follow the same parent-only rule but keep only the raw value
  and its running mean.
"""

import numpy as np
import pandas as pd

from feature_cols import (PHQ_COLS, LSM_BASE_FEATURES, FEATURE_COLS_V2, FEATURE_COLS_V3, LSMX_COLS, LSMX_CUMMEAN_COLS,
                          OWN_LIWC_COLS, BEHAVIOR_COLS)

WINDOW = 5
_SORT_TIE_COLS = ["submission_id", "comment_id"]
_INPUT_COLS = set(PHQ_COLS + ["loss_dep0", "loss_dep1", "re", "sim", "has_parent", "log_tokens"] + LSM_BASE_FEATURES
                  + OWN_LIWC_COLS)
_DERIVED_COLS = [c for c in FEATURE_COLS_V2 if c not in _INPUT_COLS] + LSMX_CUMMEAN_COLS


def sort_posts(df):
    """Chronological order within each user; ties put submissions first, then order by id. Needs datetime created_utc."""
    keys = ["user_id", "created_utc", "_is_comment"] + [c for c in _SORT_TIE_COLS if c in df.columns]
    out = df.assign(_is_comment=(df["type"] == "comment").astype(int)).sort_values(keys, kind="mergesort")
    return out.drop(columns="_is_comment").reset_index(drop=True)


def _history(df, col):
    """(max, min, gap, delta, rolling_std) of `col` within each user; df must already be time-sorted."""
    g = df.groupby("user_id", sort=False)[col]
    mx, mn = g.cummax(), g.cummin()
    delta = g.diff().fillna(0.0)
    std = g.rolling(WINDOW, min_periods=1).std().reset_index(level=0, drop=True).fillna(0.0)
    return mx, mn, mx - mn, delta, std


def _names(prefix_style, col):
    """Output column names for (max, min, gap, delta, rolling_std), matching feature_cols.FEATURE_COLS."""
    if prefix_style == "short":  # e.g. sim -> max_sim_so_far
        return (f"max_{col}_so_far", f"min_{col}_so_far", f"max_gap_{col}_so_far",
                f"delta_{col}", f"rolling_std_{col}")
    return tuple(f"{col}_{s}" for s in ("max_so_far", "min_so_far", "max_gap_so_far", "delta", "rolling_std"))


def build_features(raw):
    """Return a time-sorted copy of `raw` with every FEATURE_COLS_V2 column filled in."""
    post_cols = PHQ_COLS + ["loss_dep0", "loss_dep1", "re"] + OWN_LIWC_COLS + ["log_tokens"]
    comment_cols = ["sim"] + LSM_BASE_FEATURES
    missing = [c for c in ["user_id", "created_utc", "type", "has_parent"] + post_cols + comment_cols if c not in raw.columns]
    if missing:
        raise ValueError(f"missing input columns: {missing}")
    unknown_types = set(raw["type"].unique()) - {"comment", "submission"}
    if unknown_types:
        raise ValueError(f"unexpected type values: {unknown_types}")
    if (raw["has_parent"].astype(bool) & (raw["type"] != "comment")).any():
        raise ValueError("has_parent is only valid for comments")

    df = raw.drop(columns=[c for c in _DERIVED_COLS if c in raw.columns])  # recompute rather than duplicate
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
    df["has_parent"] = df["has_parent"].astype(bool).astype(float)
    df = sort_posts(df)
    df["is_comment"] = (df["type"] == "comment").astype(float)
    df["night"] = df["created_utc"].dt.hour.isin(range(6)).astype(float)  # UTC hours 0-5
    gap = df.groupby("user_id", sort=False)["created_utc"].diff().dt.total_seconds().fillna(0.0)
    df["log_gap"] = np.log1p(gap.clip(lower=0.0))
    has_parent = df["has_parent"] == 1

    bad = [c for c in post_cols if df[c].isna().any()]
    bad += [c for c in comment_cols if df.loc[has_parent, c].isna().any()]
    lsmx = all(c in df.columns for c in LSMX_COLS)  # optional extra parent-only columns
    if lsmx:
        bad += [c for c in LSMX_COLS if df.loc[has_parent, c].isna().any()]
    if bad:
        raise ValueError(f"NaN in required columns: {bad}")

    new = {}
    new["phq_score"] = df[PHQ_COLS].mean(axis=1)
    new["post_index"] = df.groupby("user_id", sort=False).cumcount().astype(float)
    tmp = df.assign(phq_score=new["phq_score"])
    for col, names in (("phq_score", ("max_phq_so_far", "min_phq_so_far", "max_gap", "delta_phq", "rolling_std_phq")),
                       ("re", _names("short", "re"))):
        for name, series in zip(names, _history(tmp, col)):
            new[name] = series
    for i, col in enumerate(PHQ_COLS):
        new[f"max_{i}_so_far"] = df.groupby("user_id", sort=False)[col].cummax()

    count = df.groupby("user_id", sort=False).cumcount() + 1
    post_values = {"phq_score": new["phq_score"], "re": df["re"], **{c: df[c] for c in OWN_LIWC_COLS + BEHAVIOR_COLS}}
    for col, values in post_values.items():
        new[f"{col}_cummean"] = values.groupby(df["user_id"], sort=False).cumsum() / count

    comments = df.loc[has_parent, ["user_id"] + comment_cols]
    for col in comment_cols:
        names = _names("short", col) if col == "sim" else _names("long", col)
        for name, series in zip(names, _history(comments, col)):
            full = series.reindex(df.index)
            if name.endswith("delta") or name.startswith("delta_"):
                full = full.fillna(0.0)
            else:
                full = full.groupby(df["user_id"], sort=False).ffill().fillna(0.0)
            new[name] = full
        by_user = comments.groupby("user_id", sort=False)[col]
        run_mean = by_user.cumsum() / (by_user.cumcount() + 1)
        new[f"{col}_cummean"] = run_mean.reindex(df.index).groupby(df["user_id"], sort=False).ffill().fillna(0.0)
        new[col] = df[col].where(has_parent, 0.0)

    if lsmx:
        parent_rows = df.loc[has_parent, ["user_id"] + LSMX_COLS]
        for col in LSMX_COLS:
            by_user = parent_rows.groupby("user_id", sort=False)[col]
            run_mean = by_user.cumsum() / (by_user.cumcount() + 1)
            new[f"{col}_cummean"] = run_mean.reindex(df.index).groupby(df["user_id"], sort=False).ffill().fillna(0.0)
            new[col] = df[col].where(has_parent, 0.0)

    out = pd.concat([df.drop(columns=comment_cols + (LSMX_COLS if lsmx else [])), pd.DataFrame(new, index=df.index)],
                    axis=1)
    absent = [c for c in (FEATURE_COLS_V3 if lsmx else FEATURE_COLS_V2) if c not in out.columns]
    assert not absent, f"features not produced: {absent}"
    return out
