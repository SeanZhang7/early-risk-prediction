"""Checks for the repeated cross-validation helpers (cv_splits, cv_all, cv_tfidf, cv_significance) on synthetic data.

Run: conda run -n hongyi python tests/test_cv.py
"""

import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv_all
import cv_significance as cs
import cv_splits
import cv_tfidf
import error_analysis as ea
import experiment as ex
from feature_cols import EXPERIMENTS_V2
from paths import Paths


def _labels(n=120, pos=14):
    y = np.zeros(n, dtype=int)
    y[:pos] = 1
    return pd.Series(y, index=[f"u{i:03d}" for i in range(n)]).sample(frac=1, random_state=0).sort_index()


def test_outer_splits_partition_stratify_and_extend():
    labels = _labels()
    s3 = cv_splits.outer_splits(labels, 3)
    s2 = cv_splits.outer_splits(labels, 2)
    assert len(s3) == 15
    for r in range(3):
        tests = np.concatenate([s3[r * 5 + k][1] for k in range(5)])
        assert sorted(tests) == sorted(labels.index), "every user is held out exactly once per repeat"
        for k in range(5):
            tr, te = s3[r * 5 + k]
            assert set(tr).isdisjoint(te) and len(tr) + len(te) == len(labels)
            assert 2 <= labels.loc[te].sum() <= 3, "positives are spread over the folds"
    for a, b in zip(s2, s3[:10]):
        assert np.array_equal(a[1], b[1]), "more repeats must not change the earlier ones"


def test_no_default_configuration_uses_re():
    for name in cv_all.DEFAULT_CONFIGS:
        assert not (cv_all.FORBIDDEN & set(EXPERIMENTS_V2[name])), name


def test_holm_and_nadeau_bengio():
    adj = cs.holm([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06])
    t, p = cs.nadeau_bengio([0.1, 0.2, 0.15, 0.05, 0.12], n_test=1, n_train=4)
    d = np.array([0.1, 0.2, 0.15, 0.05, 0.12])
    expected_t = d.mean() / np.sqrt((1 / 5 + 0.25) * d.var(ddof=1))
    assert abs(t - expected_t) < 1e-12 and 0 < p < 1
    assert cs.nadeau_bengio([0.1, 0.1, 0.1], 1, 4) == (0.0, 1.0)  # no variance


def test_delong_matches_sklearn_and_detects_differences():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(60), np.zeros(300)].astype(int)
    good = rng.normal(size=360) + 1.5 * y
    noise = rng.normal(size=360)
    a1, a2, p = cs.delong_p(y, good, noise)
    assert abs(a1 - roc_auc_score(y, good)) < 1e-12 and abs(a2 - roc_auc_score(y, noise)) < 1e-12
    assert p < 0.001
    assert cs.delong_p(y, good, good)[2] == 1.0
    _, _, p_same = cs.delong_p(y, good, good + rng.normal(scale=0.01, size=360))
    assert p_same > 0.05


def test_mcnemar_exact():
    a = np.array([1, 1, 1, 1, 1, 1, 0, 0], bool)
    b = np.array([0, 0, 0, 0, 0, 0, 0, 0], bool)
    assert cs.mcnemar_p(a, a) == 1.0
    assert abs(cs.mcnemar_p(a, b) - 2 * 0.5 ** 6) < 1e-12  # 6 discordant pairs, all in one direction


def _synthetic_frame(n_users=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        y = int(u % 4 == 0)
        n = int(rng.integers(5, 12))
        for t in range(n):
            rows.append({"user_id": f"u{u:03d}", "Depression": y, "post_index": float(t),
                         "a": rng.normal() + 1.0 * y, "b": rng.normal()})
    return pd.DataFrame(rows)


def test_run_outer_returns_every_test_user_once_with_a_rule():
    frame = _synthetic_frame()
    labels = frame.groupby("user_id")["Depression"].max()
    users = labels.index.to_numpy()
    tr, te = users[:30], users[30:]
    cfg = ex.Config(n_folds=3, epochs=2, patience=1, hidden_dim=16, n_heads=2, n_layers=1, dropout=0.1)
    res = cv_all.run_outer(frame, ["post_index", "a", "b"], tr, te, seed=3, device="cpu", cfg=cfg)
    assert list(res["users"]) == list(te) and len(res["probs"]) == len(te)
    for p, u in zip(res["probs"], te):
        assert len(p) == (frame["user_id"] == u).sum() and ((0 <= p) & (p <= 1)).all()
    assert res["th"] in cfg.thresholds and res["k"] in cfg.ks


def test_analyse_on_fake_results_pairs_configs_on_common_repeats():
    rng = np.random.default_rng(1)
    users = [f"u{i:03d}" for i in range(80)]
    y = np.array([int(i % 4 == 0) for i in range(80)])
    with tempfile.TemporaryDirectory() as tmp:
        P = Paths(root=tmp)
        for name, strength in (("lean2_without_RE", 2.0), ("only_PHQ", 0.3)):
            for r in range(2):
                if name == "only_PHQ" and r == 1:
                    continue  # this configuration has only one finished repeat
                for k in range(5):
                    idx = [i for i in range(80) if i % 5 == k]
                    probs = [1 / (1 + np.exp(-(strength * y[i] - 1 + rng.normal(size=int(rng.integers(3, 9))))))
                             for i in idx]
                    path = cv_splits.result_path(P, name, r, k)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pickle.dump({"users": np.array([users[i] for i in idx]), "labels": y[idx], "probs": probs,
                                 "th": 0.5, "k": 1, "oof_f_latency": 0.0, "inner_epochs": [1]}, open(path, "wb"))
        summary, pairs, n_pos, n_users = cs.analyse(P, ["lean2_without_RE", "only_PHQ"], repeats=2, n_boot=50)
    assert n_users == 80 and n_pos == 20
    assert set(summary["config"]) == {"lean2_without_RE", "only_PHQ"}
    row = pairs[(pairs.A == "lean2_without_RE") & (pairs.B == "only_PHQ")].iloc[0]
    assert row["repeats"] == 1, "only the repeat both configurations finished is compared"
    assert row["d_roc"] > 0 and row["d_roc_lo"] < row["d_roc_hi"]
    assert 0 < row["p_roc"] <= 1 and 0 < row["p_roc_holm"] <= 1


def test_tfidf_helpers():
    uc = np.array([0, 0, 0, 0, 1, 2, 2])
    y = np.array([1, 1, 1, 1, 0, 0, 0])
    w = cv_tfidf.user_weights(uc, y)
    assert abs(w.mean() - 1) < 1e-9
    per_user = [w[uc == u].sum() for u in range(3)]
    assert abs(per_user[0] / per_user[1] - 2) < 1e-9 and abs(per_user[1] - per_user[2]) < 1e-9  # 1 positive : 2 negative users
    logits = np.array([2.0, 0.0, -2.0, 1.0, 0.5, -1.0, 1.0])
    probs = cv_tfidf.running_mean_logit(logits, uc, 3)
    assert np.allclose(probs[0], 1 / (1 + np.exp(-np.array([2.0, 1.0, 0.0, 0.25]))))
    assert len(probs[1]) == 1 and len(probs[2]) == 2


def test_baseline_pairs_do_not_change_family_one_holm():
    assert set(cs.PAIRS) & set(cs.PAIRS_BASELINE) == set()
    assert all("tfidf_lr" in pair for pair in cs.PAIRS_BASELINE)


def test_error_majority_and_rule_baseline():
    runs = pd.DataFrame({"user": ["a"] * 3 + ["b"] * 3 + ["c"] * 3, "y": [1] * 3 + [0] * 3 + [1] * 3,
                         "delay": [5, 0, 0, 0, 7, 9, 3, 4, 0], "peak": [0.6, 0.4, 0.3, 0.2, 0.7, 0.8, 0.9, 0.9, 0.4],
                         "th": [0.5] * 9})
    m = ea.majority(runs)
    assert m.loc["a", "outcome"] == "FN" and m.loc["b", "outcome"] == "FP" and m.loc["c", "outcome"] == "TP"
    assert m.loc["a", "n_wrong"] == 2 and m.loc["c", "n_wrong"] == 1
    posts = pd.DataFrame({"user_id": ["a", "a", "a", "b", "b", "c"], "self_report": [0, 1, 1, 0, 0, 0]})
    labels = pd.Series([1, 0, 1], index=["a", "b", "c"])
    r = ea.rule_baseline(posts, labels).iloc[0]
    assert r["tp"] == 1 and r["fp"] == 0 and r["fn"] == 1 and r["median_delay"] == 2


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
