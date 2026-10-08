"""Checks for seq_model, seq_eval and experiment on synthetic data (CPU, a few seconds).

Run: conda run -n hongyi python tests/test_sequence.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import experiment as ex
import seq_eval as se
import seq_model as sm
from feature_cols import EXPERIMENTS_V2, FEATURE_COLS_V2, FEATURE_GROUPS_V2
from paths import Paths


def make_users(n_users=40, seed=0, n_pos=12):
    """Time-ordered posts; depressed users have a higher f0 on average, so the label is learnable."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        y = int(u < n_pos)
        n = int(rng.integers(6, 14))
        for t in range(n):
            rows.append({"user_id": f"u{u:03d}", "created_utc": pd.Timestamp("2021-01-01", tz="UTC") + pd.Timedelta(hours=t),
                         "type": "comment", "comment_id": f"c{u}_{t}", "submission_id": f"s{u}", "Depression": y,
                         "f0": rng.normal(1.5 * y, 1.0), "f1": rng.normal(), "post_index": float(t)})
    return pd.DataFrame(rows)


def test_standardizer_fit_log1p_and_roundtrip():
    df = make_users()
    s = sm.Standardizer.fit(df, ["f0", "f1", "post_index"])
    x = s.transform(df)
    assert np.allclose(x.mean(0), 0, atol=1e-5) and np.allclose(x.std(0), 1, atol=1e-5)
    raw = np.log1p(df["post_index"].to_numpy())
    assert np.isclose(s.mean[2], raw.mean())
    path = Path(tempfile.mkdtemp()) / "scaler.json"
    s.save(path)
    np.testing.assert_allclose(sm.Standardizer.load(path).transform(df), x)
    const = df.assign(c=3.0)
    assert np.isfinite(sm.Standardizer.fit(const, ["c"]).transform(const)).all()


def test_standardizer_clips_extreme_values():
    df = make_users()
    s = sm.Standardizer.fit(df, ["f0"], clip=3.0)
    spiked = df.assign(f0=df["f0"].where(df.index != 0, 1e6))
    z = s.transform(spiked)
    assert z.max() <= 3.0 + 1e-6 and z.min() >= -3.0 - 1e-6 and z[0, 0] == np.float32(3.0)


def test_make_sequences_requires_contiguous_users():
    df = make_users(5)
    x = df[["f0"]].to_numpy(np.float32)
    users, seqs, y = sm.make_sequences(df, x)
    assert list(users) == sorted(set(df["user_id"])) and sum(len(s) for s in seqs) == len(df)
    assert (y == [1, 1, 1, 1, 1]).all() or y.dtype == int
    try:
        sm.make_sequences(df.sample(frac=1, random_state=0), x)
    except ValueError:
        return
    raise AssertionError("expected ValueError for unsorted rows")


def test_batches_cover_everyone_once_and_respect_budget():
    lengths = np.random.default_rng(0).integers(1, 400, 300)
    for rng in (None, np.random.default_rng(1)):
        batches = sm.make_batches(lengths, rng, max_users=8, attn_budget=200_000)
        flat = sorted(i for b in batches for i in b)
        assert flat == list(range(len(lengths)))
        for b in batches:
            assert len(b) <= 8
            assert len(b) == 1 or len(b) * lengths[b].max() ** 2 <= 200_000


def _model(dim=4):
    torch.manual_seed(0)
    return sm.UserSequenceModel(dim, hidden_dim=16, n_heads=2, n_layers=2, dropout=0.0).eval()


def test_model_is_causal():
    m = _model()
    x = torch.randn(2, 9, 4)
    mask = torch.ones(2, 9, dtype=torch.bool)
    base = m(x, mask)
    x2 = x.clone()
    x2[:, 5:] = torch.randn(2, 4, 4)
    out = m(x2, mask)
    assert torch.allclose(base[:, :5], out[:, :5], atol=1e-5)
    assert not torch.allclose(base[:, 5:], out[:, 5:], atol=1e-5)


def test_padding_does_not_change_a_users_probabilities():
    m = _model()
    short = np.random.default_rng(0).normal(size=(5, 4)).astype(np.float32)
    long = np.random.default_rng(1).normal(size=(9, 4)).astype(np.float32)
    alone = sm.predict(m, [short], "cpu")[0]
    batched = sm.predict(m, [short, long], "cpu")
    np.testing.assert_allclose(alone, batched[0], atol=1e-5)
    assert len(batched[1]) == 9


def test_sequence_loss_matches_manual_computation():
    logits = torch.tensor([[0.3, -1.0, 5.0], [0.1, 0.2, 0.3]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    labels = torch.tensor([1.0, 0.0])
    pw = torch.tensor(3.0)
    got = sm.sequence_loss(logits, mask, labels, pw).item()
    p = torch.sigmoid(logits)
    u0 = -(3.0 * torch.log(p[0, :2])).mean()
    u1 = -(torch.log(1 - p[1])).mean()
    assert abs(got - ((u0 + u1) / 2).item()) < 1e-6


def test_fit_learns_a_separable_signal():
    df = make_users(60, n_pos=18)
    s = sm.Standardizer.fit(df, ["f0", "f1"])
    users, seqs, y = sm.make_sequences(df, s.transform(df))
    tr, va = np.arange(0, 48), np.arange(48, 60)
    model, hist = sm.fit([seqs[i] for i in tr], y[tr], [seqs[i] for i in va], y[va], "cpu", epochs=25, lr=3e-3,
                         patience=25, seed=0, verbose=False, hidden_dim=16, n_heads=2, n_layers=1, dropout=0.0)
    probs = sm.predict(model, seqs, "cpu")
    last = np.array([p[-1] for p in probs])
    assert last[y == 1].mean() > last[y == 0].mean() + 0.2
    assert model.best_epoch >= 1 and len(hist) <= 25


def test_best_weights_are_restored_not_the_last_ones():
    df = make_users(40, n_pos=12)
    s = sm.Standardizer.fit(df, ["f0", "f1"])
    users, seqs, y = sm.make_sequences(df, s.transform(df))
    tr, va = np.arange(0, 30), np.arange(30, 40)
    kw = dict(epochs=12, lr=1e-2, patience=100, seed=0, verbose=False, hidden_dim=16, n_heads=2, n_layers=1, dropout=0.0)
    model, hist = sm.fit([seqs[i] for i in tr], y[tr], [seqs[i] for i in va], y[va], "cpu", **kw)
    pw = torch.tensor((len(tr) - y[tr].sum()) / y[tr].sum(), dtype=torch.float32)
    now = sm.mean_loss(model, [seqs[i] for i in va], y[va], pw, "cpu")
    assert abs(now - min(h["val_loss"] for h in hist)) < 1e-5


def test_run_lengths_and_alarm_delay():
    ok = np.array([1, 1, 0, 1, 1, 1], dtype=bool)
    assert list(se.run_lengths(ok)) == [1, 2, 0, 1, 2, 3]
    probs = [np.array([0.9, 0.9, 0.1, 0.9, 0.9, 0.9]), np.array([0.1, 0.2]), np.array([0.8])]
    d = se.delays(probs, 0.5, [1, 2, 3])
    assert d.tolist() == [[1, 2, 6], [0, 0, 0], [1, 0, 0]]


def test_metrics_and_latency_weighting():
    d = np.array([5, 0, 20, 3, 0])
    y = np.array([1, 1, 0, 0, 0])
    m = se.metrics_from_delays(d, y)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 2, 1)
    assert abs(m["precision"] - 1 / 3) < 1e-9 and abs(m["recall"] - 0.5) < 1e-9
    penalty = -1 + 2 / (1 + np.exp(-0.0078 * (5 - 1)))
    assert abs(m["f_latency"] - m["f1"] * (1 - penalty)) < 1e-9
    none = se.metrics_from_delays(np.zeros(3, dtype=int), np.array([1, 0, 0]))
    assert none["f1"] == 0 and none["f_latency"] == 0


def test_operating_point_prefers_the_better_rule_and_breaks_ties():
    table = pd.DataFrame([
        {"th": 0.5, "k": 1, "f1": 0.8, "f_latency": 0.7, "median_delay": 10.0},
        {"th": 0.6, "k": 3, "f1": 0.8, "f_latency": 0.7, "median_delay": 5.0},
        {"th": 0.7, "k": 5, "f1": 0.9, "f_latency": 0.6, "median_delay": 2.0},
    ])
    assert se.select_operating_point(table)["th"] == 0.6  # same score and F1, shorter delay
    assert se.select_operating_point(table, metric="f1")["th"] == 0.7


def test_sweep_bootstrap_and_horizon():
    rng = np.random.default_rng(0)
    labels = np.array([1] * 20 + [0] * 60)
    probs = [np.clip(rng.normal(0.65 if y else 0.35, 0.15, size=rng.integers(5, 30)), 0, 1) for y in labels]
    table = se.sweep(probs, labels, [0.4, 0.5, 0.6], [1, 3])
    assert len(table) == 6 and set(table["k"]) == {1, 3}
    op = se.select_operating_point(table)
    ci = se.bootstrap_ci(probs, labels, op["th"], int(op["k"]), n_boot=200, seed=0)
    lo, hi = ci["f1"]
    assert lo <= op["f1"] <= hi
    auc = se.horizon_auc(probs, labels, [1, 5, 1000])
    assert all(0.5 < a <= 1 for a in auc.values())
    assert 0.5 < se.ranking_scores(probs, labels)["pr_auc"] <= 1


def test_feature_groups_partition_the_v2_columns():
    flat = [c for g in FEATURE_GROUPS_V2.values() for c in g]
    assert sorted(flat) == sorted(FEATURE_COLS_V2)
    assert set(EXPERIMENTS_V2["all"]) == set(FEATURE_COLS_V2)
    for g in ("PHQ", "SIM", "LSM", "RE", "OWN_LIWC", "BEHAVIOR"):
        assert set(EXPERIMENTS_V2[f"without_{g}"]).isdisjoint(FEATURE_GROUPS_V2[g])
        assert set(EXPERIMENTS_V2[f"only_{g}"]) == set(FEATURE_GROUPS_V2[g]) | set(FEATURE_GROUPS_V2["CONTEXT"])
    assert not any("so_far" in c or "delta" in c or "rolling" in c or "cummean" in c
                   for c in EXPERIMENTS_V2["post_level_only"])
    assert not any("cummean" in c for c in EXPERIMENTS_V2["no_cummean"])
    assert len(EXPERIMENTS_V2["base_features"]) == 101


def test_run_experiment_trains_selects_evaluates_and_caches():
    P = Paths(root=tempfile.mkdtemp())
    P.make_dirs()
    train, test = make_users(50, seed=1, n_pos=15), make_users(24, seed=2, n_pos=7)
    cfg = ex.Config(n_folds=3, epochs=4, patience=4, hidden_dim=16, n_heads=2, n_layers=1, dropout=0.0, lr=3e-3,
                    thresholds=(0.3, 0.5, 0.7), ks=(1, 3), horizons=(3, 5), n_boot=50)
    res = ex.run_experiment("toy", ["f0", "f1"], train, test, P, cfg, verbose=False)
    out = ex.experiment_dir(P, "toy")
    for f in ("results.json", "scaler.json", "probs.pkl", "oof_sweep.csv", "fold0.pt", "fold1.pt", "fold2.pt"):
        assert (out / f).exists(), f
    assert res["operating_point"]["th"] in cfg.thresholds and res["operating_point"]["k"] in cfg.ks
    assert len(res["test"]["per_fold"]) == 3 and set(res["test"]["ci"]) >= {"f1", "f_latency"}
    probs = ex.load_probs(P, "toy")
    assert all(p is not None for p in probs["oof"]["probs"])  # every training user is held out exactly once
    assert len(probs["test"]["probs"]) == test["user_id"].nunique()
    mtime = (out / "results.json").stat().st_mtime_ns
    again = ex.run_experiment("toy", ["f0", "f1"], train, test, P, cfg, verbose=False)
    assert (out / "results.json").stat().st_mtime_ns == mtime and again == res  # cached
    ex.run_experiment("toy", ["f0", "f1"], train, test, P, ex.Config(**{**cfg.__dict__, "epochs": 3}), verbose=False)
    assert json.load(open(out / "results.json"))["signature"]["config"]["epochs"] == 3  # settings changed
    assert "toy" in ex.summary_table({"toy": res}).index


def test_seed_runs_get_their_own_cache_and_are_summarised():
    from dataclasses import replace

    assert ex.seeded_name("all", ex.BASE_SEED) == "all" and ex.seeded_name("all", 7) == "all_seed7"
    P = Paths(root=tempfile.mkdtemp())
    P.make_dirs()
    train, test = make_users(50, seed=1, n_pos=15), make_users(24, seed=2, n_pos=7)
    base = ex.Config(n_folds=3, epochs=3, patience=3, hidden_dim=16, n_heads=2, n_layers=1, dropout=0.0, lr=3e-3,
                     thresholds=(0.3, 0.5), ks=(1, 3), horizons=(10, 100), n_boot=20)
    for seed in (42, 43):
        ex.run_experiment(ex.seeded_name("toy", seed), ["f0", "f1"], train, test, P, replace(base, seed=seed),
                          verbose=False)
    table = ex.seed_summary(P, ["toy", "missing"], [42, 43, 44])
    assert list(table.index) == ["toy"] and table.loc["toy", "seeds"] == 2
    assert "±" in table.loc["toy", "test_f1"]


def test_training_notebook_result_cells_run_on_toy_results():
    import ast
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nb = json.load(open(ROOT / "Develop" / "Train_and_evaluate.ipynb"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for src in code:
        ast.parse(src)
        assert "/u50/" not in src and "/nfs/" not in src

    P = Paths(root=tempfile.mkdtemp())
    P.make_dirs()
    train, test = make_users(50, seed=1, n_pos=15), make_users(30, seed=2, n_pos=9)
    cfg = ex.Config(n_folds=3, epochs=3, patience=3, hidden_dim=16, n_heads=2, n_layers=1, dropout=0.0,
                    thresholds=(0.3, 0.5), ks=(1, 3), horizons=(3, 5), n_boot=30)
    results = {"all": ex.run_experiment("all", ["f0", "f1"], train, test, P, cfg, verbose=False)}
    env = {"results": results, "ex": ex, "se": se, "np": np, "pd": pd, "plt": plt, "P": P}
    for marker in ("decision rule:", "fig, axes", "per_user ="):
        exec([s for s in code if marker in s][0], env)
    assert not ex.summary_table(results).empty
    plt.close("all")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
