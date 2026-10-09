"""Pooled metrics, confidence intervals and paired significance tests for the repeated cross-validation (cv_all.py).

    conda run -n hongyi python cv_significance.py --repeats 3

Per configuration and repeat every user is predicted exactly once (by the outer fold that held it out). Metrics are
computed on these pooled predictions; the decision-rule metrics (F1, delay, F_latency) use, for every user, the rule
(th, k) chosen inside the user's own outer fold. Uncertainty over users comes from a stratified bootstrap of the
users (the same draws for every configuration, so differences are paired); the repeats are averaged inside every
bootstrap draw. Paired tests on a difference A - B:
  - bootstrap CI and two-sided p-value of the mean difference over repeats (main test),
  - Nadeau-Bengio corrected resampled t-test on the per-fold differences (variance corrected for overlapping
    training sets),
  - DeLong test of ROC-AUC and exact McNemar test of the user-level alarm decisions, per repeat.
p-values of the main test are adjusted with Holm's method within each metric across the pre-specified comparisons.
"""
import argparse
import json
import pickle
from itertools import combinations  # noqa: F401  (kept for ad-hoc pair lists)

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

import cv_splits
import seq_eval as se
from paths import Paths

METRICS = ["roc", "pr", "auc10", "auc100", "f1", "f_latency", "median_delay"]
RANK_METRICS = ["roc", "pr", "auc10", "auc100"]
N_SPLITS = cv_splits.N_SPLITS

# Pre-specified comparisons (A, B): the difference is A - B.
PAIRS = [
    ("lean2_without_RE", "only_PHQ"),
    ("lean2_without_RE", "lean_PHQ_plus_LIWC"),
    ("lean2_without_RE", "lean_without_RE"),
    ("lean2_without_RE", "mental_emb"),
    ("lean2_without_RE", "ft_logit"),
    ("ft_logit", "mental_emb"),
    ("ft_logit_plus_lean2_without_RE", "ft_logit"),
    ("ft_logit_plus_lean2_without_RE", "lean2_without_RE"),
    ("mental_plus_lean2_without_RE", "lean2_without_RE"),
]
# Second family, added with the bag-of-words baseline (cv_tfidf.py); Holm is applied within each family so that the
# adjusted p-values of the first family do not change.
PAIRS_BASELINE = [
    ("ft_logit", "tfidf_lr"),
    ("mental_emb", "tfidf_lr"),
    ("only_PHQ", "tfidf_lr"),
    ("lean2_without_RE", "tfidf_lr"),
]
# Third family (single feature groups, cv_all.py --configs only_*): each group against the context-only control, and
# PHQ against the other groups. Holm within the family.
PAIRS_GROUPS = [(g, "context_only") for g in ("only_PHQ", "only_OWN_LIWC", "only_LSM", "only_LSMX_no_par", "only_SIM",
                                              "only_BEHAVIOR")] + [
    ("only_PHQ", "only_OWN_LIWC"), ("only_PHQ", "only_LSMX_no_par"), ("only_PHQ", "only_LSM"),
]


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def nadeau_bengio(diffs, n_test, n_train):
    """Corrected resampled t-test of the mean of per-fold differences; returns (t, two-sided p)."""
    d = np.asarray(diffs, dtype=float)
    j = len(d)
    var = d.var(ddof=1)
    if var <= 1e-18 * max(1.0, float(d.mean()) ** 2):
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1.0 / j + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), j - 1))


def delong_p(y, s1, s2):
    """Two-sided DeLong p-value for the difference of two correlated ROC-AUCs; returns (auc1, auc2, p)."""
    y = np.asarray(y).astype(bool)
    x = np.stack([np.asarray(s1)[y], np.asarray(s2)[y]])  # (2, m) positives
    z = np.stack([np.asarray(s1)[~y], np.asarray(s2)[~y]])  # (2, n) negatives
    m, n = x.shape[1], z.shape[1]
    v10, v01, aucs = [], [], []
    for c in range(2):
        psi = (x[c][:, None] > z[c][None, :]).astype(float) + 0.5 * (x[c][:, None] == z[c][None, :])
        v10.append(psi.mean(axis=1))
        v01.append(psi.mean(axis=0))
        aucs.append(psi.mean())
    s10, s01 = np.cov(np.stack(v10)), np.cov(np.stack(v01))
    cov = s10 / m + s01 / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return aucs[0], aucs[1], 1.0
    zstat = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs[0], aucs[1], float(2 * stats.norm.sf(abs(zstat)))


def mcnemar_p(correct_a, correct_b):
    """Exact two-sided McNemar test on the users that exactly one of the two decisions classifies correctly."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    if only_a + only_b == 0:
        return 1.0
    return float(stats.binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue)


def metric_values(y, peak, p10, p100, delay):
    """All pooled metrics for the given users (arrays aligned)."""
    dm = se.metrics_from_delays(delay, y)
    return {"roc": roc_auc_score(y, peak), "pr": average_precision_score(y, peak), "auc10": roc_auc_score(y, p10),
            "auc100": roc_auc_score(y, p100), "f1": dm["f1"], "f_latency": dm["f_latency"],
            "median_delay": dm["median_delay"]}


def load_config(P, name, repeats):
    """{repeat: dict of arrays over the users sorted by id} for the repeats whose 5 folds are all present, plus
    per-fold peak scores for the fold-level test."""
    out = {}
    for r in range(repeats):
        paths = [cv_splits.result_path(P, name, r, k) for k in range(N_SPLITS)]
        if not all(p.exists() for p in paths):
            continue
        rows = []
        for k, path in enumerate(paths):
            res = pickle.load(open(path, "rb"))
            for u, y, probs in zip(res["users"], res["labels"], res["probs"]):
                d = int(se.delays([probs], res["th"], [res["k"]])[0, 0])
                rows.append((u, int(y), float(np.max(probs)), float(probs[min(10, len(probs)) - 1]),
                             float(probs[min(100, len(probs)) - 1]), d, k))
        df = pd.DataFrame(rows, columns=["user", "y", "peak", "p10", "p100", "delay", "fold"]).sort_values("user")
        if df["user"].duplicated().any():
            raise ValueError(f"{name} repeat {r}: a user was predicted more than once")
        out[r] = df.reset_index(drop=True)
    return out


def bootstrap_draws(y, n_boot, seed=0):
    """Stratified resamples of user positions, shape (n_boot, n_users)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return np.stack([np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(n_boot)])


def analyse(P, configs, repeats, n_boot=2000):
    data = {c: load_config(P, c, repeats) for c in configs}
    data = {c: d for c, d in data.items() if d}
    if not data:
        raise SystemExit("no finished configuration found")
    users = None
    for c, d in data.items():
        for r, df in d.items():
            if users is None:
                users = df["user"].to_numpy()
            elif not np.array_equal(users, df["user"].to_numpy()):
                raise ValueError(f"{c}: user order differs")
    y = next(iter(next(iter(data.values())).values()))["y"].to_numpy()
    draws = bootstrap_draws(y, n_boot)

    point_rep, boot_rep = {}, {}
    for c, d in data.items():
        point_rep[c], boot_rep[c] = {}, {}
        for r in sorted(d):
            a = {col: d[r][col].to_numpy() for col in ("peak", "p10", "p100", "delay")}
            point_rep[c][r] = metric_values(y, a["peak"], a["p10"], a["p100"], a["delay"])
            per_draw = {m: np.zeros(n_boot) for m in METRICS}
            for b in range(n_boot):
                idx = draws[b]
                v = metric_values(y[idx], a["peak"][idx], a["p10"][idx], a["p100"][idx], a["delay"][idx])
                for m in METRICS:
                    per_draw[m][b] = v[m]
            boot_rep[c][r] = per_draw
        print(f"bootstrapped {c} ({len(d)} repeats)", flush=True)

    def pm(c, reps, m):
        return float(np.mean([point_rep[c][r][m] for r in reps]))

    def bm(c, reps, m):
        return np.nanmean([boot_rep[c][r][m] for r in reps], axis=0)

    summary = []
    for c in data:
        reps = sorted(data[c])
        row = {"config": c, "repeats": len(reps)}
        for m in METRICS:
            lo, hi = np.nanpercentile(bm(c, reps, m), [2.5, 97.5])
            row[m], row[f"{m}_lo"], row[f"{m}_hi"] = pm(c, reps, m), lo, hi
        summary.append(row)

    pair_rows = []
    for family, a, b in ([(1, a, b) for a, b in PAIRS] + [(2, a, b) for a, b in PAIRS_BASELINE]
                         + [(3, a, b) for a, b in PAIRS_GROUPS]):
        if a not in data or b not in data:
            continue
        common = sorted(set(data[a]) & set(data[b]))
        if not common:
            continue
        row = {"family": family, "A": a, "B": b, "repeats": len(common)}
        for m in METRICS:
            dd = bm(a, common, m) - bm(b, common, m)
            lo, hi = np.nanpercentile(dd, [2.5, 97.5])
            ok = dd[~np.isnan(dd)]
            p = min(1.0, 2 * min(np.mean(ok <= 0), np.mean(ok >= 0))) if len(ok) else np.nan
            row[f"d_{m}"], row[f"d_{m}_lo"], row[f"d_{m}_hi"] = pm(a, common, m) - pm(b, common, m), lo, hi
            row[f"p_{m}"] = max(p, 1.0 / (n_boot + 1))
        for m in ("roc", "pr"):
            f = (lambda s: roc_auc_score(s["y"], s["peak"])) if m == "roc" else (
                lambda s: average_precision_score(s["y"], s["peak"]))
            fd = []
            for r in common:
                for k in range(N_SPLITS):
                    fd.append(f(data[a][r][data[a][r]["fold"] == k]) - f(data[b][r][data[b][r]["fold"] == k]))
            n_test = len(users) / N_SPLITS
            row[f"nb_p_{m}"] = nadeau_bengio(fd, n_test, len(users) - n_test)[1]
        dl, mc = [], []
        for r in common:
            da, db = data[a][r], data[b][r]
            dl.append(delong_p(y, da["peak"].to_numpy(), db["peak"].to_numpy())[2])
            mc.append(mcnemar_p((da["delay"].to_numpy() > 0) == (y == 1), (db["delay"].to_numpy() > 0) == (y == 1)))
        row["delong_p_median"], row["delong_p_max"] = float(np.median(dl)), float(np.max(dl))
        row["mcnemar_p_median"], row["mcnemar_p_max"] = float(np.median(mc)), float(np.max(mc))
        pair_rows.append(row)
    pairs = pd.DataFrame(pair_rows)
    for m in METRICS:
        if len(pairs):
            pairs[f"p_{m}_holm"] = np.nan
            for fam in pairs["family"].unique():
                sel = (pairs["family"] == fam).to_numpy()
                pairs.loc[sel, f"p_{m}_holm"] = holm(pairs.loc[sel, f"p_{m}"].to_numpy())
    return pd.DataFrame(summary), pairs, int(y.sum()), len(y)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--configs", nargs="+", default=None)
    args = ap.parse_args()
    P = Paths()
    configs = args.configs or sorted({c for pair in PAIRS + PAIRS_BASELINE + PAIRS_GROUPS for c in pair})
    summary, pairs, n_pos, n_users = analyse(P, configs, args.repeats, args.boot)
    out = P.models_dir / "cv_all"
    summary.to_csv(out / "summary.csv", index=False)
    pairs.to_csv(out / "pairs.csv", index=False)
    json.dump({"n_users": n_users, "n_positive": n_pos, "repeats": args.repeats, "boot": args.boot},
              open(out / "meta.json", "w"))
    pd.set_option("display.width", 250, "display.max_columns", 50)
    print(f"\n{n_users} users, {n_pos} depressed; metrics = mean over repeats with 95% bootstrap interval")
    show = summary[["config", "repeats"]].copy()
    for m in ("roc", "pr", "auc10", "auc100", "f1", "f_latency"):
        show[m] = [f"{r[m]:.3f} [{r[m + '_lo']:.3f}, {r[m + '_hi']:.3f}]" for _, r in summary.iterrows()]
    print(show.to_string(index=False))
    cols = ["family", "A", "B", "repeats"]
    for m in ("pr", "roc", "auc10", "f1"):
        print(f"\nDifference A - B in {m}: mean [95% CI], bootstrap p, Holm p, Nadeau-Bengio p")
        t = pairs[cols].copy()
        t["diff"] = [f"{r['d_' + m]:+.3f} [{r['d_' + m + '_lo']:+.3f}, {r['d_' + m + '_hi']:+.3f}]" for _, r in pairs.iterrows()]
        t["p"] = pairs[f"p_{m}"].round(4)
        t["p_holm"] = pairs[f"p_{m}_holm"].round(4)
        if f"nb_p_{m}" in pairs:
            t["p_NB"] = pairs[f"nb_p_{m}"].round(4)
        print(t.to_string(index=False))
    print("\nDeLong (ROC) and McNemar (decisions), per repeat: median and maximum p")
    print(pairs[["A", "B", "delong_p_median", "delong_p_max", "mcnemar_p_median", "mcnemar_p_max"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
