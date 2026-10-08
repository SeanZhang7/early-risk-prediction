"""Early-detection metrics for per-position user probabilities.

A user is flagged as soon as `k` consecutive posts have probability >= `th`; the delay is the number of
posts read at that moment (0 means never flagged). Everything here works on lists of 1-d probability
arrays (one per user, in time order) and an array of 0/1 user labels.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

LATENCY_P = 0.0078  # eRisk latency-weighted F1 constant


def run_lengths(ok):
    """Length of the current run of True values at every position."""
    c = np.cumsum(ok)
    reset = np.maximum.accumulate(np.where(ok, 0, c))
    return c - reset


def delays(prob_list, th, ks):
    """(n_users, len(ks)) array: posts read when the alarm fires for each k, 0 if it never does."""
    out = np.zeros((len(prob_list), len(ks)), dtype=int)
    for i, p in enumerate(prob_list):
        run = run_lengths(np.asarray(p) >= th)
        for j, k in enumerate(ks):
            hit = run >= k
            if hit.any():
                out[i, j] = int(np.argmax(hit)) + 1
    return out


def metrics_from_delays(d, labels):
    """Precision, recall, F1, delay and latency-weighted F1 for one decision rule (d: delay per user)."""
    y = np.asarray(labels).astype(int)
    pred = d > 0
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    tp_delay = d[pred & (y == 1)]
    if len(tp_delay):
        speed = 1.0 - np.median(-1.0 + 2.0 / (1.0 + np.exp(-LATENCY_P * (tp_delay - 1))))
        median_delay, mean_delay = float(np.median(tp_delay)), float(tp_delay.mean())
    else:
        speed, median_delay, mean_delay = 0.0, np.nan, np.nan
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
            "median_delay": median_delay, "mean_delay": mean_delay, "f_latency": f1 * speed}


def evaluate(prob_list, labels, th, k):
    return metrics_from_delays(delays(prob_list, th, [k])[:, 0], labels)


def sweep(prob_list, labels, thresholds, ks):
    """One row per (threshold, k) with the metrics of that decision rule."""
    rows = []
    for th in thresholds:
        d = delays(prob_list, th, ks)
        for j, k in enumerate(ks):
            rows.append({"th": float(th), "k": int(k), **metrics_from_delays(d[:, j], labels)})
    return pd.DataFrame(rows)


def select_operating_point(table, metric="f_latency"):
    """Best row of a sweep; ties go to higher F1, then shorter delay, then the lower threshold."""
    t = table.assign(_delay=table["median_delay"].fillna(np.inf))
    t = t.sort_values([metric, "f1", "_delay", "th"], ascending=[False, False, True, True])
    return t.iloc[0].drop("_delay").to_dict()


def bootstrap_ci(prob_list, labels, th, k, n_boot=1000, seed=0, level=0.95):
    """Percentile intervals of the metrics when users are resampled with replacement."""
    d = delays(prob_list, th, [k])[:, 0]
    y = np.asarray(labels).astype(int)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        rows.append(metrics_from_delays(d[idx], y[idx]))
    boot = pd.DataFrame(rows)
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return {m: (float(boot[m].quantile(lo)), float(boot[m].quantile(hi)))
            for m in ("precision", "recall", "f1", "median_delay", "f_latency")}


def horizon_auc(prob_list, labels, horizons):
    """User-level ROC-AUC of the probability after the first n posts (the last one for shorter users)."""
    y = np.asarray(labels).astype(int)
    out = {}
    for n in horizons:
        score = [p[min(n, len(p)) - 1] for p in prob_list]
        out[int(n)] = float(roc_auc_score(y, score))
    return out


def ranking_scores(prob_list, labels):
    """ROC-AUC and PR-AUC of the highest probability a user ever reaches."""
    y = np.asarray(labels).astype(int)
    peak = [float(np.max(p)) for p in prob_list]
    return {"roc_auc": float(roc_auc_score(y, peak)), "pr_auc": float(average_precision_score(y, peak))}
