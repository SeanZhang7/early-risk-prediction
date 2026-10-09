"""Feature groups in the repeated cross-validation (cv_all.py): how early each group detects, how late it alarms, and
what kind of users it gets wrong.

    conda run -n hongyi python feature_group_analysis.py

Needs the cross-validation results of the configurations in GROUPS and REFS (cv_all.py --configs only_* context_only).
Writes docs/figures/groups_*.csv and fig_groups_*.png (aggregates only: no text, no user ids).

  - curves: AUC@n (probability after n posts), and the share of depressed / control users already flagged after n posts
    (cumulative recall and false-alarm rate of the decision rule chosen inside each outer fold); mean over the 3
    repeats with 95 % bootstrap intervals over users (stratified, same draws for every configuration);
  - delay table: precision, recall, F1, F_latency and the distribution of the alarm delay of the detected users;
  - error profile: users misclassified in at least 2 of the 3 repeats, their size, mental-health vocabulary and
    self-report phrases (patterns of error_analysis.py), and the overlap of the error sets between configurations.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import cv_significance as cs
import error_analysis as ea
import seq_eval as se
from paths import Paths

GROUPS = {"context_only": "context only (control)", "only_PHQ": "PHQ", "only_OWN_LIWC": "LIWC (15 categories)",
          "only_LSM": "LSM (counts)", "only_LSMX_no_par": "LSM alignment (LSMX)", "only_SIM": "cosine similarity",
          "only_BEHAVIOR": "behaviour"}
REFS = {"lean_PHQ_plus_LIWC": "PHQ + 3 LIWC", "lean_without_RE": "lean (no RE)", "mental_emb": "MentalRoBERTa frozen",
        "ft_logit": "MentalRoBERTa adapter", "tfidf_lr": "TF-IDF"}
COLORS = {"context_only": "#7f7f7f", "only_PHQ": "#1f77b4", "only_OWN_LIWC": "#2ca02c", "only_LSM": "#d62728",
          "only_LSMX_no_par": "#9467bd", "only_SIM": "#8c564b", "only_BEHAVIOR": "#e377c2",
          "lean_PHQ_plus_LIWC": "#17becf", "lean_without_RE": "#bcbd22", "mental_emb": "#ff7f0e", "ft_logit": "#000000",
          "tfidf_lr": "#7fc97f"}
NS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]
N_BOOT = 500
REPEATS = ea.REPEATS


def load_config(P, name):
    """Per repeat: users (sorted), labels, probability after each n in NS, alarm delay, and the long run table."""
    runs, probs = ea.load_runs(P, name)
    out = []
    for r in range(REPEATS):
        sub = runs[runs.repeat == r].sort_values("user")
        users = sub["user"].to_numpy()
        pn = np.array([[probs[(r, u)][min(n, len(probs[(r, u)])) - 1] for n in NS] for u in users])
        out.append({"users": users, "y": sub["y"].to_numpy(), "pn": pn, "delay": sub["delay"].to_numpy()})
    return runs, out


def percentile_band(values):
    return float(np.mean(values)), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def curves(data, draws):
    """Long table: config, metric, n, mean, lo, hi."""
    rows = []
    for name, reps in data.items():
        y = reps[0]["y"]
        for j, n in enumerate(NS):
            for metric in ("auc", "recall", "fpr"):
                point = np.mean([_metric(metric, y, rp["pn"][:, j], rp["delay"], n, np.arange(len(y))) for rp in reps])
                boot = np.array([np.mean([_metric(metric, y[idx], rp["pn"][idx, j], rp["delay"][idx], n, idx)
                                          for rp in reps]) for idx in draws])
                rows.append({"config": name, "metric": metric, "n": n, "mean": point,
                             "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5)})
    return pd.DataFrame(rows)


def _metric(kind, y, score, delay, n, idx):
    if kind == "auc":
        return roc_auc_score(y, score)
    flagged = (delay > 0) & (delay <= n)
    return flagged[y == 1].mean() if kind == "recall" else flagged[y == 0].mean()


def delay_table(data):
    rows = []
    for name, reps in data.items():
        per_rep = [se.metrics_from_delays(rp["delay"], rp["y"]) for rp in reps]
        tp_delay = np.concatenate([rp["delay"][(rp["y"] == 1) & (rp["delay"] > 0)] for rp in reps])
        rows.append({"config": name,
                     **{k: float(np.mean([m[k] for m in per_rep])) for k in
                        ("precision", "recall", "f1", "f_latency", "median_delay")},
                     "delay_q1": np.percentile(tp_delay, 25), "delay_q3": np.percentile(tp_delay, 75),
                     "tp_flagged_within_10": float((tp_delay <= 10).mean()),
                     "tp_flagged_within_50": float((tp_delay <= 50).mean())})
    return pd.DataFrame(rows)


def error_profile(P, names, all_runs):
    posts = ea.post_table(P)
    users = ea.user_table(posts)
    outcome = pd.DataFrame({n: ea.majority(all_runs[n])["outcome"] for n in names})
    ref = outcome["ft_logit"] if "ft_logit" in outcome else None
    dep = users.index.isin(outcome.index[outcome[names[0]].isin(["TP", "FN"])])
    rows = []
    for n in names:
        o = outcome[n]
        row = {"config": n, "FN": int((o == "FN").sum()), "FP": int((o == "FP").sum())}
        for kind in ("FN", "FP"):
            sub = users.loc[o.index[o == kind]]
            row.update({f"{kind}_median_posts": sub["n_posts"].median(),
                        f"{kind}_share_mental_health": sub["share_mental_health"].median(),
                        f"{kind}_self_report": sub["any_self_report"].mean(),
                        f"{kind}_also_ft_logit": float((ref.loc[sub.index] == kind).mean()) if ref is not None and n != "ft_logit" else np.nan})
        dep_users = o.index[o.isin(["TP", "FN"])]
        with_sr = [u for u in dep_users if users.at[u, "any_self_report"] == 1]
        without = [u for u in dep_users if users.at[u, "any_self_report"] == 0]
        row["recall_with_self_report"] = float((o.loc[with_sr] == "TP").mean())
        row["recall_without_self_report"] = float((o.loc[without] == "TP").mean())
        row["n_depressed_with_self_report"], row["n_depressed_without_self_report"] = len(with_sr), len(without)
        rows.append(row)
    overlaps = {}
    for kind in ("FN", "FP"):
        sets = {n: set(outcome.index[outcome[n] == kind]) for n in names}
        overlaps[kind] = pd.DataFrame([[len(sets[a] & sets[b]) / max(len(sets[a] | sets[b]), 1) for b in names]
                                       for a in names], index=names, columns=names)
    return pd.DataFrame(rows), overlaps, outcome, users


BINS = [(1, 99, "<100"), (100, 499, "100-499"), (500, 10 ** 9, ">=500")]


def length_bins(outcome, users):
    """Recall of the depressed and false-alarm rate of the control users by number of posts per user."""
    rows = []
    for lo, hi, label in BINS:
        idx = users.index[(users.n_posts >= lo) & (users.n_posts <= hi)]
        for name in outcome.columns:
            o = outcome.loc[idx, name]
            dep, ctl = o.isin(["TP", "FN"]), o.isin(["TN", "FP"])
            rows.append({"config": name, "posts": label, "n_depressed": int(dep.sum()), "n_control": int(ctl.sum()),
                         "recall": float((o[dep] == "TP").mean()) if dep.any() else np.nan,
                         "false_alarm_rate": float((o[ctl] == "FP").mean()) if ctl.any() else np.nan})
    return pd.DataFrame(rows)


def figure_curves(cur, path, labels):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = {"auc": "AUC after n posts", "recall": "Depressed users flagged within n posts",
              "fpr": "Control users flagged within n posts"}
    for ax, metric in zip(axes, ("auc", "recall", "fpr")):
        for name, label in labels.items():
            d = cur[(cur.config == name) & (cur.metric == metric)]
            style = "--" if name in REFS else "-"
            ax.plot(d["n"], d["mean"], style, marker="o", ms=3, color=COLORS[name], label=label)
            ax.fill_between(d["n"], d["lo"], d["hi"], color=COLORS[name], alpha=0.08)
        ax.set_xscale("log")
        ax.set_xlabel("number of posts read (n)")
        ax.set_title(titles[metric], fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].axhline(0.5, color="grey", lw=0.8)
    axes[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_errors(profile, overlaps, path, labels):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    order = list(labels)
    p = profile.set_index("config").loc[order]
    y = np.arange(len(order))
    axes[0].barh(y - 0.2, p["FN"], 0.4, color="#c0504d", label="false negatives (of 102)")
    axes[0].barh(y + 0.2, p["FP"], 0.4, color="#3b6ea5", label="false positives (of 789)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([labels[c] for c in order], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=7)
    axes[0].set_title("Misclassified users", fontsize=10)
    for ax, kind in zip(axes[1:], ("FN", "FP")):
        m = overlaps[kind].loc[order, order]
        im = ax.imshow(m.to_numpy(), vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([labels[c] for c in order], rotation=70, ha="right", fontsize=7)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([labels[c] for c in order], fontsize=7)
        ax.set_title(f"Overlap of {kind} sets (Jaccard)", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    P = Paths()
    out = P.root / "docs" / "figures"
    labels = {**GROUPS, **REFS}
    names = list(labels)
    data, all_runs = {}, {}
    for n in names:
        all_runs[n], data[n] = load_config(P, n)
    y = data[names[0]][0]["y"]
    for n in names:
        for rp in data[n]:
            if not np.array_equal(rp["y"], y):
                raise ValueError(f"{n}: users differ from the first configuration")
    draws = cs.bootstrap_draws(y, N_BOOT)
    cur = curves(data, draws)
    cur.to_csv(out / "groups_curves.csv", index=False, float_format="%.4f")
    dt = delay_table(data)
    dt.to_csv(out / "groups_delay.csv", index=False, float_format="%.4g")
    profile, overlaps, outcome, users = error_profile(P, names, all_runs)
    length_bins(outcome, users).to_csv(out / "groups_length_bins.csv", index=False, float_format="%.4g")
    profile.to_csv(out / "groups_errors.csv", index=False, float_format="%.4g")
    for kind, m in overlaps.items():
        m.to_csv(out / f"groups_overlap_{kind.lower()}.csv", float_format="%.3f")
    figure_curves(cur, out / "fig_groups_curves.png", labels)
    figure_errors(profile, overlaps, out / "fig_groups_errors.png", labels)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(dt.round(3).to_string(index=False), profile.round(3).T.to_string(), sep="\n\n")
    print(cur[cur.metric == "auc"].pivot(index="config", columns="n", values="mean").round(3).to_string())


if __name__ == "__main__":
    main()
