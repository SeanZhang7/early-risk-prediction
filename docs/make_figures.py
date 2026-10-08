"""Figures for docs/SYSTEM_REPORT_EN.md. Numbers are read from models/v2/classifier/*/results.json (3 seeds),
models/v2/cv_all/ or the feature tables, except fig_feature_signal, whose values are copied from Section 4.4.
Run: conda run -n hongyi python docs/make_figures.py"""
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import experiment as ex  # noqa: E402
from paths import Paths  # noqa: E402

P = Paths()
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (42, 43, 44)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
BLUE, ORANGE, GREY, GREEN, RED = "#3b6ea5", "#e08a2c", "#8c8c8c", "#4c9a6a", "#c0504d"

METRICS = {
    "f1": lambda r: r["test"]["f1"],
    "roc": lambda r: r["test"]["ranking"]["roc_auc"],
    "pr": lambda r: r["test"]["ranking"]["pr_auc"],
    "auc10": lambda r: r["test"]["horizon_auc"]["10"],
    "auc100": lambda r: r["test"]["horizon_auc"]["100"],
    "oof_pr": lambda r: r["oof_ranking"]["pr_auc"],
    "delay": lambda r: r["test"]["median_delay"],
    "flat": lambda r: r["test"]["f_latency"],
}


def runs(name):
    out = []
    for s in SEEDS:
        path = ex.experiment_dir(P, ex.seeded_name(name, s)) / "results.json"
        out.append(json.load(open(path)))
    return out


def vals(name, metric):
    return np.array([METRICS[metric](r) for r in runs(name)], dtype=float)


def mean_std(name, metric):
    v = vals(name, metric)
    return v.mean(), v.std(ddof=1)


def delta(name, base, metric):
    d = vals(name, metric) - vals(base, metric)
    return d.mean(), d.std(ddof=1)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("saved", name, flush=True)


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=color, ec="#333", lw=1, alpha=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#333"))

    box(0.1, 1.6, 1.6, 1.0, "Raw Reddit threads\n+ user labels", "#e8e8e8")
    box(2.1, 1.6, 1.9, 1.0, "Clean, user-level split,\ntime sort\n(data_prep.py)", "#cfe2f3")
    box(4.5, 3.0, 2.2, 0.9, "LIWC counts (body, parent)\n-> LSM / LSMX / own LIWC", "#d9ead3")
    box(4.5, 2.0, 2.2, 0.8, "PHQ-9 item similarity", "#d9ead3")
    box(4.5, 1.0, 2.2, 0.8, "RE (distilgpt2, 3-fold\ncross-fitted)", "#d9ead3")
    box(4.5, 0.1, 2.2, 0.7, "Frozen MentalRoBERTa\nembedding (comparison)", "#fff2cc")
    box(7.1, 1.6, 1.8, 1.0, "History features\n(features_history.py)", "#cfe2f3")
    box(9.2, 1.6, 1.7, 1.0, "Causal Transformer\n+ 5-fold CV, OOF rule\n(th, k)", "#f4cccc")
    arrow(1.7, 2.1, 2.1, 2.1)
    for y in (3.45, 2.4, 1.4):
        arrow(4.0, 2.1, 4.5, y)
        arrow(6.7, y, 7.1, 2.1)
    arrow(4.0, 2.1, 4.5, 0.45)
    arrow(6.7, 0.45, 9.2, 1.7)
    arrow(8.9, 2.1, 9.2, 2.1)
    ax.text(5.5, 4.1, "Post-level features (no history)", ha="center", fontsize=9, style="italic")
    save(fig, "fig_pipeline.png")


def fig_overview():
    names = ["post_level_only", "only_PHQ", "all", "no_cummean", "lean", "lean2", "mental_emb", "mental_plus_lean2"]
    labels = ["post-level\nonly", "only\nPHQ", "all\n(150)", "no\ncummean", "lean\n(44)", "lean2\n(76)", "Mental\nfrozen", "Mental\n+ lean2"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, metric, title in zip(axes, ("pr", "oof_pr", "auc10"), ("Test PR-AUC", "OOF PR-AUC (81 positives)", "Test AUC@10 (early detection)")):
        m = np.array([mean_std(n, metric) for n in names])
        colors = [GREEN if n.startswith("lean") else (ORANGE if n.startswith("mental") else BLUE) for n in names]
        ax.bar(range(len(names)), m[:, 0], yerr=m[:, 1], color=colors, capsize=3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(title, fontsize=10)
        lo = max(0.0, m[:, 0].min() - 0.12)
        ax.set_ylim(lo, min(1.0, m[:, 0].max() + 0.06))
    save(fig, "fig_overview.png")


def fig_length_shortcut():
    old = pd.read_pickle(ROOT / "Archive" / "Features" / "MERGED_test_full_fixed.pkl")
    new = pd.read_pickle(P.features("test"))
    thresholds = (100, 300, 600)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, df, title in zip(axes, (old, new), ("Old test set (control posts removed)", "New test set (full data)")):
        g = df.groupby("user_id").agg(n=("user_id", "size"), y=("Depression", "max"))
        dep = [(g[g.y == 1].n >= t).mean() for t in thresholds]
        ctl = [(g[g.y == 0].n >= t).mean() for t in thresholds]
        x = np.arange(len(thresholds))
        ax.bar(x - 0.18, dep, 0.36, label="depressed users", color=RED)
        ax.bar(x + 0.18, ctl, 0.36, label="control users", color=BLUE)
        ax.set_xticks(x)
        ax.set_xticklabels([f">= {t} posts" for t in thresholds])
        ax.set_title(f"{title}\nlongest control user: {int(g[g.y == 0].n.max())} posts", fontsize=9)
    axes[0].set_ylabel("share of users")
    axes[0].legend(fontsize=8)
    save(fig, "fig_length_shortcut.png")


def fig_feature_signal():
    rows = [("phq_score (cummean)", 0.928), ("liwc_HEALTH (cummean)", 0.893), ("liwc_ANX (cummean)", 0.852),
            ("RE (user mean, inverted)", 0.852), ("liwc_SAD (cummean)", 0.819),
            ("negative_mean LSM (inverted)", 0.816), ("log_tokens (cummean)", 0.732),
            ("sim (cummean)", 0.672), ("night (cummean)", 0.615), ("LSM_mean (inverted)", 0.605)]
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color=BLUE)
    ax.axvline(0.5, color=GREY, ls="--", lw=1)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("user-level AUC (training users; inverted features shown as 1 - AUC)")
    for i, r in enumerate(rows):
        ax.text(r[1] + 0.005, i, f"{r[1]:.3f}", va="center", fontsize=8)
    save(fig, "fig_feature_signal.png")


def fig_group_ablation():
    groups = ["PHQ", "SIM", "LSM", "RE", "OWN_LIWC", "BEHAVIOR"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    only = ["post_level_only"] + [f"only_{g}" for g in groups]
    lab = ["post-level only"] + [f"only {g}" for g in groups]
    m = np.array([mean_std(n, "pr") for n in only])
    axes[0].barh(lab[::-1], m[::-1, 0], xerr=m[::-1, 1], color=BLUE, capsize=3)
    axes[0].axvline(mean_std("all", "pr")[0], color=GREY, ls="--", lw=1, label="all (150 features)")
    axes[0].set_title("Single group kept: test PR-AUC", fontsize=10)
    axes[0].legend(fontsize=8, loc="lower right")
    for ax, metric, title in zip(axes[1:], ("pr", "roc"), ("Group removed: change in test PR-AUC", "Group removed: change in test ROC-AUC")):
        d = np.array([delta(f"without_{g}", "all", metric) for g in groups])
        colors = [RED if v < -0.02 else GREY for v in d[:, 0]]
        ax.barh(groups[::-1], d[::-1, 0], xerr=d[::-1, 1], color=colors[::-1], capsize=3)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    save(fig, "fig_group_ablation.png")


def fig_lean2_ablation():
    parts = ["PHQ", "cummean", "LIWC", "LSMalign", "lsmr", "dlsm", "RE", "post_index"]
    names = {"PHQ": "PHQ", "cummean": "running means", "LIWC": "3 LIWC", "LSMalign": "LSM alignment (all)",
             "lsmr": "  lsmr_*", "dlsm": "  dlsm_*", "RE": "RE", "post_index": "post_index"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, metric, title in zip(axes, ("pr", "oof_pr"), ("Change in test PR-AUC when removed", "Change in OOF PR-AUC when removed")):
        full = np.array([delta(f"lean2_without_{p}", "lean2", metric) for p in parts])
        par = np.array([delta(f"parent_lean2_without_{p}", "parent_lean2", metric) for p in parts])
        y = np.arange(len(parts))[::-1]
        ax.barh(y + 0.19, full[:, 0], 0.38, xerr=full[:, 1], color=BLUE, capsize=2, label="full data")
        ax.barh(y - 0.19, par[:, 0], 0.38, xerr=par[:, 1], color=ORANGE, capsize=2, label="parent-only data")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([names[p] for p in parts])
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=8, loc="lower left")
    save(fig, "fig_lean2_ablation.png")


def lean2_label_correlation(n_early=10):
    """Pearson r between each lean2 feature and the user label on the training users, three views."""
    import feature_cols as fc
    df = pd.read_pickle(P.features("train")).sort_values(["user_id", "created_utc"], kind="stable")
    cols = fc.EXPERIMENTS_V2["lean2"]
    parent_only = [c for c in cols if c.startswith(("lsmr_", "dlsm_")) and not c.endswith("_cummean")]
    cumulative = [c for c in cols if c.endswith("_cummean") or "_so_far" in c or c == "max_gap"]
    y_post = df["Depression"].to_numpy(float)
    rank = df.groupby("user_id").cumcount()
    g = df.groupby("user_id", sort=True)
    y_user = g["Depression"].max().to_numpy(float)
    first = df[rank < n_early].groupby("user_id", sort=True)
    last = g.tail(1).set_index("user_id").sort_index()
    at_n = df[rank < n_early].groupby("user_id", sort=True).tail(1).set_index("user_id").sort_index()
    users = g.size().index
    out, user_mat = {}, {}
    for c in cols:
        ok = (df["has_parent"] == 1).to_numpy() if c in parent_only else np.ones(len(df), bool)
        post_r = np.corrcoef(df.loc[ok, c], y_post[ok])[0, 1]
        if c in parent_only:
            m_all = df[ok].groupby("user_id")[c].mean().reindex(users)
            m_early = df[ok & (rank < n_early).to_numpy()].groupby("user_id")[c].mean().reindex(users)
        elif c in cumulative:
            m_all, m_early = last[c].reindex(users), at_n[c].reindex(users)
        else:
            m_all, m_early = g[c].mean(), first[c].mean().reindex(users)
        user_mat[c] = m_all
        r = []
        for m in (m_all, m_early):
            v = m.notna().to_numpy()
            r.append(np.corrcoef(m[v], y_user[v])[0, 1] if m[v].std() > 0 else np.nan)
        out[c] = [post_r, *r]
    res = pd.DataFrame(out, index=["post-level", "user-level (all posts)", f"user-level (first {n_early} posts)"]).T
    return res, pd.DataFrame(user_mat), pd.Series(y_user, index=users)


def _lean2_type(c):
    if c.startswith("PHQ-9_archetype_scale"):
        return "PHQ items"
    if c in ("post_index", "is_comment", "has_parent"):
        return "context"
    if c.startswith("lsmr_"):
        return "lsmr running mean" if c.endswith("_cummean") else "lsmr (rate LSM)"
    if c.startswith("dlsm_"):
        return "dlsm running mean" if c.endswith("_cummean") else "dlsm (signed diff.)"
    if c.startswith("liwc_"):
        return "LIWC"
    if c.startswith("re"):
        return "RE"
    return "PHQ score / history"


LEAN2_TYPES = ["PHQ items", "PHQ score / history", "RE", "LIWC", "lsmr (rate LSM)", "lsmr running mean",
               "dlsm (signed diff.)", "dlsm running mean", "context"]


def _lean2_short(c):
    return c.replace("PHQ-9_archetype_scale.", "item ").replace(".sim", "").replace("_cummean", " (cummean)")


def fig_lean2_correlation():
    res, user_mat, y_user = lean2_label_correlation()
    res.to_csv(OUT / "lean2_label_correlation.csv", float_format="%.4f")
    types = pd.Series({c: _lean2_type(c) for c in res.index})
    order = [c for t in LEAN2_TYPES for c in res.index if types[c] == t]
    res = res.loc[order]
    # (a) feature x label, grouped by type
    fig, ax = plt.subplots(figsize=(5.6, 8.4))
    vmax = float(np.nanmax(np.abs(res.to_numpy())))
    im = ax.imshow(res.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            v = res.iat[i, j]
            ax.text(j, i, "nan" if np.isnan(v) else f"{v:+.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if abs(v) > 0.6 * vmax else "black")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_lean2_short(c) for c in order], fontsize=5.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["post-level", "user-level\nall posts", "user-level\nfirst 10 posts"], fontsize=6.5)
    ax.xaxis.tick_top()
    start = 0
    for t in LEAN2_TYPES:
        n = int((types[order] == t).sum())
        if start:
            ax.axhline(start - 0.5, color="black", lw=0.8)
        ax.text(3.0, start + n / 2 - 0.5, t, ha="left", va="center", fontsize=7, fontweight="bold", clip_on=False)
        start += n
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.025, pad=0.012, shrink=0.8)
    cb.set_label("Pearson r with the depression label", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    save(fig, "fig_lean2_correlation.png")
    # (b) correlation between user-level feature values, label as the last row/column
    m = user_mat[order].assign(label=y_user).corr()
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(m.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    mid, start = [], 0
    for t in LEAN2_TYPES:
        n = int((types[order] == t).sum())
        mid.append(start + (n - 1) / 2)
        if start:
            ax.axhline(start - 0.5, color="black", lw=0.5)
            ax.axvline(start - 0.5, color="black", lw=0.5)
        start += n
    ax.axhline(len(order) - 0.5, color="black", lw=1.2)
    ax.axvline(len(order) - 0.5, color="black", lw=1.2)
    ticks, labels = mid + [len(order)], LEAN2_TYPES + ["label"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("Pearson r (user-level values, training users)", fontsize=7)
    save(fig, "fig_lean2_corr_matrix.png")


def fig_cv():
    """Pooled 5x3 cross-validation: mean over repeats with 95 % bootstrap intervals over users (cv_significance.py)."""
    s = pd.read_csv(P.models_dir / "cv_all" / "summary.csv").sort_values("pr")
    colors = {"tfidf_lr": GREY, "mental_emb": ORANGE, "ft_logit": "#b35a2a"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, m, title in zip(axes, ("roc", "pr", "f_latency"), ("ROC-AUC", "PR-AUC", "F_latency")):
        y = np.arange(len(s))
        ax.errorbar(s[m], y, xerr=[s[m] - s[m + "_lo"], s[m + "_hi"] - s[m]], fmt="none", ecolor=GREY, capsize=2, lw=1)
        ax.scatter(s[m], y, c=[colors.get(c, BLUE if "lean" in c or c == "only_PHQ" else "#7a4a9e") for c in s["config"]], zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(s["config"], fontsize=7)
        ax.set_title(f"{title} (5x3 CV, 891 users)", fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    save(fig, "fig_cv.png")


def fig_lsm_story():
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    alone = [("old LSM", "only_LSM", "parent_only_LSM"), ("LSMX: alignment\nonly", "only_LSMX_no_par", "parent_only_LSMX_no_par"),
             ("LSMX: parent\nrates only", "only_LSMX_par", "parent_only_LSMX_par"), ("LSMX: all", "only_LSMX", "parent_only_LSMX")]
    plus = [("PHQ only", "only_PHQ", "parent_only_PHQ"), ("PHQ + old LSM", "lean_PHQ_plus_LSM", "parent_lean_PHQ_plus_LSM"),
            ("PHQ + alignment", "lean_PHQ_plus_LSMX_no_par", "parent_lean_PHQ_plus_LSMX_no_par"),
            ("PHQ + parent rates", "lean_PHQ_plus_LSMX_par", "parent_lean_PHQ_plus_LSMX_par")]
    for ax, spec, title in zip(axes, (alone, plus), ("LSM variants used alone", "Added on top of PHQ")):
        x = np.arange(len(spec))
        for off, idx, color, lab in ((-0.19, 1, BLUE, "full data"), (0.19, 2, ORANGE, "parent-only data")):
            m = np.array([mean_std(s[idx], "pr") for s in spec])
            ax.bar(x + off, m[:, 0], 0.38, yerr=m[:, 1], color=color, capsize=3, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels([s[0] for s in spec], fontsize=8)
        ax.set_ylabel("test PR-AUC")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0.4, 0.9)
    axes[0].legend(fontsize=8)
    save(fig, "fig_lsm_story.png")


def fig_mental():
    names = ["only_PHQ", "lean2", "mental_emb", "ft_logit", "ft_logit_plus_lean2"]
    labels = ["only PHQ", "lean2", "Mental\n(frozen)", "Mental\n(adapter FT)", "adapter FT\n+ lean2"]
    legend = ["only PHQ", "lean2", "MentalRoBERTa frozen", "MentalRoBERTa adapter fine-tuned", "adapter fine-tuned + lean2"]
    colors = [BLUE, GREEN, ORANGE, "#b35a2a", "#7a4a9e"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9), gridspec_kw={"width_ratios": [2.8, 1, 1]})
    metrics = [("roc", "ROC"), ("pr", "PR"), ("auc10", "AUC@10"), ("auc100", "AUC@100")]
    w = 0.16
    for i, (n, c, lab) in enumerate(zip(names, colors, legend)):
        m = np.array([mean_std(n, k) for k, _ in metrics])
        axes[0].bar(np.arange(len(metrics)) + (i - 2) * w, m[:, 0], w, yerr=m[:, 1], color=c, capsize=2, label=lab)
    axes[0].set_xticks(range(len(metrics)))
    axes[0].set_xticklabels([t for _, t in metrics])
    axes[0].set_ylim(0.6, 1.0)
    axes[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
    axes[0].set_title("Ranking quality (test)", fontsize=10)
    for ax, metric, title in ((axes[1], "delay", "Median alarm delay (posts)"), (axes[2], "flat", "F_latency")):
        m = np.array([mean_std(n, metric) for n in names])
        ax.bar(range(len(names)), m[:, 0], yerr=m[:, 1], color=colors, capsize=3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_title(title, fontsize=10)
    save(fig, "fig_mental.png")


def fig_trajectories(name="lean2", seed=42, max_n=300):
    path = ex.experiment_dir(P, ex.seeded_name(name, seed))
    probs = pickle.load(open(path / "probs.pkl", "rb"))["test"]
    th = json.load(open(path / "results.json"))["operating_point"]["th"]
    y, seqs = np.asarray(probs["labels"]), probs["probs"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for label, color, text in ((1, RED, "depressed users"), (0, BLUE, "control users")):
        sel = [s for s, l in zip(seqs, y) if l == label]
        xs, med, lo, hi = [], [], [], []
        for n in range(1, max_n + 1):
            v = [s[n - 1] for s in sel if len(s) >= n]
            if len(v) >= 5:
                xs.append(n)
                med.append(np.median(v))
                lo.append(np.percentile(v, 25))
                hi.append(np.percentile(v, 75))
        ax.plot(xs, med, color=color, label=f"{text} (median)")
        ax.fill_between(xs, lo, hi, color=color, alpha=0.2)
    ax.axhline(th, color=GREY, ls="--", lw=1, label=f"selected threshold = {th}")
    ax.set_xlabel("number of posts read")
    ax.set_ylabel("predicted probability of depression")
    ax.set_xscale("log")
    ax.set_title(f"{name} (seed {seed}): probability after n posts, test users (IQR band)", fontsize=9)
    ax.legend(fontsize=8)
    save(fig, "fig_trajectories.png")


def fig_cost():
    """Per-post inference cost (batch 1) from docs/benchmark_results.json; parts are summed, not measured end to end."""
    rows = json.load(open(ROOT / "docs" / "benchmark_results.json"))["results"]

    def ms(component, setting):
        r = next(r for r in rows if r["component"].startswith(component) and r["setting"].startswith(setting))
        return r["ms_mean"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, dev, title in ((axes[0], "GPU fp32", "RTX 4090 (fp32, batch 1)"), (axes[1], "CPU 8 thread(s) fp32", "CPU, 8 threads (fp32, batch 1)")):
        liwc = ms("LIWC, body + parent", "CPU 1 thread")
        hist = ms("History features", "30 users")
        phq = ms("PHQ-9", f"batch 1, {dev}")
        ment = ms("MentalRoBERTa", f"batch 1, {dev}")
        re_ = ms("RE: 2", f"batch 1, {dev}")
        seq = ms("Sequence model", f"one forward over 100 posts, {dev}")
        systems = [("lean2\nwithout RE", dict(LIWC=liwc, PHQ=phq, History=hist, Sequence=seq)),
                   ("lean2\n(1 RE pair)", dict(LIWC=liwc, PHQ=phq, RE=re_, History=hist, Sequence=seq)),
                   ("MentalRoBERTa\nfrozen", dict(MentalRoBERTa=ment, Sequence=seq)),
                   ("MentalRoBERTa\n+ lean2 w/o RE", dict(MentalRoBERTa=ment, LIWC=liwc, PHQ=phq, History=hist, Sequence=seq))]
        colors = {"LIWC": "#9fc5e8", "PHQ": BLUE, "RE": GREEN, "History": GREY, "Sequence": RED, "MentalRoBERTa": ORANGE}
        for i, (name, parts) in enumerate(systems):
            bottom = 0.0
            for k, v in parts.items():
                ax.bar(i, v, bottom=bottom, color=colors[k], label=k if i in (0, 1, 2) else None)
                bottom += v
            ax.text(i, bottom, f"{bottom:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(systems)))
        ax.set_xticklabels([s[0] for s in systems], fontsize=7.5)
        ax.set_ylabel("ms per post")
        ax.set_title(title, fontsize=10)
    handles, labels = {}, []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles[l] = h
    fig.legend(handles.values(), handles.keys(), fontsize=8, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.06))
    save(fig, "fig_cost.png")


if __name__ == "__main__":
    for f in (fig_pipeline, fig_overview, fig_length_shortcut, fig_feature_signal, fig_group_ablation,
              fig_lean2_ablation, fig_lean2_correlation, fig_cv, fig_lsm_story, fig_mental, fig_trajectories, fig_cost):
        f()
