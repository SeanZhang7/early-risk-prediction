"""Error analysis of the repeated cross-validation (results of cv_all.py and cv_tfidf.py): which users are misclassified,
by which models, and how they differ from the correctly classified users.

    conda run -n hongyi python error_analysis.py

A user counts as misclassified by a configuration when the alarm decision is wrong in the majority of the 3 repeats.
Aggregate tables and the figure go to docs/figures/ (no text, no user ids). Per-user tables and post excerpts for manual
reading go to Results/v2/error_analysis/ (git-ignored: it contains Reddit text covered by the data-use agreement).
"""
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

import cv_all
import cv_splits
import cv_tfidf
import seq_eval as se
from paths import Paths

CONFIGS = ["ft_logit", "lean2_without_RE", "only_PHQ", "mental_emb", "tfidf_lr"]
MAIN = "ft_logit"
REPEATS = 3
LEXICON = {
    "mental_health": r"\b(?:depress\w*|anxi\w*|therap\w*|psychiatr\w*|antidepress\w*|ssris?|zoloft|lexapro|prozac|"
                     r"sertraline|wellbutrin|mental health|suicid\w*|panic attacks?|bipolar|ptsd|self[- ]harm\w*)\b",
    "self_report": r"\b(?:i (?:have|had|was diagnosed with|got diagnosed with|suffer from|struggle with|deal with) "
                   r"(?:\w+ ){0,2}depression|my depression|i'?m depressed|i am depressed|"
                   r"diagnosed with (?:\w+ ){0,2}depression)\b",
    "life_events": r"\b(?:died|passed away|funeral|griev\w*|grief|divorc\w*|break ?up|broke up|cancer|lost my job|"
                   r"got fired|laid off)\b",
    "physical_health": r"\b(?:pain|doctors?|surgery|chronic|illness|symptoms?|hospital\w*|medications?|meds)\b",
    "substances": r"\b(?:alcohol\w*|drunk|weed|sober|sobriety|addict\w*|relapse\w*)\b",
}
DESCRIPTORS = ["n_posts", "share_comment", "median_words", "mean_phq", "max_phq", "mean_liwc_health", "mean_liwc_anx",
               "mean_liwc_sad", "mean_ft_logit", "share_mental_health", "any_self_report", "share_life_events",
               "share_physical_health", "share_substances", "span_days"]


def load_runs(P, name):
    """One row per (repeat, user): label, alarm delay (0 = no alarm), peak probability and the fold's rule."""
    rows, probs = [], {}
    for r in range(REPEATS):
        for k in range(cv_splits.N_SPLITS):
            res = pickle.load(open(cv_splits.result_path(P, name, r, k), "rb"))
            d = se.delays(res["probs"], res["th"], [res["k"]])[:, 0]
            for u, y, p, dd in zip(res["users"], res["labels"], res["probs"], d):
                rows.append({"repeat": r, "fold": k, "user": u, "y": int(y), "delay": int(dd),
                             "peak": float(np.max(p)), "th": res["th"], "k": res["k"]})
                probs[(r, u)] = np.asarray(p)
    return pd.DataFrame(rows), probs


def majority(runs):
    """Per user: label, majority prediction over repeats, outcome (TP/FN/FP/TN), mean peak / threshold."""
    runs = runs.assign(pred=(runs["delay"] > 0).astype(int), ratio=runs["peak"] / runs["th"])
    g = runs.assign(wrong=(runs["pred"] != runs["y"]).astype(int)).groupby("user")
    out = pd.DataFrame({"y": g["y"].first(), "pred": (g["pred"].mean() > 0.5).astype(int),
                        "n_wrong": g["wrong"].sum(), "peak_over_th": g["ratio"].mean(), "delay": g["delay"].median()})
    out["outcome"] = np.select([(out.y == 1) & (out.pred == 1), out.y == 1, out.pred == 1], ["TP", "FN", "FP"], "TN")
    return out


def post_table(P):
    pooled = cv_all.load_pooled(P, False)
    posts = pooled[["pid", "user_id", "created_utc", "is_comment", "phq_score", "liwc_HEALTH", "liwc_ANX",
                    "liwc_SAD"]].copy()
    posts["text"] = cv_tfidf.pooled_text(P, pooled)
    posts["words"] = posts["text"].str.split().str.len()
    for key, pattern in LEXICON.items():
        posts[key] = posts["text"].str.contains(pattern, case=False, regex=True).astype(int)
    posts["ft_logit"] = held_out_ft_logit(P, posts)
    return posts


def held_out_ft_logit(P, posts):
    """ft_logit of every post from the adapters that did not see its user, averaged over the repeats."""
    splits = cv_splits.outer_splits(cv_splits.all_users(P), REPEATS)
    total = np.zeros(len(posts))
    for t, (_, te_users) in enumerate(splits):
        r, k = divmod(t, cv_splits.N_SPLITS)
        ft = pd.read_pickle(cv_splits.cvft_path(P, r, k)).set_index("pid")["ft_logit"]
        mask = posts["user_id"].isin(set(te_users)).to_numpy()
        total[mask] += posts.loc[mask, "pid"].map(ft).to_numpy()
    return total / REPEATS


def user_table(posts):
    g = posts.groupby("user_id")
    span = g["created_utc"].agg(lambda s: (s.max() - s.min()).total_seconds() / 86400)
    return pd.DataFrame({
        "n_posts": g.size(), "share_comment": g["is_comment"].mean(), "median_words": g["words"].median(),
        "mean_phq": g["phq_score"].mean(), "max_phq": g["phq_score"].max(), "mean_liwc_health": g["liwc_HEALTH"].mean(),
        "mean_liwc_anx": g["liwc_ANX"].mean(), "mean_liwc_sad": g["liwc_SAD"].mean(),
        "mean_ft_logit": g["ft_logit"].mean(),
        "share_mental_health": g["mental_health"].mean(), "any_self_report": (g["self_report"].sum() > 0).astype(int),
        "share_life_events": g["life_events"].mean(), "share_physical_health": g["physical_health"].mean(),
        "share_substances": g["substances"].mean(), "span_days": span})


def compare(table, outcome_col):
    """Median (rate for any_self_report) of every descriptor per outcome, and Mann-Whitney p of FN vs TP, FP vs TN."""
    rows = []
    for o in ("TP", "FN", "TN", "FP"):
        sub = table[table[outcome_col] == o]
        row = {"group": o, "n_users": len(sub)}
        for d in DESCRIPTORS:
            row[d] = sub[d].mean() if d == "any_self_report" else sub[d].median()
        rows.append(row)
    for a, b in (("FN", "TP"), ("FP", "TN")):
        row = {"group": f"p({a} vs {b})", "n_users": np.nan}
        for d in DESCRIPTORS:
            x, z = table.loc[table[outcome_col] == a, d], table.loc[table[outcome_col] == b, d]
            if d == "any_self_report":
                row[d] = stats.fisher_exact([[x.sum(), len(x) - x.sum()], [z.sum(), len(z) - z.sum()]])[1]
            else:
                row[d] = stats.mannwhitneyu(x, z).pvalue if len(x) and len(z) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def excerpts(posts, table, probs, path, n_top=3, width=280):
    """Local file for manual reading: for every error of MAIN, the posts with the highest held-out ft_logit and,
    for false positives, the post at which the alarm fired (repeat 0)."""
    by_user = posts.groupby("user_id")
    lines = [f"# Errors of {MAIN} (majority over {REPEATS} repeats); local only, contains Reddit text\n"]
    for outcome in ("FP", "FN"):
        lines.append(f"\n## {outcome}\n")
        for u, row in table[table[f"{MAIN}_outcome"] == outcome].sort_values("peak_over_th").iterrows():
            p = by_user.get_group(u).reset_index(drop=True)
            lines.append(f"\n### {u}: {len(p)} posts, peak/th {row['peak_over_th']:.2f}, wrong in {row['n_wrong']}/3, "
                         f"wrong in {row['n_models_wrong']}/{len(CONFIGS)} models, mental-health posts "
                         f"{row['share_mental_health']:.2f}, self-report {int(row['any_self_report'])}\n")
            show = list(p["ft_logit"].nlargest(n_top).index)
            show += [i for i in p.index[p["self_report"] == 1][:2] if i not in show]
            if outcome == "FP" and (0, u) in probs:
                d = int(se.delays([probs[(0, u)]], row["th0"], [int(row["k0"])])[0, 0])
                if d:
                    show = [d - 1] + [i for i in show if i != d - 1]
                    lines.append(f"- alarm (repeat 0) fired at post {d}\n")
            for i in show:
                t = " ".join(str(p.at[i, "text"]).split())[:width]
                lines.append(f"- post {i + 1}, ft_logit {p.at[i, 'ft_logit']:+.2f}, phq {p.at[i, 'phq_score']:.3f}: {t}\n")
    path.write_text("".join(lines), encoding="utf-8")


def rule_baseline(posts, labels):
    """No training: alarm at the first post that matches the self-report pattern; score = number of such posts."""
    pos = posts.groupby("user_id").cumcount() + 1
    hits = posts[posts["self_report"] == 1].assign(pos=pos)
    users, y = labels.index, labels.to_numpy()
    first = hits.groupby("user_id")["pos"].min().reindex(users).fillna(0).astype(int).to_numpy()
    count = hits.groupby("user_id").size().reindex(users).fillna(0).to_numpy()
    m = se.metrics_from_delays(first, y)
    row = {"rule": "first self-report", "roc": roc_auc_score(y, count), "pr": average_precision_score(y, count)}
    for n in (10, 100):
        early = hits[hits["pos"] <= n].groupby("user_id").size().reindex(users).fillna(0).to_numpy()
        row[f"auc{n}"] = roc_auc_score(y, early)
    row.update({k: m[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn", "median_delay", "f_latency")})
    return pd.DataFrame([row])


def figure(table, path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for outcome, color, off in (("FN", "#c0504d", -0.2), ("FP", "#3b6ea5", 0.2)):
        is_err = table[[f"{c}_outcome" for c in CONFIGS]].eq(outcome)
        counts = is_err.sum(axis=1)
        vals = [int((counts == n).sum()) for n in range(1, len(CONFIGS) + 1)]
        axes[0].bar(np.arange(1, len(CONFIGS) + 1) + off, vals, 0.4, color=color, label=outcome)
    axes[0].set_xlabel(f"number of the {len(CONFIGS)} models that err")
    axes[0].set_ylabel("users")
    axes[0].set_title("Overlap of errors across models", fontsize=10)
    axes[0].legend(fontsize=8)
    order = ["TP", "FN", "TN", "FP"]
    colors = ["#e6a5a3", "#c0504d", "#a9c1dc", "#3b6ea5"]
    for ax, col, title, log in ((axes[1], "n_posts", "Posts per user", True),
                                (axes[2], "share_mental_health", "Share of posts with mental-health terms", False),
                                (axes[3], "mean_ft_logit", "Mean held-out ft_logit per user", False)):
        data = [table.loc[table[f"{MAIN}_outcome"] == o, col] for o in order]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels([f"{o}\n(n={len(d)})" for o, d in zip(order, data)], fontsize=8)
        ax.set_title(f"{title}\n({MAIN})", fontsize=9)
        if log:
            ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(main_name=MAIN):
    global MAIN
    MAIN = main_name
    suffix = "" if MAIN == "ft_logit" else f"_{MAIN}"
    P = Paths()
    out_local = P.work / "error_analysis"
    out_local.mkdir(parents=True, exist_ok=True)
    out_docs = P.root / "docs" / "figures"
    posts = post_table(P)
    table = user_table(posts)
    probs_main = None
    for name in CONFIGS:
        runs, probs = load_runs(P, name)
        m = majority(runs)
        table[f"{name}_outcome"] = m["outcome"]
        if name == MAIN:
            probs_main = probs
            first = runs[runs.repeat == 0].set_index("user")
            table = table.join(m[["y", "n_wrong", "peak_over_th", "delay"]]).join(
                first[["th", "k"]].rename(columns={"th": "th0", "k": "k0"}))
    err = pd.DataFrame({c: table[f"{c}_outcome"].isin(["FN", "FP"]) for c in CONFIGS})
    table["n_models_wrong"] = err.sum(axis=1)

    summary = []
    for name in CONFIGS:
        o = table[f"{name}_outcome"]
        summary.append({"config": name, "FN": int((o == "FN").sum()), "FP": int((o == "FP").sum()),
                        "TP": int((o == "TP").sum()), "TN": int((o == "TN").sum())})
    summary = pd.DataFrame(summary)
    dep, ctl = table["y"] == 1, table["y"] == 0
    overlap = pd.DataFrame({
        "n_models_wrong": range(len(CONFIGS) + 1),
        "depressed_users": [int(((table.n_models_wrong == n) & dep).sum()) for n in range(len(CONFIGS) + 1)],
        "control_users": [int(((table.n_models_wrong == n) & ctl).sum()) for n in range(len(CONFIGS) + 1)]})
    groups = compare(table, f"{MAIN}_outcome")
    fn = table[table[f"{MAIN}_outcome"] == "FN"]
    fp = table[table[f"{MAIN}_outcome"] == "FP"]
    near = pd.DataFrame([{
        "FN_users": len(fn), "FN_peak_crossed_th": int((fn.peak_over_th >= 1).sum()),
        "FN_peak_over_th_median": fn.peak_over_th.median(),
        "FN_without_mental_health_terms": int((fn.share_mental_health == 0).sum()),
        "FN_with_self_report": int(fn.any_self_report.sum()),
        "FP_users": len(fp), "FP_alarm_delay_median": fp.delay.median(),
        "FP_with_self_report": int(fp.any_self_report.sum()),
        "controls_with_self_report": int(table.loc[ctl, "any_self_report"].sum()),
        "depressed_with_self_report": int(table.loc[dep, "any_self_report"].sum()),
        "FN_also_missed_by_ft_logit": int((fn["ft_logit_outcome"] == "FN").sum()),
        "FP_also_flagged_by_ft_logit": int((fp["ft_logit_outcome"] == "FP").sum())}])

    summary.to_csv(out_docs / f"error_counts{suffix}.csv", index=False)
    overlap.to_csv(out_docs / f"error_overlap{suffix}.csv", index=False)
    groups.to_csv(out_docs / f"error_groups{suffix}.csv", index=False, float_format="%.4g")
    near.to_csv(out_docs / f"error_misc{suffix}.csv", index=False, float_format="%.4g")
    rule = rule_baseline(posts, cv_splits.all_users(P))
    rule.to_csv(out_docs / "error_rule_baseline.csv", index=False, float_format="%.4g")
    table.to_csv(out_local / f"users{suffix}.csv", float_format="%.4g")
    excerpts(posts, table, probs_main, out_local / f"excerpts{suffix}.md")
    figure(table, out_docs / f"fig_errors{suffix}.png")
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(summary.to_string(index=False), overlap.to_string(index=False), groups.round(3).to_string(index=False),
          near.T.to_string(), rule.round(3).to_string(index=False), sep="\n\n")


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else MAIN)
