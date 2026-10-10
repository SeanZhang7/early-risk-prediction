"""Recompute the numbers quoted in the prose of paper/main.tex: `conda run -n hongyi python verify_claims.py`."""
import numpy as np
import pandas as pd

import error_analysis as ea
from paths import Paths

P = Paths()
S = pd.read_csv("models/v2/cv_all/summary.csv").set_index("config")
PR = pd.read_csv("models/v2/cv_all/pairs.csv")
CU = pd.read_csv("docs/figures/groups_curves.csv")
DL = pd.read_csv("docs/figures/groups_delay.csv").set_index("config")
LB = pd.read_csv("docs/figures/groups_length_bins.csv")


def pair(a, b):
    return PR[(PR.A == a) & (PR.B == b)].iloc[0]


def cur(c, m, n):
    return CU[(CU.config == c) & (CU.metric == m) & (CU.n == n)]["mean"].iloc[0]


print("== Data table")
posts = ea.post_table(P)
labels = pd.read_csv("Results/v2/posts.csv", usecols=["user_id", "Depression"]).drop_duplicates().set_index("user_id")["Depression"]
posts["y"] = posts.user_id.map(labels)
pooled = __import__("cv_all").load_pooled(P, False)
posts["has_parent"] = pooled["has_parent"].to_numpy()
for y in (1, 0):
    d = posts[posts.y == y]
    n = d.groupby("user_id").size()
    span = d.groupby("user_id").created_utc.agg(lambda s: (s.max() - s.min()).total_seconds() / 86400)
    sr = d.groupby("user_id").self_report.max()
    print(y, "users", n.size, "posts", len(d), "median [q1,q3]", n.median(), n.quantile([.25, .75]).tolist(),
          "words median", d.words.median(), "comments %.1f" % (100 * d.is_comment.mean()),
          "parent %.1f" % (100 * d.has_parent.mean()), "span", span.median(), "SR", int(sr.sum()), "%.1f" % (100 * sr.mean()))
print("total posts", len(posts), "parent known overall %.1f" % (100 * posts.has_parent.mean()))

print("== RQ1 ranking (roc, pr)")
for c in ["only_PHQ", "only_OWN_LIWC", "only_LSM", "only_LSMX_no_par", "only_BEHAVIOR", "only_SIM", "context_only", "ft_logit"]:
    print(c, S.loc[c, ["roc", "pr", "auc10", "f1"]].round(3).tolist())
for a, b in [("only_PHQ", "only_OWN_LIWC"), ("only_PHQ", "only_LSM"), ("only_PHQ", "only_LSMX_no_par")]:
    r = pair(a, b)
    print(a, b, "dROC %.3f dPR %.3f holm roc %.3f pr %.3f nb roc %.3f nb pr %.3f mcn %.3f" % (
        r.d_roc, r.d_pr, r.p_roc_holm, r.p_pr_holm, r.nb_p_roc, r.nb_p_pr, r.mcnemar_p_median), "dF1 %.3f" % r.d_f1)
print("-- each group vs context_only")
for _, r in PR[PR.B == "context_only"].iterrows():
    print(r.A, "dROC %.3f holm %.3f nb %.3f | dPR %.3f holm %.3f nb %.3f" % (r.d_roc, r.p_roc_holm, r.nb_p_roc, r.d_pr, r.p_pr_holm, r.nb_p_pr))
print("-- PHQ vs others NB roc:", [(r.B, round(r.nb_p_roc, 3)) for _, r in PR[PR.A == "only_PHQ"].iterrows()])

print("== RQ2")
for c in ["only_OWN_LIWC", "only_PHQ", "only_BEHAVIOR", "only_LSM", "only_LSMX_no_par", "only_SIM", "mental_emb", "ft_logit", "tfidf_lr"]:
    print(c, "auc@1 %.3f auc@10 %.3f auc@100 %.3f" % (cur(c, "auc", 1), cur(c, "auc", 10), cur(c, "auc", 100)),
          "fpr@500 %.3f" % cur(c, "fpr", 500), "prec %.3f delay %.2f within10 %.2f" % (DL.loc[c, "precision"], DL.loc[c, "median_delay"], cur(c, "recall", 10)))
print("context fpr@500 %.3f" % cur("context_only", "fpr", 500))
print("-- delay tests")
for _, r in PR[PR.p_median_delay.notna()].iterrows():
    print(r.A, "-", r.B, "d_delay %.2f p %.4f holm %s" % (r.d_median_delay, r.p_median_delay, r.get("p_median_delay_holm")))

print("== RQ3")
for a, b in [("lean2_without_RE", "only_PHQ"), ("lean_PHQ_plus_LIWC", "only_PHQ"), ("lean_without_RE", "only_PHQ"),
             ("lean_without_RE", "lean2_without_RE"), ("ft_logit", "tfidf_lr"), ("ft_logit", "mental_emb"), ("ft_logit", "lean2_without_RE"),
             ("ft_logit_plus_lean2_without_RE", "ft_logit")]:
    m = PR[(PR.A == a) & (PR.B == b)]
    if m.empty:
        m = PR[(PR.A == b) & (PR.B == a)]
        print("(reversed)", end=" ")
    if m.empty:
        print(a, b, "not in pairs.csv")
        continue
    r = m.iloc[0]
    print(r.A, "-", r.B, "dROC %.3f dPR %.3f dF1 %.3f | p roc %.3f pr %.3f f1 %.3f | holm roc %.3f pr %.3f f1 %s | nb pr %.3f" % (
        r.d_roc, r.d_pr, r.d_f1, r.p_roc, r.p_pr, r.p_f1, r.p_roc_holm, r.p_pr_holm, r.get("p_f1_holm"), r.nb_p_pr))
print("rule baseline", pd.read_csv("docs/figures/error_rule_baseline.csv").round(3).to_dict("records"))

print("== RQ4")
groups = {"context_only": 0, "only_PHQ": 0, "only_OWN_LIWC": 0, "only_BEHAVIOR": 0, "only_LSMX_no_par": 0}
for c in ["only_BEHAVIOR", "ft_logit", "only_PHQ"]:
    runs, probs = ea.load_runs(P, c)
    m = ea.majority(runs)
    u = posts.groupby("user_id").size()
    tn = m[m.outcome == "TN"].index
    tp = m[m.outcome == "TP"].index
    print(c, "TN median posts", u.loc[tn].median(), "TP median", u.loc[tp].median(), "n", m.outcome.value_counts().to_dict())
ov_fn = pd.read_csv("docs/figures/groups_overlap_fn.csv", index_col=0)
ov_fp = pd.read_csv("docs/figures/groups_overlap_fp.csv", index_col=0)
print("FN jaccard PHQ/PHQ+L %.2f PHQ/LIWC %.2f PHQ/LSMX %.2f PHQ/adapter %.2f" % (
    ov_fn.loc["only_PHQ", "lean_PHQ_plus_LIWC"], ov_fn.loc["only_PHQ", "only_OWN_LIWC"], ov_fn.loc["only_PHQ", "only_LSMX_no_par"], ov_fn.loc["only_PHQ", "ft_logit"]))
off = ov_fp.where(~np.eye(len(ov_fp), dtype=bool))
print("FP jaccard between groups: min %.2f max %.2f; LSM/LSMX %.2f" % (off.min().min(), off.max().max(), ov_fp.loc["only_LSM", "only_LSMX_no_par"]))
print(ov_fp.round(2).to_string())

print("-- length bins")
print(LB[LB.config.isin(["tfidf_lr", "mental_emb", "ft_logit", "only_PHQ", "only_OWN_LIWC"])].round(2).to_string())

print("-- adapter FN vs TP descriptors")
users = pd.read_csv("Results/v2/error_analysis/users.csv", index_col=0)
col = "ft_logit_outcome"
for o in ("FN", "TP"):
    sub = users[users[col] == o]
    print(o, len(sub), "posts median", sub.n_posts.median(), "MH share median %.3f" % sub.share_mental_health.median())
from scipy import stats
a, b = users[users[col] == "FN"].n_posts, users[users[col] == "TP"].n_posts
print("MW posts p %.3f" % stats.mannwhitneyu(a, b).pvalue,
      "MH p %.2g" % stats.mannwhitneyu(users[users[col] == "FN"].share_mental_health, users[users[col] == "TP"].share_mental_health).pvalue)

print("-- alarm post of false positives: percentile of held-out ft_logit within the user's own posts (repeat 0)")
import seq_eval as se
for c in ["ft_logit", "lean2_without_RE"]:
    runs, probs = ea.load_runs(P, c)
    m = ea.majority(runs)
    by_user = {u: g.reset_index(drop=True) for u, g in posts.groupby("user_id")}
    pct, delays, rel = [], [], []
    for u in m.index[m.outcome == "FP"]:
        r = runs[(runs.repeat == 0) & (runs.user == u)].iloc[0]
        if r.delay <= 0:
            continue
        g = by_user[u]
        pct.append((g.ft_logit < g.ft_logit[int(r.delay) - 1]).mean())
        delays.append(r.delay)
        rel.append(r.delay / len(g))
    print(c, "FP with alarm in repeat 0:", len(pct), "median percentile of alarm post %.2f" % np.median(pct),
          "share above user's median: %.2f" % np.mean(np.array(pct) > .5), "median delay", np.median(delays), "median alarm position/history %.2f" % np.median(rel),
          "share in top 10%%: %.2f" % np.mean(np.array(pct) > .9), "within 30 posts: %d" % sum(d <= 30 for d in delays),
          "after 100: %d" % sum(d > 100 for d in delays))
