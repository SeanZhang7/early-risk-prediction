"""Why are depressed users missed? Mechanism analysis of the false negatives annotated by hand.

    conda run -n hongyi python fn_analysis.py

Reads Results/v2/error_analysis/annotation/annotation_sheet_annotated.csv (local, git-ignored) and the per-post
probabilities of the repeated cross-validation. For every annotated false negative and every model it reports how close
the sequence probability came to the alarm rule (never reached the threshold / reached it but not for k consecutive
posts / alarm in a minority of repeats), and how the probability reacts to posts with a disclosure phrase. True positives
are summarised the same way as a reference. Aggregate tables only go to docs/figures/.
"""
import numpy as np
import pandas as pd

import error_analysis as ea
from paths import Paths

MODELS = {"adapter": "ft_logit", "74feat": "lean2_without_RE"}
FN_SET = {"adapter": "FN_adapter", "74feat": "FN_74feat_only"}
DISCLOSURE = (r"\b(?:i (?:have|had|was diagnosed with|got diagnosed with|suffer from|struggle with|deal with|am dealing with|"
              r"have been dealing with|battle) (?:\w+ ){0,2}depression|my (?:\w+ )?depression|i'?m depressed|i am depressed|"
              r"i'?ve been depressed|diagnosed (?:me )?with (?:\w+ ){0,3}(?:depression|mdd|dysthymia)|"
              r"i (?:have|had) (?:mdd|dysthymia))\b")
WINDOW = 20


def longest_run(above):
    best = run = 0
    for a in above:
        run = run + 1 if a else 0
        best = max(best, run)
    return best


def mechanism(p, th, k, alarmed):
    if alarmed:
        return "alarm"
    if p.max() < th:
        return "never reached threshold"
    return "reached threshold, run shorter than k"


def main():
    P = Paths()
    folder = P.root / "Results" / "v2" / "error_analysis"
    ann = pd.read_csv(folder / "annotation" / "annotation_sheet_annotated.csv", keep_default_na=False)
    ann["statement"] = ann["fn_first_person_statement"].str.strip().str.lower()
    fn_users = ann[ann.kind == "FN"].set_index("user_id")
    posts = ea.post_table(P)
    posts["disclosure"] = posts["text"].str.contains(DISCLOSURE, case=False, regex=True)
    by_user = {u: g.reset_index(drop=True) for u, g in posts.groupby("user_id")}
    users = pd.read_csv(folder / "users.csv").set_index("user_id")

    rows, resp = [], []
    for short, name in MODELS.items():
        runs, probs = ea.load_runs(P, name)
        outcome = users[f"{name}_outcome"]
        own_fn = list(fn_users.index[fn_users.sets.str.contains(FN_SET[short])])
        for u, group in [(u, "FN") for u in own_fn] + [(u, "TP") for u in outcome.index[outcome == "TP"]]:
            g = by_user[u]
            for r in range(ea.REPEATS):
                p = probs[(r, u)]
                if len(p) != len(g):
                    raise ValueError(f"{u}: {len(p)} probabilities for {len(g)} posts")
                row = runs[(runs.repeat == r) & (runs.user == u)].iloc[0]
                th, k = float(row.th), int(row.k)
                rows.append({"model": short, "user_id": u, "group": group, "repeat": r, "n_posts": len(g),
                             "peak_over_th": p.max() / th, "longest_run": longest_run(p >= th), "k": k,
                             "mechanism": mechanism(p, th, k, row.delay > 0),
                             "disclosure_posts": int(g.disclosure.sum())})
                for t in np.flatnonzero(g.disclosure.to_numpy()):
                    after = p[t:t + WINDOW]
                    resp.append({"model": short, "user_id": u, "group": group, "repeat": r, "post": t + 1,
                                 "rel_position": t / len(g), "before_over_th": p[t - 1] / th if t else np.nan,
                                 "at_over_th": p[t] / th, "max_after_over_th": after.max() / th,
                                 "reaches_th_within_window": bool((after >= th).any()),
                                 "ft_logit_pct": (g.ft_logit < g.ft_logit[t]).mean()})
    rows, resp = pd.DataFrame(rows), pd.DataFrame(resp)

    own = rows[rows.group == "FN"].merge(fn_users[["statement"]], left_on="user_id", right_index=True)
    per_user = (own.groupby(["model", "user_id", "statement", "n_posts", "disclosure_posts"])
                .agg(peak_over_th=("peak_over_th", "mean"),
                     mechanism=("mechanism", lambda m: m.value_counts().index[0]),
                     alarms=("mechanism", lambda m: int((m == "alarm").sum())))
                .reset_index())
    per_user.to_csv(folder / "fn_mechanism_users.csv", index=False, float_format="%.3g")

    summary = per_user.groupby(["model", "mechanism"]).size().unstack(fill_value=0)
    near = per_user.groupby("model").agg(users=("user_id", "size"), median_peak_over_th=("peak_over_th", "median"),
                                         alarm_in_1_of_3=("alarms", lambda a: int((a == 1).sum())),
                                         median_posts=("n_posts", "median"),
                                         with_disclosure_phrase=("disclosure_posts", lambda d: int((d > 0).sum())))
    react = (resp.groupby(["model", "group"])
             .agg(disclosure_posts=("post", "size"), users=("user_id", "nunique"),
                  median_ft_logit_pct=("ft_logit_pct", "median"),
                  median_before_over_th=("before_over_th", "median"), median_at_over_th=("at_over_th", "median"),
                  median_max_after_over_th=("max_after_over_th", "median"),
                  share_reaching_th_within_window=("reaches_th_within_window", "mean"),
                  median_rel_position=("rel_position", "median")))
    out = P.root / "docs" / "figures"
    summary.to_csv(out / "fn_mechanism_summary.csv")
    react.to_csv(out / "fn_disclosure_response.csv", float_format="%.3f")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(summary, near.round(3), react.round(3), per_user.round(2).to_string(index=False), sep="\n\n")


if __name__ == "__main__":
    main()
