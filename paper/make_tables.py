"""Generate the LaTeX tables of the paper from the aggregate result files. Run: python paper/make_tables.py"""
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from feature_cols import EXPERIMENTS_V2  # noqa: E402

summary = pd.read_csv(ROOT / "models" / "v2" / "cv_all" / "summary.csv").set_index("config")
pairs = pd.read_csv(ROOT / "models" / "v2" / "cv_all" / "pairs.csv")
curves = pd.read_csv(ROOT / "docs" / "figures" / "groups_curves.csv")
delay = pd.read_csv(ROOT / "docs" / "figures" / "groups_delay.csv").set_index("config")
errors = pd.read_csv(ROOT / "docs" / "figures" / "groups_errors.csv").set_index("config")
rule = pd.read_csv(ROOT / "docs" / "figures" / "error_rule_baseline.csv").iloc[0]

NAMES = {"only_PHQ": "PHQ", "only_OWN_LIWC": "LIWC (15 categories)", "only_LSM": "LSM (counts)",
         "only_LSMX_no_par": "LSM alignment", "only_BEHAVIOR": "Behaviour", "only_SIM": "Cosine similarity",
         "context_only": "Context only (control)", "lean_PHQ_plus_LIWC": "PHQ + 3 LIWC categories",
         "lean_without_RE": "PHQ + 3 LIWC + 4 LSM scores", "lean2_without_RE": "PHQ + 3 LIWC + LSM alignment",
         "mental_emb": "MentalRoBERTa, frozen", "ft_logit": "MentalRoBERTa, adapter-tuned", "tfidf_lr": "TF-IDF + logistic regression"}
GROUPS = ["only_PHQ", "only_OWN_LIWC", "only_LSM", "only_LSMX_no_par", "only_BEHAVIOR", "only_SIM", "context_only"]
COMBOS = ["lean_PHQ_plus_LIWC", "lean_without_RE", "lean2_without_RE", "mental_emb", "ft_logit", "tfidf_lr"]
SHORT = {"only_PHQ": "PHQ", "only_OWN_LIWC": "LIWC", "only_LSM": "LSM (counts)", "only_LSMX_no_par": "LSM alignment",
         "only_BEHAVIOR": "Behaviour", "only_SIM": "Cosine sim.", "context_only": "Context only",
         "lean_PHQ_plus_LIWC": "PHQ + 3 LIWC", "lean_without_RE": "PHQ + LIWC + LSM", "lean2_without_RE": "PHQ + LIWC + LSM-al.",
         "mental_emb": "MentalRoBERTa frozen", "ft_logit": "MentalRoBERTa adapter", "tfidf_lr": "TF-IDF",
         "ft_logit_plus_lean2_without_RE": "Adapter + 74 feat.", "mental_plus_lean2_without_RE": "Frozen + 74 feat."}


def f3(x):
    return f"{x:.3f}"[1:] if x < 1 else f"{x:.3f}"


def ci(c, m):
    r = summary.loc[c]
    return f"{f3(r[m])} [{f3(r[m + '_lo'])}, {f3(r[m + '_hi'])}]"


def auc(c, n):
    return curves[(curves.config == c) & (curves.metric == "auc") & (curves.n == n)]["mean"].iloc[0]


def rec(c, n):
    return curves[(curves.config == c) & (curves.metric == "recall") & (curves.n == n)]["mean"].iloc[0]


def results():
    rows = [r"\toprule", r"Configuration & \#feat. & ROC-AUC & PR-AUC & AUC@10 & F1 & F$_{\mathrm{lat}}$ \\", r"\midrule",
            r"\multicolumn{7}{l}{\textit{Single feature groups (plus three context features)}} \\"]
    for c in GROUPS:
        rows.append(f"{NAMES[c]} & {len(EXPERIMENTS_V2[c])} & {ci(c, 'roc')} & {ci(c, 'pr')} & {f3(summary.loc[c, 'auc10'])} & "
                    f"{f3(summary.loc[c, 'f1'])} & {f3(summary.loc[c, 'f_latency'])} \\\\")
    rows += [r"\midrule", r"\multicolumn{7}{l}{\textit{Combinations and reference systems}} \\"]
    for c in COMBOS:
        nf = "--" if c == "tfidf_lr" else {"mental_emb": "771", "ft_logit": "4"}.get(c, str(len(EXPERIMENTS_V2[c])))
        rows.append(f"{NAMES[c]} & {nf} & {ci(c, 'roc')} & {ci(c, 'pr')} & {f3(summary.loc[c, 'auc10'])} & "
                    f"{f3(summary.loc[c, 'f1'])} & {f3(summary.loc[c, 'f_latency'])} \\\\")
    rows.append(f"First self-report phrase (no learning) & -- & {f3(rule['roc'])} & {f3(rule['pr'])} & {f3(rule['auc10'])} & "
                f"{f3(rule['f1'])} & {f3(rule['f_latency'])} \\\\")
    rows.append(r"\bottomrule")
    return "\n".join(rows)


def early():
    rows = [r"\toprule", r"Configuration & AUC@1 & AUC@10 & Flagged $\leq$10 & Precision & Median delay [Q1--Q3] \\", r"\midrule"]
    for c in GROUPS + ["lean_PHQ_plus_LIWC", "mental_emb", "ft_logit", "tfidf_lr"]:
        d = delay.loc[c]
        rows.append(f"{SHORT[c]} & {f3(auc(c, 1))} & {f3(auc(c, 10))} & {f3(rec(c, 10))} & {f3(d.precision)} & "
                    f"{d.median_delay:.1f} [{d.delay_q1:.0f}--{d.delay_q3:.0f}] \\\\")
    rows.append(f"First self-report phrase & -- & {f3(rule['auc10'])} & -- & {f3(rule['precision'])} & "
                f"{rule['median_delay']:.0f} [--] \\\\")
    rows.append(r"\bottomrule")
    return "\n".join(rows)


def errs():
    rows = [r"\toprule", r"& & & \multicolumn{2}{c}{False negatives} & \multicolumn{2}{c}{False positives} & \multicolumn{2}{c}{Recall by self-report} \\",
            r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
            r"Configuration & FN & FP & Posts & SR & Posts & SR & with & without \\", r"\midrule"]
    for c in ["only_PHQ", "only_OWN_LIWC", "only_LSM", "only_LSMX_no_par", "only_BEHAVIOR", "lean_PHQ_plus_LIWC", "ft_logit", "tfidf_lr"]:
        e = errors.loc[c]
        rows.append(f"{SHORT[c]} & {int(e.FN)} & {int(e.FP)} & {e.FN_median_posts:.0f} & {e.FN_self_report * 100:.0f}\\% & "
                    f"{e.FP_median_posts:.0f} & {e.FP_self_report * 100:.0f}\\% & {e.recall_with_self_report:.2f} & "
                    f"{e.recall_without_self_report:.2f} \\\\")
    rows.append(r"\bottomrule")
    return "\n".join(rows)


def pval(p):
    if p >= 0.9995:
        return "1"
    return "$<$.001" if p < 0.001 else f"{p:.3f}"[1:]


def pairtable():
    rows = [r"\toprule",
            r"Fam. & A $-$ B & $\Delta$ROC & Holm & NB & DeLong & $\Delta$PR & Holm & NB & $\Delta$F1 & Holm & McNemar \\", r"\midrule"]
    for _, r in pairs.iterrows():
        a, b = SHORT[r.A], SHORT[r.B]
        rows.append(f"{int(r.family)} & {a} $-$ {b} & {r.d_roc:+.3f} & {pval(r.p_roc_holm)} & {pval(r.nb_p_roc)} & "
                    f"{pval(r.delong_p_median)} & {r.d_pr:+.3f} & {pval(r.p_pr_holm)} & {pval(r.nb_p_pr)} & "
                    f"{r.d_f1:+.3f} & {pval(r.p_f1_holm)} & {pval(r.mcnemar_p_median)} \\\\")
    rows.append(r"\bottomrule")
    return "\n".join(rows)


def keytable():
    rows = [r"\toprule", r"Comparison (A $-$ B) & $\Delta$ROC & $\Delta$PR & $\Delta$F1 \\", r"\midrule"]
    spec = [("only_PHQ", "only_OWN_LIWC", "PHQ $-$ LIWC", 1), ("only_PHQ", "only_LSM", "PHQ $-$ LSM (counts)", 1),
            ("only_PHQ", "only_LSMX_no_par", "PHQ $-$ LSM alignment", 1),
            ("lean2_without_RE", "only_PHQ", "74 feat. $-$ PHQ", 1),
            ("lean2_without_RE", "lean_without_RE", "74 feat. $-$ PHQ + LIWC + LSM scores", 1),
            ("ft_logit", "mental_emb", "Adapter $-$ frozen encoder", 1), ("ft_logit", "tfidf_lr", "Adapter $-$ TF-IDF", 1),
            ("lean2_without_RE", "ft_logit", "Adapter $-$ 74 feat.", -1)]
    for a, b, label, sign in spec:
        r = pairs[(pairs.A == a) & (pairs.B == b)].iloc[0]
        cells = [f"{sign * r['d_' + m]:+.3f} ({pval(r['p_' + m + '_holm'])})" for m in ("roc", "pr", "f1")]
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    rows.append(r"\bottomrule")
    return "\n".join(rows)


def lengthtable():
    d = pd.read_csv(ROOT / "docs" / "figures" / "groups_length_bins.csv")
    bins = ["<100", "100-499", ">=500"]
    n = d[d.config == "only_PHQ"].set_index("posts")

    def lab(b):
        return b.replace(">=", "$\\geq$").replace("-", "--") + f" ({int(n.loc[b, 'n_depressed'])}/{int(n.loc[b, 'n_control'])})"

    rows = [r"\toprule", "Posts per user & " + " & ".join(lab(b) for b in bins) + " \\\\", r"\midrule"]
    for c in ["only_PHQ", "only_OWN_LIWC", "only_LSM", "only_LSMX_no_par", "only_BEHAVIOR", "lean_PHQ_plus_LIWC", "mental_emb",
              "ft_logit", "tfidf_lr"]:
        x = d[d.config == c].set_index("posts")
        rows.append(f"{SHORT[c]} & " + " & ".join(f"{f3(x.loc[b, 'recall'])} ({f3(x.loc[b, 'false_alarm_rate'])})" for b in bins) + " \\\\")
    rows.append(r"\bottomrule")
    return "\n".join(rows)


SPECS = {"tab_results": "lrccccc", "tab_early": "lccccc", "tab_errors": "lccccccrr", "tab_pairs": "llrcccrccrcc", "tab_key": "lccc", "tab_length": "lccc"}


if __name__ == "__main__":
    for name, text in (("tab_results", results()), ("tab_early", early()), ("tab_errors", errs()), ("tab_pairs", pairtable()), ("tab_key", keytable()), ("tab_length", lengthtable())):
        body = "\\begin{tabular}{" + SPECS[name] + "}\n" + text + "\n\\end{tabular}\n"
        (HERE / f"{name}.tex").write_text(body, encoding="utf-8")
        print("wrote", name)
