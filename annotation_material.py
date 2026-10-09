"""Blind reading material for the manual annotation of misclassified users (run after error_analysis.py).

    conda run -n hongyi python annotation_material.py

For every user in Results/v2/error_analysis/annotation/annotation_sheet.csv the material shows, in time order and
without any score or label: the alarm post of each model that flagged the user (repeat 0), the 3 posts with the highest
held-out adapter logit, the 3 posts most similar to the PHQ-9 items, up to 10 random posts matching a broad
mental-health keyword list (searched over the whole history) and 3 random other posts. The selection is not driven by
one model, so a statement the adapter scored low can still be found. Output stays in the git-ignored Results/ folder.
"""
import numpy as np
import pandas as pd

import error_analysis as ea
from paths import Paths

KEYWORDS = (r"\b(?:depress\w*|diagnos\w*|antidepress\w*|ssris?|zoloft|lexapro|prozac|sertraline|fluoxetine|"
            r"wellbutrin|escitalopram|therap\w*|psychiatr\w*|meds|medication|mental (?:health|illness)|suicid\w*|"
            r"mdd|dysthymi\w*|bipolar|self[- ]harm\w*)\b")
N_TOP, N_KEYWORD, N_RANDOM, WIDTH = 3, 10, 3, 1000
MODELS = {"FP_adapter": "ft_logit", "FP_74feat": "lean2_without_RE"}


def alarm_posts(P, users):
    """user -> 0-based indices of the alarm posts (repeat 0) of the models that flagged the user."""
    out = {}
    for name in set(MODELS.values()):
        runs, _ = ea.load_runs(P, name)
        r0 = runs[(runs.repeat == 0) & runs.user.isin(users) & (runs.delay > 0)]
        for u, d in zip(r0.user, r0.delay):
            out.setdefault(u, set()).add(int(d) - 1)
    return out


def main():
    P = Paths()
    folder = P.root / "Results" / "v2" / "error_analysis" / "annotation"
    sheet = pd.read_csv(folder / "annotation_sheet.csv", keep_default_na=False)
    posts = ea.post_table(P)
    alarms = alarm_posts(P, set(sheet.user_id[sheet.kind == "FP"]))
    by_user = posts.groupby("user_id")
    rng = np.random.default_rng(0)
    lines = ["# Annotation material (local only, contains Reddit text). Posts in time order; no scores or labels.\n"]
    index = []
    for u, kind in zip(sheet.user_id, sheet.kind):
        p = by_user.get_group(u).reset_index(drop=True)
        chosen = {}
        for i in sorted(alarms.get(u, set())) if kind == "FP" else []:
            chosen.setdefault(i, []).append("alarm")
        for col, tag in (("ft_logit", "top_adapter"), ("phq_score", "top_phq")):
            for i in p[col].nlargest(N_TOP).index:
                chosen.setdefault(i, []).append(tag)
        hits = p.index[p["text"].str.contains(KEYWORDS, case=False, regex=True)].to_numpy()
        for i in rng.permutation(hits)[:N_KEYWORD]:
            chosen.setdefault(int(i), []).append("keyword")
        rest = np.setdiff1d(p.index.to_numpy(), list(chosen))
        for i in rng.permutation(rest)[:N_RANDOM]:
            chosen.setdefault(int(i), []).append("random")
        lines.append(f"\n## {u} ({kind}, {len(p)} posts, {len(hits)} with keywords)\n")
        for i in sorted(chosen):
            text = " ".join(str(p.at[i, "text"]).split())
            cut = "..." if len(text) > WIDTH else ""
            lines.append(f"- post {i + 1}: {text[:WIDTH]}{cut}\n")
            index.append({"user_id": u, "post": i + 1, "sources": ";".join(chosen[i])})
    (folder / "material.md").write_text("".join(lines), encoding="utf-8")
    pd.DataFrame(index).to_csv(folder / "material_index.csv", index=False)
    print(f"{len(sheet)} users, {len(index)} posts -> {folder / 'material.md'}")


if __name__ == "__main__":
    main()
