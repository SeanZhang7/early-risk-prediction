"""Paper figures (ACL column widths) from the aggregate tables in docs/figures/. Run: python paper/make_figures.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "docs" / "figures"
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})

LABELS = {"only_PHQ": "PHQ", "only_OWN_LIWC": "LIWC", "only_LSM": "LSM (counts)", "only_LSMX_no_par": "LSM alignment",
          "only_BEHAVIOR": "behaviour", "only_SIM": "cosine sim.", "context_only": "context only",
          "lean_PHQ_plus_LIWC": "PHQ + 3 LIWC", "mental_emb": "MentalRoBERTa frozen", "ft_logit": "MentalRoBERTa adapter",
          "tfidf_lr": "TF-IDF"}
COLORS = {"context_only": "#7f7f7f", "only_PHQ": "#1f77b4", "only_OWN_LIWC": "#2ca02c", "only_LSM": "#d62728",
          "only_LSMX_no_par": "#9467bd", "only_SIM": "#8c564b", "only_BEHAVIOR": "#e377c2",
          "lean_PHQ_plus_LIWC": "#17becf", "mental_emb": "#ff7f0e", "ft_logit": "#000000", "tfidf_lr": "#7fc97f"}
REFS = {"lean_PHQ_plus_LIWC", "mental_emb", "ft_logit", "tfidf_lr"}


def fig_curves():
    cur = pd.read_csv(FIG / "groups_curves.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35))
    titles = {"auc": "AUC after $n$ posts", "recall": "Depressed users flagged within $n$ posts",
              "fpr": "Control users flagged within $n$ posts"}
    for ax, metric in zip(axes, ("auc", "recall", "fpr")):
        for name, label in LABELS.items():
            d = cur[(cur.config == name) & (cur.metric == metric)]
            ax.plot(d["n"], d["mean"], "--" if name in REFS else "-", marker="o", ms=2, lw=1, color=COLORS[name],
                    label=label)
            ax.fill_between(d["n"], d["lo"], d["hi"], color=COLORS[name], alpha=0.07, lw=0)
        ax.set_xscale("log")
        ax.set_xlabel("posts read ($n$)")
        ax.set_title(titles[metric])
        ax.grid(alpha=0.3, lw=0.4)
    axes[0].axhline(0.5, color="grey", lw=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.12),
               columnspacing=1.0, handlelength=1.8)
    fig.tight_layout()
    fig.savefig(HERE / "fig_curves.pdf", bbox_inches="tight")
    plt.close(fig)


SHORT = {"only_PHQ": "PHQ", "only_OWN_LIWC": "LIWC", "only_LSM": "LSM", "only_LSMX_no_par": "LSM-al.",
         "only_BEHAVIOR": "Beh.", "lean_PHQ_plus_LIWC": "PHQ+L", "ft_logit": "Adapter", "tfidf_lr": "TF-IDF"}


def fig_errors():
    prof = pd.read_csv(FIG / "groups_errors.csv").set_index("config")
    order = list(SHORT)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55), gridspec_kw={"width_ratios": [1.0, 1.05, 1.05], "wspace": 0.55})
    y = np.arange(len(order))
    axes[0].barh(y - 0.2, prof.loc[order, "FN"], 0.4, color="#c0504d", label="false negatives (of 102)")
    axes[0].barh(y + 0.2, prof.loc[order, "FP"], 0.4, color="#3b6ea5", label="false positives (of 789)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABELS[c] for c in order])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("users")
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.22), fontsize=6)
    axes[0].set_title("Misclassified users")
    for ax, kind in zip(axes[1:], ("fn", "fp")):
        m = pd.read_csv(FIG / f"groups_overlap_{kind}.csv", index_col=0).loc[order, order]
        im = ax.imshow(m.to_numpy(), vmin=0, vmax=1, cmap="Blues")
        for i in range(len(order)):
            for j in range(len(order)):
                if i != j:
                    ax.text(j, i, f"{m.iat[i, j]:.2f}"[1:], ha="center", va="center", fontsize=4.5,
                            color="white" if m.iat[i, j] > 0.6 else "black")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([SHORT[c] for c in order], rotation=50, ha="right")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([SHORT[c] for c in order])
        ax.set_title("Overlap of false negatives" if kind == "fn" else "Overlap of false positives")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.03).set_label("Jaccard index")
    fig.savefig(HERE / "fig_errors.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_curves()
    fig_errors()
    print("saved")
