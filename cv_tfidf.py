"""Bag-of-words baseline for the repeated cross-validation of cv_all.py (same outer/inner splits, same decision rule).

    nohup conda run -n hongyi python -u cv_tfidf.py --repeats 3 > Results/v2/cv_tfidf.log 2>&1 &

A TF-IDF logistic regression scores every post (label = the user's label, each user weighted equally and the classes
balanced). The user's score after n posts is the running mean of the post logits, so the baseline is causal and
produces a probability per post like the sequence models. Inside each outer fold the inner 5-fold models give
out-of-fold scores for the training users (these choose (th, k)) and an averaged score for the held-out users.
The vocabulary and IDF weights are fitted on the outer training users' text only (no labels involved).
Results: models/<version>/cv_all/tfidf_lr/r<r>_k<k>.pkl (same format as cv_all.py).
"""
import argparse
import pickle
import time

import numpy as np
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import cv_all
import cv_splits
import data_prep as dp
import experiment as ex
import seq_eval as se
from paths import Paths

NAME = "tfidf_lr"


def pooled_text(P, pooled):
    """Post body for every row of the pooled feature frame (matched on the unique post id `pid`)."""
    offset = int(dp.read_table(P.split_csv("train"), usecols=["row_id"])["row_id"].max()) + 1
    bodies = {}
    for split, shift in (("train", 0), ("test", offset)):
        t = dp.read_table(P.split_csv(split), usecols=["row_id", "body"])
        bodies.update(zip(t["row_id"].to_numpy() + shift, t["body"].astype(str)))
    return np.array([bodies[p] for p in pooled["pid"].to_numpy()], dtype=object)


def user_weights(user_codes, y_post):
    """Equal total weight per user, positive and negative users balanced, mean weight 1."""
    n_user = np.bincount(user_codes)
    w = 1.0 / n_user[user_codes]
    users_pos = len(np.unique(user_codes[y_post == 1]))
    users_neg = len(np.unique(user_codes[y_post == 0]))
    w = w * np.where(y_post == 1, users_neg / max(users_pos, 1), 1.0)
    return w / w.mean()


def running_mean_logit(logits, user_codes, n_users):
    """Probability sequences per user: sigmoid of the running mean of the post logits (rows are time-ordered)."""
    out = [None] * n_users
    order = np.argsort(user_codes, kind="stable")
    bounds = np.flatnonzero(np.diff(user_codes[order])) + 1
    for rows in np.split(order, bounds):
        z = logits[rows]
        out[user_codes[rows[0]]] = expit(np.cumsum(z) / np.arange(1, len(z) + 1))
    return out


def run_outer(texts, users, y_post, tr_users, te_users, seed, cfg, C, max_features):
    codes = {u: i for i, u in enumerate(np.unique(users))}
    uc = np.array([codes[u] for u in users])
    y_user = np.zeros(len(codes), dtype=int)
    y_user[uc] = y_post
    tr_codes = np.array([codes[u] for u in tr_users])
    te_codes = np.array([codes[u] for u in te_users])
    in_tr = np.isin(uc, tr_codes)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=max_features, sublinear_tf=True, dtype=np.float32)
    vec.fit(texts[in_tr])
    X = vec.transform(texts)

    inner = StratifiedKFold(cfg.n_folds, shuffle=True, random_state=seed)
    oof_logit = np.zeros(len(texts))
    te_rows = np.isin(uc, te_codes)
    te_logit = np.zeros(te_rows.sum())
    for a, held in inner.split(np.zeros(len(tr_codes)), y_user[tr_codes]):
        fit_rows = np.isin(uc, tr_codes[a])
        held_rows = np.isin(uc, tr_codes[held])
        clf = LogisticRegression(C=C, solver="liblinear", max_iter=200)
        clf.fit(X[fit_rows], y_post[fit_rows], sample_weight=user_weights(uc[fit_rows], y_post[fit_rows]))
        oof_logit[held_rows] = clf.decision_function(X[held_rows])
        te_logit += clf.decision_function(X[te_rows]) / cfg.n_folds

    full = np.zeros(len(texts))
    full[te_rows] = te_logit
    oof_probs = running_mean_logit(oof_logit, uc, len(codes))
    te_probs = running_mean_logit(full, uc, len(codes))
    table = se.sweep([oof_probs[c] for c in tr_codes], y_user[tr_codes], cfg.thresholds, cfg.ks)
    op = se.select_operating_point(table, cfg.metric)
    return {"users": np.array(te_users), "labels": y_user[te_codes], "probs": [te_probs[c] for c in te_codes],
            "th": float(op["th"]), "k": int(op["k"]), "oof_f_latency": float(op["f_latency"]), "C": C}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--max-features", type=int, default=200000)
    ap.add_argument("--only", nargs=2, type=int, default=None, metavar=("R", "K"), help="run a single outer fold")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    P = Paths()
    cfg = ex.Config()
    labels = cv_splits.all_users(P)
    splits = cv_splits.outer_splits(labels, args.repeats)
    pooled = cv_all.load_pooled(P, False)[["pid", "user_id", "Depression"]]
    texts = pooled_text(P, pooled)
    users, y_post = pooled["user_id"].to_numpy(), pooled["Depression"].to_numpy().astype(int)
    print(f"{len(labels)} users, {len(texts)} posts, {len(splits)} outer folds", flush=True)
    for t, (tr_users, te_users) in enumerate(splits):
        r, k = divmod(t, cv_splits.N_SPLITS)
        if args.only and (r, k) != tuple(args.only):
            continue
        path = cv_splits.result_path(P, NAME, r, k)
        if path.exists() and not args.force:
            continue
        t0 = time.perf_counter()
        res = run_outer(texts, users, y_post, tr_users, te_users, 7 + 1000 * r + k, cfg, args.C, args.max_features)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(res, f)
        print(f"[{NAME}] r{r} k{k}: rule th={res['th']} k={res['k']} oof F_lat={res['oof_f_latency']:.3f} "
              f"in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
