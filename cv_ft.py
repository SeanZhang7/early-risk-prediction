"""Adapter fine-tuning of MentalRoBERTa inside the repeated cross-validation (nested, leak-free).

    python cv_ft.py --repeat 0 --outer 0 --device cuda:0

For outer fold (r, k) the users are split into outer-train and outer-test users. The outer-train users are cut
into 3 inner folds; one adapter is trained on each pair of inner folds. The posts of an outer-train user get the
logit of the model that never saw the user, the posts of the outer-test users the mean logit of the 3 models. The
result (parts/cvft_r<r>_k<k>.pkl, one row per post of every user, key `pid`) is the `ft_logit` feature that
cv_all.py feeds to the sequence model for that outer fold, exactly as in the single-split experiment.
"""
import argparse
import json
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

import cv_splits
import data_prep as dp
import finetune_mental as fm
from paths import Paths

INNER_FOLDS = 3


def pooled_posts(P):
    """Posts of all users with a unique id `pid`: row_id for train, row_id + n_train for test."""
    cols = ["row_id", "user_id", "Depression", "body"]
    tr = dp.read_table(P.split_csv("train"), usecols=cols)
    te = dp.read_table(P.split_csv("test"), usecols=cols)
    offset = int(tr["row_id"].max()) + 1
    te = te.assign(row_id=te["row_id"] + offset)
    return pd.concat([tr, te], ignore_index=True).rename(columns={"row_id": "pid"})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeat", type=int, required=True)
    ap.add_argument("--outer", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-repeats", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--posts-per-user", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    P = Paths()
    out_path = cv_splits.cvft_path(P, args.repeat, args.outer)
    if out_path.exists() and not args.force:
        print("exists:", out_path)
        return
    labels = cv_splits.all_users(P)
    tr_users, te_users = cv_splits.outer_splits(labels, args.n_repeats)[args.repeat * cv_splits.N_SPLITS + args.outer]
    posts = pooled_posts(P)
    texts = posts["body"].astype(str).tolist()
    y_post = posts["Depression"].to_numpy()
    post_users = posts["user_id"].to_numpy()
    by_user = posts.groupby("user_id").indices
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(fm.MODEL)

    base_seed = 42 + 100 * args.repeat + 10 * args.outer
    inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=base_seed)
    scores, test_scores, meta = {}, [], []
    te_pos = np.flatnonzero(np.isin(post_users, te_users))
    for j, (a, held) in enumerate(inner.split(np.zeros(len(tr_users)), labels.loc[tr_users].to_numpy())):
        seed = base_seed + j
        fit_pool = tr_users[a]
        fit_users, val_users = train_test_split(fit_pool, test_size=0.1, stratify=labels.loc[fit_pool].to_numpy(),
                                                random_state=seed)
        a_args = SimpleNamespace(epochs=args.epochs, posts_per_user=args.posts_per_user, batch=args.batch,
                                 lr=args.lr, max_len=args.max_len, seed=seed)
        torch.manual_seed(seed)
        print(f"outer ({args.repeat},{args.outer}) inner {j}: fit {len(fit_users)}, val {len(val_users)}, "
              f"held {len(held)}, outer-test {len(te_users)} users", flush=True)
        model, log, n_params, t_train, peak = fm.train_adapter(texts, y_post, post_users, by_user, fit_users, val_users,
                                                               a_args, args.device, tok, np.random.default_rng(seed))
        held_pos = np.flatnonzero(np.isin(post_users, tr_users[held]))
        scores[j] = pd.Series(fm.score(model, tok, [texts[i] for i in held_pos], args.device, args.max_len),
                              index=posts["pid"].to_numpy()[held_pos])
        test_scores.append(pd.Series(fm.score(model, tok, [texts[i] for i in te_pos], args.device, args.max_len),
                                     index=posts["pid"].to_numpy()[te_pos]))
        meta.append({"inner": j, "train_seconds": t_train, "peak_gpu_gb": peak,
                     "best_val_post_auc": max(e["val_post_auc"] for e in log), "epochs": len(log)})
        del model
        torch.cuda.empty_cache()
    ft = pd.concat(list(scores.values()) + [pd.concat(test_scores, axis=1).mean(axis=1)])
    out = ft.rename("ft_logit").rename_axis("pid").reset_index()
    if out["pid"].duplicated().any() or len(out) != len(posts):
        raise ValueError("every post must be scored exactly once")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_pickle(out_path)
    json.dump({"repeat": args.repeat, "outer": args.outer, "inner": meta, "args": vars(args)},
              open(str(out_path).replace(".pkl", ".json"), "w"), indent=2)
    print("saved", out_path, time.strftime("%H:%M:%S"), flush=True)


if __name__ == "__main__":
    main()
