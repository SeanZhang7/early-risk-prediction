"""Repeated stratified cross-validation over all labelled users (train and test pooled), nested decision rule.

    nohup conda run -n hongyi python -u cv_all.py --repeats 3 --shard 0 3 --device cuda:0 > Results/v2/cv_0.log 2>&1 &

Every outer fold (repeat r, fold k) holds out about 20 % of the users. The sequence model is trained exactly as in
experiment.run_experiment on the remaining users with an inner 5-fold CV; the pooled inner out-of-fold probabilities
choose the decision rule (th, k), the 5 inner models are averaged on the held-out users. All configurations use the
same outer and inner splits and seeds. Results go to models/<version>/cv_all/<config>/r<r>_k<k>.pkl and are
analysed by cv_significance.py.

Only configurations whose features are not trained on labels (PHQ, LIWC, LSM, frozen embeddings) or that are re-fitted
inside each outer fold (ft_logit, from cv_ft.py) are allowed; the RE feature would leak across folds and is excluded.
"""
import argparse
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

import cv_splits
import experiment as ex
import seq_eval as se
import seq_model as sm
from feature_cols import EXPERIMENTS_V2
from paths import Paths

DEFAULT_CONFIGS = ["only_PHQ", "lean_PHQ_plus_LIWC", "lean_without_RE", "lean2_without_RE", "mental_emb",
                   "mental_plus_lean2_without_RE", "ft_logit", "ft_logit_plus_lean2_without_RE"]
FORBIDDEN = {"re", "loss_dep0", "loss_dep1", "re_cummean"}


def load_pooled(P, with_mental):
    """Feature tables of train and test stacked, with a unique post id `pid` (row_id, + n_train for test posts)."""
    train_df, test_df = ex.load_features(P)
    if with_mental:
        train_df, test_df = ex.attach_embeddings(P, train_df, "train"), ex.attach_embeddings(P, test_df, "test")
    offset = int(train_df["row_id"].max()) + 1
    pooled = pd.concat([train_df.assign(pid=train_df["row_id"]), test_df.assign(pid=test_df["row_id"] + offset)],
                       ignore_index=True)
    if pooled["user_id"].isin(train_df["user_id"]).sum() != len(train_df):
        raise ValueError("a user appears in both train and test")
    return pooled


def run_outer(frame, cols, tr_users, te_users, seed, device, cfg):
    """Train on `tr_users`, return the ensemble probabilities of `te_users` and the chosen decision rule.
    `frame` holds `cols`, user_id and Depression, sorted by user and time."""
    mask = frame["user_id"].isin(set(tr_users)).to_numpy()
    scaler = sm.Standardizer.fit(frame[mask], cols)
    users_arr, seqs, y = sm.make_sequences(frame, scaler.transform(frame))
    pos = {u: i for i, u in enumerate(users_arr)}
    tr_idx = np.array([pos[u] for u in tr_users])
    te_idx = np.array([pos[u] for u in te_users])
    model_kw = dict(hidden_dim=cfg.hidden_dim, n_heads=cfg.n_heads, n_layers=cfg.n_layers, dropout=cfg.dropout)

    inner = StratifiedKFold(cfg.n_folds, shuffle=True, random_state=seed)
    oof, test_probs, epochs = {}, [], []
    for j, (a, held) in enumerate(inner.split(np.zeros(len(tr_idx)), y[tr_idx])):
        fit_pool, held_pos = tr_idx[a], tr_idx[held]
        fit_idx, val_idx = train_test_split(fit_pool, test_size=cfg.inner_val_frac, stratify=y[fit_pool],
                                            random_state=seed + j)
        model, history = sm.fit([seqs[i] for i in fit_idx], y[fit_idx], [seqs[i] for i in val_idx], y[val_idx],
                                device, epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
                                patience=cfg.patience, seed=seed + j, verbose=False, **model_kw)
        for i, p in zip(held_pos, sm.predict(model, [seqs[i] for i in held_pos], device)):
            oof[i] = p
        test_probs.append(sm.predict(model, [seqs[i] for i in te_idx], device))
        epochs.append(len(history))
    table = se.sweep([oof[i] for i in tr_idx], y[tr_idx], cfg.thresholds, cfg.ks)
    op = se.select_operating_point(table, cfg.metric)
    ensemble = [np.mean([tp[i] for tp in test_probs], axis=0) for i in range(len(te_idx))]
    return {"users": users_arr[te_idx], "labels": y[te_idx], "probs": ensemble, "th": float(op["th"]),
            "k": int(op["k"]), "oof_f_latency": float(op["f_latency"]), "inner_epochs": epochs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS, choices=sorted(EXPERIMENTS_V2))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--shard", nargs=2, type=int, default=[0, 1], metavar=("I", "N"),
                    help="process the outer folds with index %% N == I")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    P = Paths()
    device = args.device or sm.pick_device()
    for name in args.configs:
        if FORBIDDEN & set(EXPERIMENTS_V2[name]):
            raise ValueError(f"{name} uses the RE feature, which leaks across the pooled folds")
    labels = cv_splits.all_users(P)
    splits = cv_splits.outer_splits(labels, args.repeats)
    pooled = load_pooled(P, any(c.startswith("mental_") for n in args.configs for c in EXPERIMENTS_V2[n]))
    cfg = ex.Config()
    print(f"device {device}; {len(labels)} users ({int(labels.sum())} depressed); {len(splits)} outer folds", flush=True)
    for t in range(len(splits)):
        if t % args.shard[1] != args.shard[0]:
            continue
        r, k = divmod(t, cv_splits.N_SPLITS)
        tr_users, te_users = splits[t]
        for name in args.configs:
            path = cv_splits.result_path(P, name, r, k)
            if path.exists() and not args.force:
                continue
            cols = EXPERIMENTS_V2[name]
            plain = [c for c in cols if c != "ft_logit"]
            frame = pooled[plain + ["user_id", "Depression"]]
            if "ft_logit" in cols:
                ft_path = cv_splits.cvft_path(P, r, k)
                if not ft_path.exists():
                    print(f"[{name}] r{r} k{k}: waiting for {ft_path.name}, skipped", flush=True)
                    continue
                ft = pd.read_pickle(ft_path).set_index("pid")["ft_logit"]
                frame = frame.assign(ft_logit=pooled["pid"].map(ft).to_numpy(np.float64))
                if frame["ft_logit"].isna().any():
                    raise ValueError("ft_logit missing for some posts")
            t0 = time.perf_counter()
            res = run_outer(frame, cols, tr_users, te_users, 7 + 1000 * r + k, device, cfg)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(res, f)
            print(f"[{name}] r{r} k{k}: rule th={res['th']} k={res['k']} in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
