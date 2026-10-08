"""Cross-validated training and evaluation of the user-sequence model for one feature set.

run_experiment trains one model per fold of the training users, picks the decision rule (threshold, k) from
the pooled out-of-fold probabilities, and evaluates the ensemble of the fold models once on the test users.
Results are cached under models/<version>/classifier/<name>/ and reused unless the settings change.
"""

import json
import pickle
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

import seq_eval as se
import seq_model as sm
from features_history import sort_posts


@dataclass(frozen=True)
class Config:
    n_folds: int = 5
    seed: int = 42
    inner_val_frac: float = 0.12  # share of each fold's training users used for early stopping
    epochs: int = 30
    lr: float = 3e-5  # chosen on out-of-fold PR-AUC in run_grid.py
    weight_decay: float = 0.01
    patience: int = 5
    hidden_dim: int = 256
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.3
    thresholds: tuple = tuple(round(0.05 * i, 2) for i in range(1, 20))
    ks: tuple = (1, 3, 5, 10)
    metric: str = "f_latency"  # what the decision rule is chosen for
    horizons: tuple = (10, 50, 100, 200)
    n_boot: int = 1000


def load_features(P):
    """Train and test feature tables, time-ordered within each user."""
    return tuple(sort_posts(pd.read_pickle(P.features(s))) for s in ("train", "test"))


def attach_ft_logit(P, df, split):
    """Add the cross-fitted adapter-fine-tuned MentalRoBERTa logit `ft_logit` (finetune_mental.py), matched on row_id."""
    ft = pd.read_pickle(P.part(split, "ft_mental")).set_index("row_id")["ft_logit"]
    out = df.assign(ft_logit=df["row_id"].map(ft).to_numpy(np.float64))
    if out["ft_logit"].isna().any():
        raise ValueError(f"{split}: some posts have no ft_logit")
    return out


def attach_embeddings(P, df, split, name="mental"):
    """Add the frozen embedding columns (<name>_0 ...) to `df`, matched on row_id."""
    emb = np.load(P.embeddings(split, name), mmap_mode="r")
    cols = [f"{name}_{i}" for i in range(emb.shape[1])]
    block = pd.DataFrame(np.asarray(emb[df["row_id"].to_numpy()], dtype=np.float32), columns=cols, index=df.index)
    return pd.concat([df, block], axis=1)


def experiment_dir(P, name):
    return P.models_dir / "classifier" / name


BASE_SEED = 42


def seeded_name(name, seed):
    """The base seed keeps the plain name so single-seed results and notebooks share one cache."""
    return name if seed == BASE_SEED else f"{name}_seed{seed}"


def _plain(obj):
    """Make numpy values JSON-safe; NaN becomes None."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def run_experiment(name, cols, train_df, test_df, P, cfg=Config(), device="cpu", force=False, verbose=True):
    out_dir = experiment_dir(P, name)
    results_path = out_dir / "results.json"
    signature = _plain({"config": asdict(cfg), "cols": list(cols)})
    if results_path.exists() and not force:
        saved = json.load(open(results_path))
        if saved["signature"] == signature:
            print(f"[{name}] reusing {results_path}")
            return saved
        print(f"[{name}] settings changed, training again")
    out_dir.mkdir(parents=True, exist_ok=True)

    scaler = sm.Standardizer.fit(train_df, cols)
    scaler.save(out_dir / "scaler.json")
    users, seqs, y = sm.make_sequences(train_df, scaler.transform(train_df))
    test_users, test_seqs, test_y = sm.make_sequences(test_df, scaler.transform(test_df))
    model_kw = dict(hidden_dim=cfg.hidden_dim, n_heads=cfg.n_heads, n_layers=cfg.n_layers, dropout=cfg.dropout)

    folds = StratifiedKFold(cfg.n_folds, shuffle=True, random_state=cfg.seed).split(np.zeros(len(y)), y)
    oof, test_probs, fold_info = [None] * len(seqs), [], []
    for k, (tr, held) in enumerate(folds):
        fit_idx, val_idx = train_test_split(tr, test_size=cfg.inner_val_frac, stratify=y[tr],
                                            random_state=cfg.seed + k)
        if verbose:
            print(f"[{name}] fold {k + 1}/{cfg.n_folds}: fit {len(fit_idx)} users, early-stop {len(val_idx)}, "
                  f"held out {len(held)}")
        model, history = sm.fit([seqs[i] for i in fit_idx], y[fit_idx], [seqs[i] for i in val_idx], y[val_idx],
                                device, epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
                                patience=cfg.patience, seed=cfg.seed + k, verbose=verbose, **model_kw)
        for i, p in zip(held, sm.predict(model, [seqs[i] for i in held], device)):
            oof[i] = p
        test_probs.append(sm.predict(model, test_seqs, device))
        torch.save(model.state_dict(), out_dir / f"fold{k}.pt")
        fold_info.append({"fold": k, "best_epoch": model.best_epoch, "epochs_run": len(history),
                          "best_val_loss": min(h["val_loss"] for h in history)})

    table = se.sweep(oof, y, cfg.thresholds, cfg.ks)
    table.to_csv(out_dir / "oof_sweep.csv", index=False)
    op = se.select_operating_point(table, cfg.metric)
    th, k = op["th"], int(op["k"])

    ensemble = [np.mean([tp[i] for tp in test_probs], axis=0) for i in range(len(test_seqs))]
    results = {
        "name": name, "signature": signature, "n_features": len(cols), "folds": fold_info,
        "operating_point": {"th": th, "k": k, "oof": {m: op[m] for m in ("precision", "recall", "f1", "median_delay",
                                                                         "f_latency")}},
        "oof_ranking": se.ranking_scores(oof, y),
        "oof_horizon_auc": se.horizon_auc(oof, y, cfg.horizons),
        "test": {**se.evaluate(ensemble, test_y, th, k), "ci": se.bootstrap_ci(ensemble, test_y, th, k, cfg.n_boot,
                                                                                cfg.seed),
                 "ranking": se.ranking_scores(ensemble, test_y),
                 "horizon_auc": se.horizon_auc(ensemble, test_y, cfg.horizons),
                 "per_fold": [se.evaluate(tp, test_y, th, k) for tp in test_probs]},
    }
    with open(out_dir / "probs.pkl", "wb") as f:
        pickle.dump({"oof": {"users": users, "labels": y, "probs": oof},
                     "test": {"users": test_users, "labels": test_y, "probs": ensemble}}, f)
    with open(results_path, "w") as f:
        json.dump(_plain(results), f, indent=2)
    return json.load(open(results_path))


def load_probs(P, name):
    with open(experiment_dir(P, name) / "probs.pkl", "rb") as f:
        return pickle.load(f)


def summary_table(results):
    """One row per experiment: how it did out-of-fold and on the test users (95% interval for test F1)."""
    rows = []
    for name, r in results.items():
        t, op = r["test"], r["operating_point"]
        lo, hi = t["ci"]["f1"]
        rows.append({
            "experiment": name, "features": r["n_features"], "th": op["th"], "k": op["k"],
            "oof_f1": op["oof"]["f1"], "oof_f_latency": op["oof"]["f_latency"],
            "test_precision": t["precision"], "test_recall": t["recall"], "test_f1": t["f1"],
            "test_f1_ci": f"{lo:.2f}-{hi:.2f}", "test_median_delay": t["median_delay"],
            "test_f_latency": t["f_latency"], "test_roc_auc": t["ranking"]["roc_auc"],
            "test_pr_auc": t["ranking"]["pr_auc"],
            **{f"auc@{n}": a for n, a in t["horizon_auc"].items()},
        })
    return pd.DataFrame(rows).set_index("experiment").round(3)


def seed_summary(P, names, seeds):
    """Mean ± std over random seeds of the main metrics; each seed re-draws the folds and the initialisation."""
    rows = []
    for name in names:
        runs = []
        for seed in seeds:
            path = experiment_dir(P, seeded_name(name, seed)) / "results.json"
            if path.exists():
                runs.append(json.load(open(path)))
        if not runs:
            continue
        metrics = {
            "oof_pr_auc": [r["oof_ranking"]["pr_auc"] for r in runs],
            "oof_f_latency": [r["operating_point"]["oof"]["f_latency"] for r in runs],
            "test_precision": [r["test"]["precision"] for r in runs],
            "test_recall": [r["test"]["recall"] for r in runs],
            "test_f1": [r["test"]["f1"] for r in runs],
            "test_median_delay": [r["test"]["median_delay"] for r in runs],
            "test_f_latency": [r["test"]["f_latency"] for r in runs],
            "test_roc_auc": [r["test"]["ranking"]["roc_auc"] for r in runs],
            "test_pr_auc": [r["test"]["ranking"]["pr_auc"] for r in runs],
            "auc@10": [r["test"]["horizon_auc"]["10"] for r in runs],
            "auc@100": [r["test"]["horizon_auc"]["100"] for r in runs],
        }
        row = {"experiment": name, "seeds": len(runs)}
        for m, v in metrics.items():
            v = np.array([x for x in v if x is not None], dtype=float)
            row[m] = f"{v.mean():.3f} ± {v.std(ddof=1) if len(v) > 1 else 0:.3f}" if len(v) else "n/a"
        rows.append(row)
    return pd.DataFrame(rows).set_index("experiment")
