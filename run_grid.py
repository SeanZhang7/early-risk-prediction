"""Small hyper-parameter grid for the full model, scored on out-of-fold predictions of the training users.

    python run_grid.py --device cuda:0 --shard 0/2      # train half of the grid
    python run_grid.py --device cuda:1 --shard 1/2      # ... and the other half, in parallel
    python run_grid.py --summary                        # table of out-of-fold results, best first

Configurations are compared on the training users only (every user is held out exactly once, with the same
folds for every configuration). The test users are not used to choose a configuration.
"""

import argparse
import itertools
import json
from dataclasses import replace

import pandas as pd

import experiment as ex
import seq_model as sm
from feature_cols import EXPERIMENTS_V2
from paths import Paths

SIZES = [(128, 2), (256, 3)]  # (hidden_dim, n_layers)
DROPOUTS = [0.1, 0.3]
LEARNING_RATES = [1e-4, 3e-5]
INNER_VAL = [0.12, 0.25]  # share of each fold's training users used for early stopping


def grid(base=ex.Config()):
    for (hidden, layers), dropout, lr, val in itertools.product(SIZES, DROPOUTS, LEARNING_RATES, INNER_VAL):
        name = f"grid_h{hidden}x{layers}_do{dropout}_lr{lr:g}_val{val}"
        yield name, replace(base, hidden_dim=hidden, n_layers=layers, dropout=dropout, lr=lr, inner_val_frac=val)


def summary(P):
    rows = []
    for name, cfg in grid():
        path = ex.experiment_dir(P, name) / "results.json"
        if not path.exists():
            continue
        r = json.load(open(path))
        op = r["operating_point"]
        rows.append({"name": name, "size": f"{cfg.hidden_dim}x{cfg.n_layers}", "dropout": cfg.dropout, "lr": cfg.lr,
                     "inner_val": cfg.inner_val_frac, "oof_pr_auc": r["oof_ranking"]["pr_auc"],
                     "oof_roc_auc": r["oof_ranking"]["roc_auc"], "oof_f1": op["oof"]["f1"],
                     "oof_f_latency": op["oof"]["f_latency"], "oof_auc@10": r["oof_horizon_auc"]["10"],
                     "oof_auc@50": r["oof_horizon_auc"]["50"],
                     "best_epochs": [f["best_epoch"] for f in r["folds"]]})
    return pd.DataFrame(rows).sort_values("oof_pr_auc", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=None)
    parser.add_argument("--shard", default="0/1", help="i/n: train every n-th configuration, starting at i")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    P = Paths()
    if args.summary:
        pd.set_option("display.width", 250)
        table = summary(P)
        table.to_csv(P.models_dir / "classifier" / "grid_summary.csv", index=False)
        print(table.round(3).to_string())
        return
    i, n = map(int, args.shard.split("/"))
    device = args.device or sm.pick_device()
    train_df, test_df = ex.load_features(P)
    for name, cfg in list(grid())[i::n]:
        ex.run_experiment(name, EXPERIMENTS_V2["all"], train_df, test_df, P, cfg, device=device, verbose=False)
        print("done", name, flush=True)


if __name__ == "__main__":
    main()
