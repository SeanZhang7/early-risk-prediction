"""Run experiments from the command line so a long training run survives a dropped notebook connection.

    nohup conda run -n hongyi python run_experiments.py all only_PHQ --seeds 42 43 44 > Results/v2/train.log 2>&1 &

Uses the same settings and cache as Develop/Train_and_evaluate.ipynb, which then just loads the results.
Names are keys of feature_cols.EXPERIMENTS_V2 (all, post_level_only, only_PHQ, without_RE, ...).
Seed 42 keeps the plain name; other seeds are stored as <name>_seed<seed>.
"""

import argparse
from dataclasses import replace

import experiment as ex
import seq_model as sm
from feature_cols import EXPERIMENTS_V2
from features_history import build_features
from paths import Paths


def parent_only(df):
    """Keep only comments whose parent text is known and rebuild every history feature on that subsequence."""
    return build_features(df[df["has_parent"] == 1].reset_index(drop=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="+", choices=sorted(EXPERIMENTS_V2))
    parser.add_argument("--seeds", nargs="+", type=int, default=[ex.BASE_SEED])
    parser.add_argument("--force", action="store_true", help="train again even if cached results exist")
    parser.add_argument("--device", default=None, help="default: the GPU with the most free memory")
    parser.add_argument("--parent-only", action="store_true",
                        help="use only comments with a known parent text; results are stored as parent_<name>")
    args = parser.parse_args()

    P = Paths()
    device = args.device or sm.pick_device()
    print("device:", device, flush=True)
    train_df, test_df = ex.load_features(P)
    prefix = ""
    if args.parent_only:
        train_df, test_df, prefix = parent_only(train_df), parent_only(test_df), "parent_"
        for split, d in (("train", train_df), ("test", test_df)):
            users = d.groupby("user_id")["Depression"].max()
            print(f"parent-only {split}: {len(d)} rows, {len(users)} users ({int(users.sum())} depressed)", flush=True)
    if any(c.startswith("mental_") for n in args.names for c in EXPERIMENTS_V2[n]):
        train_df, test_df = ex.attach_embeddings(P, train_df, "train"), ex.attach_embeddings(P, test_df, "test")
    if any("ft_logit" in EXPERIMENTS_V2[n] for n in args.names):
        train_df, test_df = ex.attach_ft_logit(P, train_df, "train"), ex.attach_ft_logit(P, test_df, "test")
    for seed in args.seeds:
        cfg = replace(ex.Config(), seed=seed)
        for name in args.names:
            ex.run_experiment(ex.seeded_name(prefix + name, seed), EXPERIMENTS_V2[name], train_df, test_df, P, cfg,
                              device=device, force=args.force)


if __name__ == "__main__":
    main()
