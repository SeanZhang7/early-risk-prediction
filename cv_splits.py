"""Shared helpers for the repeated cross-validation over every labelled user (train and test pooled).

Repeat r, fold k is entry r * N_SPLITS + k of `outer_splits`; the splits are stratified by label and identical for
every configuration, so that configurations can be compared pairwise. Increasing the number of repeats keeps the
earlier repeats unchanged.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

import data_prep as dp

N_SPLITS = 5
SEED = 2025


def all_users(P):
    """Label (0/1) of every labelled user, indexed by user_id and sorted."""
    frames = [dp.read_table(P.split_csv(s), usecols=["user_id", "Depression"]) for s in ("train", "test")]
    return pd.concat(frames).groupby("user_id")["Depression"].max().sort_index()


def outer_splits(labels, n_repeats, n_splits=N_SPLITS, seed=SEED):
    """[(train_users, test_users)] for every repeat and fold, user ids as numpy arrays."""
    users = labels.index.to_numpy()
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return [(users[tr], users[te]) for tr, te in rskf.split(np.zeros(len(users)), labels.to_numpy())]


def cvft_path(P, repeat, outer):
    return P.work / "parts" / f"cvft_r{repeat}_k{outer}.pkl"


def result_path(P, config, repeat, outer):
    return P.models_dir / "cv_all" / config / f"r{repeat}_k{outer}.pkl"
