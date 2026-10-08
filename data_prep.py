"""Raw eRisk threads -> cleaned, time-ordered target posts -> user-level train/test tables.

Stages 1 and 2 of data-processing.ipynb. Only posts written by target users (the labelled users) are kept;
the other participants' texts are used just to look up the parent text of a target user's comment.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from features_history import sort_posts

N_RE_FOLDS = 3
PLACEHOLDER_BODIES = {"[removed]", "[deleted]"}
POST_COLUMNS = ["user_id", "type", "title", "body", "parent", "created_utc", "submission_id", "parent_id",
                "comment_id", "Depression"]


def read_table(path, **kwargs):
    """read_csv that only treats empty cells as missing, so a post whose text is literally 'NA' survives."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""], **kwargs)


def normalize_text(s):
    """Strip, repair UTF-8 that was decoded as latin-1, and map empty text to None."""
    if not isinstance(s, str):
        return s
    s = s.strip()
    if not s:
        return None
    try:
        return s.encode("latin1").decode("utf-8")
    except UnicodeError:
        return s


def read_labels(path):
    labels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                user_id, label = line.split()
                labels[user_id] = 1 if label == "1" else 0
    return labels


def _post(item, kind, labels, parent):
    user = item.get("user_id")
    return {
        "user_id": user,
        "type": kind,
        "title": normalize_text(item.get("title")) if kind == "submission" else None,
        "body": normalize_text(item.get("body")),
        "parent": parent,
        "created_utc": item.get("created_utc"),
        "submission_id": item.get("submission_id"),
        "parent_id": item.get("parent_id"),
        "comment_id": item.get("comment_id") if kind == "comment" else None,
        "Depression": labels.get(user),
    }


def read_target_posts(threads_dir, labels_path):
    """Every submission and comment written by a target user, with the text of the item it replies to.

    Returns (posts, skipped_files). A comment's parent is looked up inside its own thread.
    """
    labels = read_labels(labels_path)
    rows, skipped = [], []
    for path in sorted(Path(threads_dir).glob("*.json")):
        try:
            threads = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            skipped.append(path.name)
            continue
        for thread in threads:
            sub = thread.get("submission", {})
            comments = thread.get("comments", [])
            sub_text = {sub.get("submission_id"): normalize_text(sub.get("body"))}
            com_text = {c.get("comment_id"): normalize_text(c.get("body")) for c in comments}

            def parent_text(pid):
                if pid is None:
                    return None
                return sub_text[pid] if pid in sub_text else com_text.get(pid)

            if sub.get("target", False):
                rows.append(_post(sub, "submission", labels, None))
            for c in comments:
                if c.get("target", False):
                    rows.append(_post(c, "comment", labels, parent_text(c.get("parent_id"))))
    return pd.DataFrame(rows, columns=POST_COLUMNS), skipped


def clean_posts(posts):
    """Drop posts that cannot be used. Returns (clean_posts, report) with per-step counts by post type."""
    steps = []

    def drop(df, mask, reason):
        mask = mask.to_numpy()
        kind = df["type"].to_numpy()
        steps.append({"step": reason, "submissions": int((mask & (kind == "submission")).sum()),
                      "comments": int((mask & (kind == "comment")).sum())})
        return df[~mask]

    df = posts.copy()
    df = drop(df, df["Depression"].isna(), "user has no label")
    df["Depression"] = df["Depression"].astype(int)
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True, errors="coerce")
    df = drop(df, df["created_utc"].isna(), "missing or invalid timestamp")
    df = drop(df, df["body"].isna(), "empty body")
    df = drop(df, df["body"].str.lower().isin(PLACEHOLDER_BODIES), "[removed]/[deleted] body")
    df = drop(df, df.duplicated(["type", "submission_id", "comment_id"]), "duplicate post")
    # A comment whose parent was removed has no usable parent text either.
    df["parent"] = df["parent"].where(~df["parent"].str.lower().isin(PLACEHOLDER_BODIES))
    steps.append({"step": "kept", "submissions": int((df["type"] == "submission").sum()),
                  "comments": int((df["type"] == "comment").sum())})
    return df.reset_index(drop=True), pd.DataFrame(steps)


def load_or_create_user_split(posts, path, test_size=0.2, seed=42):
    """user_id -> 'train'/'test'. The assignment is stored in `path` so it never depends on file order.

    If the file is missing it is created with a label-stratified split of the sorted user ids.
    """
    path = Path(path)
    users = posts.groupby("user_id")["Depression"].max()
    if path.exists():
        table = pd.read_csv(path)
        unknown = set(users.index) - set(table["user_id"])
        if unknown:
            raise ValueError(f"{len(unknown)} users are missing from {path}, e.g. {sorted(unknown)[:3]}")
        labels = table.set_index("user_id")["Depression"].reindex(users.index)
        if (labels != users).any():
            raise ValueError(f"labels in {path} disagree with the posts")
    else:
        parts = []
        for label in (1, 0):
            ids = np.sort(users.index[users == label].to_numpy())
            train, test = train_test_split(ids, test_size=test_size, random_state=seed)
            parts += [pd.DataFrame({"user_id": train, "Depression": label, "split": "train"}),
                      pd.DataFrame({"user_id": test, "Depression": label, "split": "test"})]
        table = pd.concat(parts, ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
    return table.set_index("user_id")["split"]


def assign_re_folds(train_df, n_folds=N_RE_FOLDS, seed=42):
    """User-level, label-stratified fold id for every row of `train_df`."""
    users = train_df.groupby("user_id")["Depression"].max()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of = {}
    for k, (_, held) in enumerate(skf.split(users.index.to_numpy(), users.to_numpy())):
        fold_of.update({u: k for u in users.index[held]})
    return train_df["user_id"].map(fold_of).astype(int)


def make_splits(posts, user_split_path):
    """Time-ordered train and test tables with `row_id` (row position) and `re_fold` (-1 for test)."""
    split_of = load_or_create_user_split(posts, user_split_path)
    out = {}
    for name in ("train", "test"):
        d = sort_posts(posts[posts["user_id"].map(split_of) == name])
        d["re_fold"] = assign_re_folds(d) if name == "train" else -1
        d.insert(0, "row_id", np.arange(len(d)))
        out[name] = d
    return out["train"], out["test"]


def describe_split(df):
    users = df.groupby("user_id")["Depression"].max()
    return (f"users={len(users)} (dep={int(users.sum())}, ctrl={int((users == 0).sum())}) | posts={len(df)} "
            f"(dep={int((df.Depression == 1).sum())}, ctrl={int((df.Depression == 0).sum())})")
