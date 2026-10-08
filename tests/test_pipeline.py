"""Checks for the data-processing pipeline stages on synthetic threads (no models, no GPU).

Run: conda run -n hongyi python tests/test_pipeline.py
"""

import ast
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import data_prep as dp
import post_features as pf
from feature_cols import FEATURE_COLS_V3, LSMX_COLS, OWN_LIWC_COLS, PHQ_COLS
from paths import Paths


def _item(user, ts, body, **kw):
    return {"user_id": user, "target": user.startswith("subject_"), "created_utc": ts, "body": body, **kw}


def write_threads(folder, n_users=12):
    """Threads with: target/non-target authors, empty and '[removed]' bodies, replies to submissions and comments."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    labels = {}
    for u in range(n_users):
        labels[f"subject_{u}"] = int(u < 3)
        threads = []
        for t in range(6):
            sid = f"s{u}_{t}"
            base = pd.Timestamp("2021-01-01", tz="UTC") + pd.Timedelta(hours=int(rng.integers(0, 10**4)))
            sub_body = ["", "[removed]", f"post {u} {t}"][t % 3]
            threads.append({
                "submission": _item(f"subject_{u}", str(base), sub_body, submission_id=sid, parent_id=None,
                                    title=f"title {u} {t}"),
                "comments": [
                    _item("other_1", str(base + pd.Timedelta(minutes=1)), f"other says {u} {t}", submission_id=sid,
                          comment_id=f"c{u}_{t}_a", parent_id=sid),
                    _item(f"subject_{u}", str(base + pd.Timedelta(minutes=2)), f"reply to other {u} {t}",
                          submission_id=sid, comment_id=f"c{u}_{t}_b", parent_id=f"c{u}_{t}_a"),
                    _item(f"subject_{u}", str(base + pd.Timedelta(minutes=3)), f"reply to submission {u} {t}",
                          submission_id=sid, comment_id=f"c{u}_{t}_c", parent_id=sid),
                    _item(f"subject_{u}", str(base + pd.Timedelta(minutes=4)), "   ", submission_id=sid,
                          comment_id=f"c{u}_{t}_d", parent_id=sid),
                ],
            })
        (folder / f"part{u}.json").write_text(json.dumps(threads))
    (folder / "broken.json").write_text("{not json")
    return labels


def make_paths():
    root = Path(tempfile.mkdtemp())
    labels = write_threads(root / "Data" / "all_combined")
    (root / "Data" / "shuffled_ground_truth_labels.txt").write_text(
        "\n".join(f"{u} {v}" for u, v in labels.items()) + "\n")
    P = Paths(root=root)
    P.make_dirs()
    return P


def test_read_target_posts_keeps_targets_and_finds_parent_text():
    P = make_paths()
    posts, skipped = dp.read_target_posts(P.threads_dir, P.labels_file)
    assert skipped == ["broken.json"]
    assert posts["user_id"].str.startswith("subject_").all()
    assert set(posts["Depression"]) == {0, 1}
    by_id = posts.set_index(["type", "submission_id", "comment_id"], drop=False)
    reply_to_comment = posts[posts["comment_id"] == "c0_1_b"].iloc[0]
    assert reply_to_comment["parent"] == "other says 0 1"
    reply_to_sub = posts[posts["comment_id"] == "c0_2_c"].iloc[0]
    assert reply_to_sub["parent"] == "post 0 2"
    reply_to_empty_sub = posts[posts["comment_id"] == "c0_0_c"].iloc[0]
    assert reply_to_empty_sub["parent"] is None
    assert len(by_id) == len(posts)


def test_clean_posts_removes_empty_and_placeholder_posts():
    P = make_paths()
    posts, _ = dp.read_target_posts(P.threads_dir, P.labels_file)
    clean, report = dp.clean_posts(posts)
    assert clean["body"].notna().all() and (clean["body"].str.strip() != "").all()
    assert not clean["body"].str.lower().isin(dp.PLACEHOLDER_BODIES).any()
    assert not clean["parent"].dropna().str.lower().isin(dp.PLACEHOLDER_BODIES).any()
    kept = report[report["step"] == "kept"].iloc[0]
    assert kept["submissions"] == (clean["type"] == "submission").sum()
    assert kept["comments"] == (clean["type"] == "comment").sum()
    assert report.set_index("step").loc["empty body", "comments"] == 12 * 6  # the blank comment of every thread
    assert report.set_index("step").loc["empty body", "submissions"] == 12 * 2
    assert report.set_index("step").loc["[removed]/[deleted] body", "submissions"] == 12 * 2
    assert str(clean["created_utc"].dtype).startswith("datetime64")


def test_splits_are_user_level_ordered_and_stable():
    P = make_paths()
    posts, _ = dp.read_target_posts(P.threads_dir, P.labels_file)
    clean, _ = dp.clean_posts(posts)
    train, test = dp.make_splits(clean, P.user_split_file)
    assert P.user_split_file.exists()
    assert set(train["user_id"]).isdisjoint(set(test["user_id"]))
    assert len(train) + len(test) == len(clean)
    for d in (train, test):
        assert (d["row_id"].to_numpy() == np.arange(len(d))).all()
        keys = list(zip(d["user_id"], d["created_utc"]))
        assert keys == sorted(keys)
    assert (train["re_fold"].isin(range(dp.N_RE_FOLDS))).all() and (test["re_fold"] == -1).all()

    # A second run reads the stored assignment, so shuffling the posts cannot change it.
    train2, test2 = dp.make_splits(clean.sample(frac=1, random_state=1), P.user_split_file)
    assert set(train2["user_id"]) == set(train["user_id"]) and set(test2["user_id"]) == set(test["user_id"])


def test_split_file_must_cover_every_user():
    P = make_paths()
    posts, _ = dp.read_target_posts(P.threads_dir, P.labels_file)
    clean, _ = dp.clean_posts(posts)
    dp.make_splits(clean, P.user_split_file)
    extra = clean.iloc[:1].assign(user_id="subject_new")
    try:
        dp.make_splits(pd.concat([clean, extra]), P.user_split_file)
    except ValueError:
        return
    raise AssertionError("expected ValueError for a user missing from the split file")


def test_assemble_features_on_synthetic_pieces():
    P = make_paths()
    rng = np.random.default_rng(1)
    posts, _ = dp.read_target_posts(P.threads_dir, P.labels_file)
    clean, _ = dp.clean_posts(posts)
    for name, d in zip(("train", "test"), dp.make_splits(clean, P.user_split_file)):
        d.to_csv(P.split_csv(name), index=False)
        base = dp.read_table(P.split_csv(name))
        pd.DataFrame(rng.normal(size=(len(base), 9)), columns=PHQ_COLS).to_pickle(P.part(name, "phq9"))
        l0, l1 = rng.uniform(size=(2, len(base)))
        pd.DataFrame({"row_id": base["row_id"], "loss_dep0": l0, "loss_dep1": l1, "re": l1 - l0}).to_pickle(
            P.part(name, "re"))
        hp = pf.has_parent(base)
        assert 0 < hp.sum() < (base["type"] == "comment").sum(), "fixture needs comments with and without parent"
        lsm = pd.DataFrame(rng.uniform(size=(hp.sum(), 10)), columns=list(pf.LSM_GROUPS))
        lsm.insert(0, "row_id", base.loc[hp, "row_id"].to_numpy())
        lsm.to_pickle(P.part(name, "lsm"))
        pd.DataFrame({"row_id": base.loc[hp, "row_id"].to_numpy(), "sim": rng.uniform(size=hp.sum())}).to_pickle(
            P.part(name, "sim"))
        rates = pd.DataFrame(rng.uniform(0, 0.2, size=(len(base), len(OWN_LIWC_COLS))), columns=OWN_LIWC_COLS)
        rates.insert(0, "row_id", base["row_id"].to_numpy())
        rates["log_tokens"] = rng.uniform(0, 6, len(base))
        rates.to_pickle(P.part(name, "liwc_rates"))
        lsmx = pd.DataFrame(rng.normal(size=(hp.sum(), len(LSMX_COLS))), columns=LSMX_COLS)
        lsmx.insert(0, "row_id", base.loc[hp, "row_id"].to_numpy())
        lsmx.to_pickle(P.part(name, "lsmx"))

        out = pf.assemble_features(P, name)
        assert len(out) == len(base) and set(out["row_id"]) == set(base["row_id"])
        assert out[FEATURE_COLS_V3].notna().all().all()
        assert P.features(name).exists()
        assert (out["has_parent"] == hp.set_axis(base["row_id"]).reindex(out["row_id"]).to_numpy().astype(float)).all()


def test_attach_embeddings_matches_rows_by_row_id():
    import experiment as ex

    P = make_paths()
    emb = np.arange(5 * 4, dtype=np.float16).reshape(5, 4)
    np.save(P.embeddings("train"), emb)
    df = pd.DataFrame({"row_id": [3, 0, 4], "x": [1.0, 2.0, 3.0]}, index=[10, 11, 12])
    out = ex.attach_embeddings(P, df, "train")
    assert list(out.columns) == ["row_id", "x", "mental_0", "mental_1", "mental_2", "mental_3"]
    np.testing.assert_array_equal(out[["mental_0", "mental_1", "mental_2", "mental_3"]].to_numpy(), emb[[3, 0, 4]])
    assert list(out.index) == [10, 11, 12]


def test_attach_ft_logit_matches_rows_by_row_id_and_rejects_gaps():
    import experiment as ex

    P = make_paths()
    pd.DataFrame({"row_id": [0, 1, 2, 3], "ft_logit": [0.5, -1.0, 2.0, 0.0]}).to_pickle(P.part("train", "ft_mental"))
    df = pd.DataFrame({"row_id": [3, 0, 2], "x": [1.0, 2.0, 3.0]}, index=[10, 11, 12])
    out = ex.attach_ft_logit(P, df, "train")
    assert list(out["ft_logit"]) == [0.0, 0.5, 2.0] and list(out.index) == [10, 11, 12]
    try:
        ex.attach_ft_logit(P, df.assign(row_id=[3, 0, 9]), "train")
    except ValueError:
        return
    raise AssertionError("expected ValueError for a post without ft_logit")


def test_paths_honour_environment_overrides():
    os.environ["ERISK_THREADS_DIR"] = "/somewhere/else"
    try:
        assert Paths().threads_dir == Path("/somewhere/else")
    finally:
        del os.environ["ERISK_THREADS_DIR"]


def test_notebook_has_no_machine_specific_paths_and_parses():
    nb = json.load(open(ROOT / "data-processing.ipynb"))
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "code":
            ast.parse(src)
            assert "/u50/" not in src and "/nfs/" not in src, src[:80]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
