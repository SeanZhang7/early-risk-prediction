"""Checks for post_features. Run: conda run -n hongyi python tests/test_post_features.py (or pytest)."""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import data_prep as dp
import post_features as pf


def _users(n_pos=9, n_neg=30, posts=4):
    rows = []
    for u in range(n_pos + n_neg):
        for p in range(posts):
            rows.append({"user_id": f"u{u}", "Depression": int(u < n_pos), "body": f"text {u} {p}"})
    df = pd.DataFrame(rows)
    df.insert(0, "row_id", np.arange(len(df)))
    return df


def test_re_folds_are_user_disjoint_and_stratified():
    df = _users()
    df["re_fold"] = dp.assign_re_folds(df)
    assert set(df["re_fold"]) == {0, 1, 2}
    assert (df.groupby("user_id")["re_fold"].nunique() == 1).all()
    users = df.groupby("user_id").agg(fold=("re_fold", "first"), y=("Depression", "max"))
    for k in range(3):
        assert users[(users.fold == k)].y.sum() == 3  # 9 positives spread evenly over 3 folds


def test_lsm_matches_formula():
    rng = np.random.default_rng(0)
    cols = sorted({c for g in pf.LSM_GROUPS.values() for c in g})
    body = pd.DataFrame(rng.integers(0, 5, (6, len(cols))), columns=cols)
    parent = pd.DataFrame(rng.integers(0, 5, (6, len(cols))), columns=cols)
    body.insert(0, "row_id", np.arange(6))
    parent.insert(0, "row_id", np.arange(6))
    out = pf.compute_lsm(body, parent)
    for name, group in pf.LSM_GROUPS.items():
        expected = np.mean([1 - abs(body[c] - parent[c]) / (body[c] + parent[c] + 1e-6) for c in group], axis=0)
        np.testing.assert_allclose(out[name], expected)


def test_lsmx_uses_rates_not_counts():
    rng = np.random.default_rng(1)
    cols = sorted({c for g in pf.LSM_GROUPS.values() for c in g})
    body = pd.DataFrame(rng.integers(0, 5, (6, len(cols))), columns=cols)
    parent = pd.DataFrame(rng.integers(0, 5, (6, len(cols))), columns=cols)
    body["num_tokens"], parent["num_tokens"] = [10, 20, 40, 5, 80, 1], [100, 20, 10, 50, 8, 0]
    body.insert(0, "row_id", np.arange(6))
    parent.insert(0, "row_id", np.arange(6))
    out = pf.compute_lsmx(body, parent)
    assert list(out.columns) == ["row_id"] + pf.LSMX_COLS
    rb = body[cols].div(body["num_tokens"].clip(lower=1), axis=0)
    rp = parent[cols].div(parent["num_tokens"].clip(lower=1), axis=0)
    for name, group in pf.LSM_GROUPS.items():
        expected = np.mean([1 - abs(rb[c] - rp[c]) / (rb[c] + rp[c] + 1e-6) for c in group], axis=0)
        np.testing.assert_allclose(out[f"lsmr_{name}"], expected)
        np.testing.assert_allclose(out[f"dlsm_{name}"], rb[group].mean(axis=1) - rp[group].mean(axis=1))
        np.testing.assert_allclose(out[f"par_{name}"], rp[group].mean(axis=1))
    np.testing.assert_allclose(out["par_log_tokens"], np.log1p(parent["num_tokens"]))


def test_lsm_rejects_misaligned_rows():
    cols = sorted({c for g in pf.LSM_GROUPS.values() for c in g})
    a = pd.DataFrame(np.zeros((3, len(cols))), columns=cols)
    b = a.copy()
    a.insert(0, "row_id", [0, 1, 2])
    b.insert(0, "row_id", [0, 2, 1])
    try:
        pf.compute_lsm(a, b)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_crossfit_routes_rows_to_models_that_never_saw_them(monkeypatch=None):
    fitted = {}

    def fake_finetune(texts, path, **kw):
        fitted[path] = set(texts)

    def fake_score(path, texts, **kw):
        # a model "knows" the texts it was fitted on, which would show up as a lower loss
        return np.array([0.0 if (not t) else (0.1 if t in fitted[path] else 1.0) for t in texts])

    orig = pf.finetune_lm, pf.score_losses
    pf.finetune_lm, pf.score_losses = fake_finetune, fake_score
    try:
        tr = _users()
        tr["re_fold"] = dp.assign_re_folds(tr)
        te = pd.DataFrame({"row_id": np.arange(5), "body": ["unseen", "", "x", "y", None]})
        tr_s, te_s = pf.crossfit_re(tr, te, n_folds=3, model_root="m")
    finally:
        pf.finetune_lm, pf.score_losses = orig

    assert (tr_s["loss_dep0"] == 1.0).all() and (tr_s["loss_dep1"] == 1.0).all(), "a train row was scored in-sample"
    for k in range(3):
        for c in (0, 1):
            seen = fitted[f"m/re_fold{k}_Dep{c}_LM"]
            held_users = set(tr.loc[tr.re_fold == k, "user_id"])
            held_texts = set(tr.loc[tr.user_id.isin(held_users), "body"])
            assert not (seen & held_texts)
            assert all(tr.loc[tr.body == t, "Depression"].iloc[0] == c for t in seen)
    np.testing.assert_allclose(te_s["loss_dep0"], [1.0, 0.0, 1.0, 1.0, 0.0])
    np.testing.assert_allclose(te_s["re"], te_s["loss_dep1"] - te_s["loss_dep0"])


def test_own_liwc_rates_divide_by_word_count():
    from feature_cols import OWN_LIWC_CATS

    df = pd.DataFrame({"row_id": [3, 4, 5], "num_tokens": [10, 0, 4]})
    for c in OWN_LIWC_CATS:
        df[c] = [2, 0, 1]
    out = pf.own_liwc_rates(df)
    assert list(out["row_id"]) == [3, 4, 5]
    np.testing.assert_allclose(out["liwc_I"], [0.2, 0.0, 0.25])
    np.testing.assert_allclose(out["log_tokens"], np.log1p([10, 0, 4]))


def test_batched_loss_matches_single_text_loss():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = os.path.join(ROOT, "models", "train_Dep0_LM")
    if not os.path.isdir(path):
        print("skip test_batched_loss_matches_single_text_loss (no local LM)")
        return
    texts = ["I feel tired and nothing helps.", "ok", "", "Went to the park today with my dog, it was great!",
             "word " * 400]
    got = pf.score_losses(path, texts, batch_size=2, device="cpu")

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path).eval()
    expected = []
    for t in texts:
        if not t.strip():
            expected.append(0.0)
            continue
        enc = tok(t, return_tensors="pt", truncation=True, max_length=256)
        if enc["input_ids"].size(1) < 2:
            expected.append(0.0)
            continue
        with torch.no_grad():
            loss = model(**enc, labels=enc["input_ids"]).loss.item()
        expected.append(loss)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-7)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
