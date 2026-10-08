"""Post-level feature helpers for data-processing.ipynb (no history columns; see features_history.py).

Every frame handed around here carries a `row_id` that is the row position in the split table, so
LIWC, LSM, RE, cosine and PHQ outputs are merged on `row_id` rather than on row order.
Locations come from paths.Paths.
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data_prep import read_table
from feature_cols import FEATURE_COLS_V2, FEATURE_COLS_V3, LSMX_COLS, OWN_LIWC_CATS, PHQ_COLS
from features_history import build_features

MODEL_NAME = "distilgpt2"
MAX_LEN = 256

LIWC_COLS = ["ARTICLE", "AUXVERB", "CONJ", "ADVERB", "PPRON", "IPRON", "PREP", "NEGATE", "QUANT", "VERB"]
SOCIAL_COLS = ["SOCIAL", "FRIEND", "FAMILY", "WE", "YOU", "THEY", "AFFILIATION", "DRIVES", "POWER", "RELIG",
               "HEALTH", "WORK", "MONEY"]
POSITIVE_EMO = ["POSEMO", "REWARD", "ACHIEV", "JOY", "AFFILIATION", "CERTAIN", "POWER"]
NEGATIVE_EMO = ["NEGEMO", "SAD", "ANX", "ANGER", "RISK", "DEATH", "FEAR", "DISCREP", "TENTAT"]
COGNITIVE_PROC = ["CAUSE", "DISCREP", "INSIGHT", "CERTAIN", "COGPROC"]
PERCEPTUAL_PROC = ["SEE", "HEAR", "FEEL", "PERCEPT"]

LSM_GROUPS = {
    "LSM_mean": LIWC_COLS,
    "social_mean": SOCIAL_COLS,
    "positive_mean": POSITIVE_EMO,
    "negative_mean": NEGATIVE_EMO,
    "first_person_singular": ["I"],
    "second_person": ["YOU"],
    "third_person_singular": ["SHEHE"],
    "third_person_plural": ["THEY"],
    "cognitive_process": COGNITIVE_PROC,
    "perceptual_process": PERCEPTUAL_PROC,
}
_ALL_LIWC = sorted({c for cols in LSM_GROUPS.values() for c in cols})
_EPS = 1e-6


def has_parent(df):
    """True for comments whose parent text is known; only these rows get sim and LSM values."""
    return (df["type"] == "comment") & df["parent"].fillna("").astype(str).str.strip().ne("")


def run_liwc(P, split, which):
    """Run liwc_script.py on the `body` or `parent` column of a split table; output lands at P.liwc(split, which)."""
    out = P.liwc(split, which)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(P.root / "liwc_script.py"), "--data", str(P.split_csv(split)),
         "--column", which, "--savedir", str(out.parent), "--save", f"{split}_{which}"],
        check=True, cwd=P.root, env={**os.environ, "NLTK_DATA": str(P.nltk_data)},
    )
    if not out.exists():
        raise FileNotFoundError(f"liwc_script.py did not produce {out}")


def run_phq9(P, split, device="cuda"):
    """Score every body against the PHQ-9 archetype items with the extremism submodule; output at P.part(split, 'phq9')."""
    out = P.part(split, "phq9")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "item-scoring/item_scoring.py", "--custom-dataset", str(P.split_csv(split)),
         "--text-column", "body", "--scale", "PHQ-9_archetype_scale", "--device", device,
         "--output", str(out), "-w"],
        check=True, cwd=P.extremism_dir,  # the script resolves scale names relative to its own folder
    )


ID_COLS = ["row_id", "user_id", "created_utc", "type", "Depression", "comment_id", "submission_id", "re_fold"]


def assemble_features(P, split):
    """Merge the post-level pieces of one split, derive all history features and save P.features(split)."""
    base = read_table(P.split_csv(split))
    if base["body"].isna().any():
        raise ValueError(f"{P.split_csv(split)} has posts without text; rerun the cleaning stage")
    phq = pd.read_pickle(P.part(split, "phq9")).reset_index(drop=True)
    if len(phq) != len(base):
        raise ValueError("PHQ-9 rows do not line up with the split table")
    df = pd.concat([base, phq[PHQ_COLS]], axis=1)
    df["has_parent"] = has_parent(df)  # sim and LSM exist only for these rows
    for piece in ("re", "lsm", "sim", "liwc_rates", "lsmx"):
        df = df.merge(pd.read_pickle(P.part(split, piece)), on="row_id", how="left", validate="1:1")

    out = build_features(df)[ID_COLS + FEATURE_COLS_V3]
    if out[FEATURE_COLS_V3].isna().any().any():
        raise ValueError("NaN left in the feature matrix")
    P.features(split).parent.mkdir(parents=True, exist_ok=True)
    out.to_pickle(P.features(split))
    return out


def compute_lsm(body_liwc, parent_liwc):
    """The 10 LSM features from LIWC counts of body and parent; both frames must list the same rows in the same order."""
    ids = body_liwc["row_id"].to_numpy()
    if not (ids == parent_liwc["row_id"].to_numpy()).all():
        raise ValueError("body and parent LIWC frames are not row-aligned")
    lsm = {c: 1 - (body_liwc[c] - parent_liwc[c]).abs() / (body_liwc[c] + parent_liwc[c] + _EPS)
           for c in _ALL_LIWC}
    out = pd.DataFrame({"row_id": ids})
    for name, cols in LSM_GROUPS.items():
        out[name] = pd.concat([lsm[c] for c in cols], axis=1).mean(axis=1).to_numpy()
    return out


def compute_lsmx(body_liwc, parent_liwc):
    """LSMX features: LSM on word rates, signed body-minus-parent group rates, parent group rates, parent length."""
    ids = body_liwc["row_id"].to_numpy()
    if not (ids == parent_liwc["row_id"].to_numpy()).all():
        raise ValueError("body and parent LIWC frames are not row-aligned")
    nb = body_liwc["num_tokens"].clip(lower=1).to_numpy()
    npar = parent_liwc["num_tokens"].clip(lower=1).to_numpy()
    rate_b = {c: body_liwc[c].to_numpy() / nb for c in _ALL_LIWC}
    rate_p = {c: parent_liwc[c].to_numpy() / npar for c in _ALL_LIWC}
    out = {"row_id": ids}
    for name, cols in LSM_GROUPS.items():
        lsm = np.mean([1 - np.abs(rate_b[c] - rate_p[c]) / (rate_b[c] + rate_p[c] + _EPS) for c in cols], axis=0)
        body_rate = np.mean([rate_b[c] for c in cols], axis=0)
        parent_rate = np.mean([rate_p[c] for c in cols], axis=0)
        out[f"lsmr_{name}"], out[f"dlsm_{name}"], out[f"par_{name}"] = lsm, body_rate - parent_rate, parent_rate
    out["par_log_tokens"] = np.log1p(parent_liwc["num_tokens"].to_numpy())
    return pd.DataFrame(out)[["row_id"] + LSMX_COLS]


def own_liwc_rates(body_liwc):
    """Rate of each LIWC category in a post's own text (count / word count) and the log word count, by row_id."""
    n = body_liwc["num_tokens"].clip(lower=1)
    out = pd.DataFrame({"row_id": body_liwc["row_id"].to_numpy()})
    for c in OWN_LIWC_CATS:
        out[f"liwc_{c}"] = (body_liwc[c] / n).to_numpy()
    out["log_tokens"] = np.log1p(body_liwc["num_tokens"]).to_numpy()
    return out


def cosine_sims(encoder, bodies, parents, batch_size=256):
    sims = []
    for i in range(0, len(bodies), batch_size):
        emb_b = encoder.encode(bodies[i:i + batch_size], batch_size=64, convert_to_tensor=True)
        emb_p = encoder.encode(parents[i:i + batch_size], batch_size=64, convert_to_tensor=True)
        sims.extend(F.cosine_similarity(emb_b, emb_p).cpu().numpy())
    return np.asarray(sims, dtype=float)


def finetune_lm(texts, save_path, epochs=3, batch_size=8, lr=5e-5, device="cuda", seed=42):
    """Fine-tune distilgpt2 on `texts`; a no-op if `save_path` already holds a finished model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if os.path.exists(os.path.join(save_path, "config.json")):
        print(f"[skip] {save_path} already exists")
        return
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    texts = list(texts)
    model.train()
    for ep in range(epochs):
        order = np.random.default_rng(seed + ep).permutation(len(texts))
        total, steps = 0.0, 0
        for i in range(0, len(order), batch_size):
            batch = [texts[j] for j in order[i:i + batch_size]]
            enc = tok(batch, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
            labels = enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)  # pads must not count as targets
            loss = model(**enc, labels=labels).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1
        print(f"{save_path} epoch {ep + 1} loss = {total / steps:.4f}")
    tmp = save_path + ".tmp"
    model.save_pretrained(tmp)
    tok.save_pretrained(tmp)
    os.rename(tmp, save_path)


@torch.no_grad()
def score_losses(model_dir, texts, batch_size=32, device="cuda"):
    """Mean token cross-entropy of each text, so texts of different length are comparable (0.0 for empty or
    single-token text)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device).eval()
    texts = [t.strip() for t in texts]
    out = np.zeros(len(texts))
    todo = sorted((i for i, t in enumerate(texts) if t), key=lambda i: -len(texts[i]))
    for s in range(0, len(todo), batch_size):
        ids = todo[s:s + batch_size]
        enc = tok([texts[i] for i in ids], truncation=True, max_length=MAX_LEN, padding=True,
                  return_tensors="pt").to(device)
        logits = model(**enc).logits[:, :-1].float()
        target = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:]
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction="none")
        ce = ce.view(target.shape)
        mean_ce = (ce * mask).sum(1) / mask.sum(1).clamp(min=1)
        n_tok = enc["attention_mask"].sum(1)
        val = torch.where(n_tok < 2, torch.zeros_like(mean_ce), mean_ce)
        out[ids] = val.cpu().numpy()
    del model
    return out


def crossfit_re(train_df, test_df, n_folds=3, model_root="models", device="cuda", epochs=3):
    """Relative-entropy features without in-sample scoring.

    A train row is scored by language models fitted on the other folds; test rows get the mean loss of
    the fold models. `train_df` needs row_id, Depression, body and re_fold (0..n_folds-1).
    Returns (train_scores, test_scores) with row_id, loss_dep0, loss_dep1, re.
    """
    tr_text = train_df["body"].fillna("").astype(str).str.strip().to_numpy()
    te_text = test_df["body"].fillna("").astype(str).str.strip().tolist()
    label = train_df["Depression"].to_numpy()
    fold = train_df["re_fold"].to_numpy()
    tr_loss = np.zeros((2, len(train_df)))
    te_loss = np.zeros((2, len(test_df)))

    for k in range(n_folds):
        paths = [os.path.join(model_root, f"re_fold{k}_Dep{c}_LM") for c in (0, 1)]
        for c, path in enumerate(paths):
            fit = (fold != k) & (label == c) & (tr_text != "")
            finetune_lm(tr_text[fit].tolist(), path, epochs=epochs, device=device)
        held = fold == k
        for c, path in enumerate(paths):
            tr_loss[c, held] = score_losses(path, tr_text[held].tolist(), device=device)
            te_loss[c] += score_losses(path, te_text, device=device) / n_folds

    def frame(df, loss):
        return pd.DataFrame({"row_id": df["row_id"].to_numpy(), "loss_dep0": loss[0], "loss_dep1": loss[1],
                             "re": loss[1] - loss[0]})

    return frame(train_df, tr_loss), frame(test_df, te_loss)
