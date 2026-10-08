"""Canonical feature column list shared by training notebooks and Test/feature_extractor.py."""

PHQ_COLS = [f"PHQ-9_archetype_scale.{i}.sim" for i in range(9)]

LSM_BASE_FEATURES = [
    "LSM_mean", "social_mean", "positive_mean", "negative_mean",
    "first_person_singular", "second_person",
    "third_person_singular", "third_person_plural",
    "cognitive_process", "perceptual_process",
]

# Order matters: nn.Linear input weights are positional.
FEATURE_COLS = (
    PHQ_COLS +
    ["phq_score", "max_phq_so_far", "min_phq_so_far", "max_gap",
     "delta_phq", "rolling_std_phq", "post_index"] +
    [f"max_{i}_so_far" for i in range(9)] +
    ["sim", "max_sim_so_far", "min_sim_so_far", "max_gap_sim_so_far",
     "delta_sim", "rolling_std_sim"] +
    LSM_BASE_FEATURES +
    [f"{f}_{s}" for f in LSM_BASE_FEATURES
     for s in ["max_so_far", "min_so_far", "max_gap_so_far", "delta", "rolling_std"]] +
    ["loss_dep0", "loss_dep1", "re",
     "max_re_so_far", "min_re_so_far", "max_gap_re_so_far", "delta_re", "rolling_std_re"]
)

assert len(FEATURE_COLS) == 99, f"Expected 99 features, got {len(FEATURE_COLS)}"

# V2 = rebuilt offline pipeline (features_history.py); FEATURE_COLS stays for the legacy 99-dim online model.
# is_comment: row is a comment. has_parent: comment whose parent text is known, i.e. sim/LSM are real values.
FEATURE_COLS_BASE = FEATURE_COLS + ["is_comment", "has_parent"]

# Rate of each LIWC category in the user's own text, a few behaviour features, and running means (history).
OWN_LIWC_CATS = ["I", "NEGEMO", "POSEMO", "SAD", "ANX", "ANGER", "DEATH", "FEEL", "COGPROC", "SOCIAL", "HEALTH",
                 "BIO", "FOCUSPAST", "FOCUSFUTURE", "NEGATE"]
OWN_LIWC_COLS = [f"liwc_{c}" for c in OWN_LIWC_CATS]
# log word count; posted between 00:00 and 05:59 UTC; log1p(seconds since the user's previous post, 0 for the first)
BEHAVIOR_COLS = ["log_tokens", "night", "log_gap"]
CUMMEAN_BASE = ["phq_score", "re", "sim"] + LSM_BASE_FEATURES + OWN_LIWC_COLS + BEHAVIOR_COLS
CUMMEAN_COLS = [f"{c}_cummean" for c in CUMMEAN_BASE]

FEATURE_COLS_V2 = FEATURE_COLS_BASE + OWN_LIWC_COLS + BEHAVIOR_COLS + CUMMEAN_COLS

# LSMX (post_features.compute_lsmx), parent rows only like sim/LSM: LSM on word rates instead of counts (lsmr_), signed
# body-minus-parent group rate (dlsm_), parent group rate (par_) and parent length. Raw value + running mean only.
LSMX_COLS = ([f"{p}_{g}" for p in ("lsmr", "dlsm", "par") for g in LSM_BASE_FEATURES] + ["par_log_tokens"])
LSMX_CUMMEAN_COLS = [f"{c}_cummean" for c in LSMX_COLS]
FEATURE_COLS_V3 = FEATURE_COLS_V2 + LSMX_COLS + LSMX_CUMMEAN_COLS

# Feature groups for the ablation study; every V2 column belongs to exactly one group.
FEATURE_GROUPS_V2 = {
    "PHQ": PHQ_COLS + ["phq_score", "max_phq_so_far", "min_phq_so_far", "max_gap", "delta_phq", "rolling_std_phq",
                       "phq_score_cummean"]
           + [f"max_{i}_so_far" for i in range(9)],
    "SIM": ["sim", "max_sim_so_far", "min_sim_so_far", "max_gap_sim_so_far", "delta_sim", "rolling_std_sim",
            "sim_cummean"],
    "LSM": LSM_BASE_FEATURES + [f"{f}_{s}" for f in LSM_BASE_FEATURES
                                for s in ["max_so_far", "min_so_far", "max_gap_so_far", "delta", "rolling_std"]]
           + [f"{f}_cummean" for f in LSM_BASE_FEATURES],
    "RE": ["loss_dep0", "loss_dep1", "re", "max_re_so_far", "min_re_so_far", "max_gap_re_so_far", "delta_re",
           "rolling_std_re", "re_cummean"],
    "OWN_LIWC": OWN_LIWC_COLS + [f"{c}_cummean" for c in OWN_LIWC_COLS],
    "BEHAVIOR": BEHAVIOR_COLS + [f"{c}_cummean" for c in BEHAVIOR_COLS],
    "CONTEXT": ["post_index", "is_comment", "has_parent"],
}
_grouped = [c for g in FEATURE_GROUPS_V2.values() for c in g]
assert sorted(_grouped) == sorted(FEATURE_COLS_V2), "feature groups must partition FEATURE_COLS_V2"

# Features that describe a single post, with no history.
POST_LEVEL_V2 = (PHQ_COLS + ["phq_score", "sim"] + LSM_BASE_FEATURES + ["loss_dep0", "loss_dep1", "re"]
                 + ["is_comment", "has_parent"] + OWN_LIWC_COLS + [c for c in BEHAVIOR_COLS if c != "log_gap"])


def _keep(cols):
    wanted = set(cols)
    return [c for c in FEATURE_COLS_V2 if c in wanted]


# name -> columns. The context features stay in every single-group run so the model knows which rows are comments.
# base_features is the feature set before the relative-entropy fix, own-text LIWC, behaviour and running means.
EXPERIMENTS_V2 = {"all": list(FEATURE_COLS_V2), "post_level_only": _keep(POST_LEVEL_V2),
                  "base_features": _keep(FEATURE_COLS_BASE), "no_cummean": _keep(set(FEATURE_COLS_V2) - set(CUMMEAN_COLS))}
for _g in ("PHQ", "SIM", "LSM", "RE", "OWN_LIWC", "BEHAVIOR"):
    EXPERIMENTS_V2[f"only_{_g}"] = _keep(FEATURE_GROUPS_V2[_g] + FEATURE_GROUPS_V2["CONTEXT"])
    EXPERIMENTS_V2[f"without_{_g}"] = _keep(set(FEATURE_COLS_V2) - set(FEATURE_GROUPS_V2[_g]))

# lean: only the signals that held up in the ablations (PHQ, RE, three LIWC categories, four LSM sub-scores); no sim.
LEAN_LIWC = ["liwc_HEALTH", "liwc_ANX", "liwc_SAD"]
LEAN_LSM = ["negative_mean", "perceptual_process", "social_mean", "first_person_singular"]
_lean = (FEATURE_GROUPS_V2["PHQ"] + FEATURE_GROUPS_V2["CONTEXT"] + ["re", "re_cummean"]
         + [f"{c}{s}" for c in LEAN_LIWC + LEAN_LSM for s in ("", "_cummean")])
EXPERIMENTS_V2["lean"] = _keep(_lean)

# Ablation inside lean: drop one part (lean_without_X) or add one part to PHQ + context (lean_PHQ_plus_X).
_LEAN_PARTS = {"RE": ["re", "re_cummean"],
               "LIWC": [f"{c}{s}" for c in LEAN_LIWC for s in ("", "_cummean")],
               "LSM": [f"{c}{s}" for c in LEAN_LSM for s in ("", "_cummean")]}
_LEAN_PARTS["PHQ"] = FEATURE_GROUPS_V2["PHQ"]
_LEAN_PARTS["cummean"] = [c for c in _lean if c.endswith("_cummean")]
_LEAN_PARTS["post_index"] = ["post_index"]
for _p, _cols in _LEAN_PARTS.items():
    EXPERIMENTS_V2[f"lean_without_{_p}"] = _keep(set(_lean) - set(_cols))
_phq_ctx = FEATURE_GROUPS_V2["PHQ"] + FEATURE_GROUPS_V2["CONTEXT"]
for _p in ("RE", "LIWC", "LSM"):
    EXPERIMENTS_V2[f"lean_PHQ_plus_{_p}"] = _keep(_phq_ctx + _LEAN_PARTS[_p])

# Redefined LSM (LSMX). Needs Features/v2 rebuilt with the lsmx piece; the older experiments above are unaffected.
_lsmx = LSMX_COLS + LSMX_CUMMEAN_COLS
_ctx = FEATURE_GROUPS_V2["CONTEXT"]


def _keep_v3(cols):
    wanted = set(cols)
    return [c for c in FEATURE_COLS_V3 if c in wanted]


EXPERIMENTS_V2["only_LSMX"] = _keep_v3(_lsmx + _ctx)
EXPERIMENTS_V2["lean_PHQ_plus_LSMX"] = _keep_v3(_phq_ctx + _lsmx)
EXPERIMENTS_V2["lean_plus_LSMX"] = _keep_v3(_lean + _lsmx)
EXPERIMENTS_V2["all_v3"] = list(FEATURE_COLS_V3)

# Is the LSMX gain style alignment (lsmr_, dlsm_) or just what the user replies to (par_ = parent's own rates)?
_par = [c for c in _lsmx if c.startswith("par_")]
_align = [c for c in _lsmx if not c.startswith("par_")]
EXPERIMENTS_V2["only_LSMX_no_par"] = _keep_v3(_align + _ctx)
EXPERIMENTS_V2["only_LSMX_par"] = _keep_v3(_par + _ctx)
EXPERIMENTS_V2["lean_PHQ_plus_LSMX_no_par"] = _keep_v3(_phq_ctx + _align)
EXPERIMENTS_V2["lean_PHQ_plus_LSMX_par"] = _keep_v3(_phq_ctx + _par)

# lean2: PHQ + RE + three LIWC categories + style-alignment LSM (rate-based LSM and signed difference), no parent rates.
EXPERIMENTS_V2["lean2"] = _keep_v3(_phq_ctx + _LEAN_PARTS["RE"] + _LEAN_PARTS["LIWC"] + _align)

# Leave-one-part-out inside lean2.
_lean2 = EXPERIMENTS_V2["lean2"]
_LEAN2_PARTS = {"PHQ": FEATURE_GROUPS_V2["PHQ"], "RE": _LEAN_PARTS["RE"], "LIWC": _LEAN_PARTS["LIWC"],
                "LSMalign": _align, "lsmr": [c for c in _align if c.startswith("lsmr_")],
                "dlsm": [c for c in _align if c.startswith("dlsm_")],
                "cummean": [c for c in _lean2 if c.endswith("_cummean")], "post_index": ["post_index"]}
for _p, _cols in _LEAN2_PARTS.items():
    EXPERIMENTS_V2[f"lean2_without_{_p}"] = _keep_v3(set(_lean2) - set(_cols))

# Frozen MentalRoBERTa post embeddings (embed_posts.py), fed to the same sequence model as every other experiment.
MENTAL_DIM = 768
MENTAL_COLS = [f"mental_{i}" for i in range(MENTAL_DIM)]
EXPERIMENTS_V2["mental_emb"] = MENTAL_COLS + _keep_v3(_ctx)
EXPERIMENTS_V2["mental_plus_lean2"] = MENTAL_COLS + _lean2

# Adapter-fine-tuned MentalRoBERTa (finetune_mental.py): one cross-fitted post-level logit, then the same sequence model.
EXPERIMENTS_V2["ft_logit"] = ["ft_logit"] + _keep_v3(_ctx)
EXPERIMENTS_V2["ft_logit_plus_lean2"] = ["ft_logit"] + _lean2

# RE-free combinations used by the repeated cross-validation (no feature is trained on labels except ft_logit).
EXPERIMENTS_V2["mental_plus_lean2_without_RE"] = MENTAL_COLS + EXPERIMENTS_V2["lean2_without_RE"]
EXPERIMENTS_V2["ft_logit_plus_lean2_without_RE"] = ["ft_logit"] + EXPERIMENTS_V2["lean2_without_RE"]
