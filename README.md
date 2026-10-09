# Early depression detection from Reddit post sequences (eRisk 2025 Task 2)

Offline pipeline and experiments for early depression detection on the eRisk 2025 Task 2 data. For every user the posts
are read in time order and a probability of depression is produced after each post; an alarm is raised once the
probability stays at or above a threshold `th` for `k` consecutive posts. The probabilities come from a small causal
Transformer over per-post features (PHQ-9 item similarity, LIWC, LSM, relative entropy, running statistics), and the
pipeline is compared with MentalRoBERTa (frozen and adapter fine-tuned) and a TF-IDF logistic-regression baseline.

**This is research code, not a diagnostic tool.** It must not be used to screen or label real people.

## Results in one table

5×3 repeated stratified cross-validation over all 891 labelled users (102 depressed), mean over repeats with 95 %
bootstrap intervals over users; the decision rule `(th, k)` is chosen on inner out-of-fold predictions only
(full protocol in `docs/SYSTEM_REPORT_EN.md`, Section 10.5).

| Configuration | ROC | PR | F1 | F_latency |
|---|---|---|---|---|
| `ft_logit` (adapter fine-tuned MentalRoBERTa) | 0.963 [0.947, 0.977] | 0.792 [0.732, 0.852] | 0.788 | 0.762 |
| `lean_without_RE` | 0.958 | 0.777 | 0.751 | 0.713 |
| `lean2_without_RE` | 0.953 | 0.753 | 0.723 | 0.680 |
| `only_PHQ` | 0.952 | 0.742 | 0.718 | 0.681 |
| `mental_emb` (frozen MentalRoBERTa) | 0.938 | 0.726 | 0.729 | 0.708 |
| `tfidf_lr` (bag of words) | 0.914 | 0.680 | 0.715 | 0.702 |

What the evidence supports: adapter fine-tuning is significantly better than frozen embeddings and than TF-IDF;
features beyond PHQ-9 similarity (LIWC, LSM alignment) show no demonstrable gain; the TF-IDF baseline ranks users worse
but alarms as early as or earlier than the other methods. Intervals are wide (about ±0.07 PR) because there are only 102 positives.

Error analysis (report Section 10.6): most missed depressed users do disclose their depression, but in a few posts within
long histories of unrelated comments; about a third of the false alarms are control users who describe their own
depression. A rule without any training that alarms at the first self-reported diagnosis reaches F1 0.783, close to the
best model, but alarms much later (median 55 posts), so the learned models' advantage is mainly earliness.
The hand-crafted `lean2` fails differently (Section 10.7): it misses users with an explicit disclosure but a short history and
flags control users who write about health topics.

## Data

The data are **not** in this repository. Get the eRisk 2025 Task 2 collection from the organisers
(<https://erisk.irlab.org/2025/index.html>) and follow its data-use agreement. Expected locations (override with
environment variables, see `paths.py`):

- `Data/all_combined/` raw thread JSON files (`ERISK_THREADS_DIR`)
- `Data/shuffled_ground_truth_labels.txt` user id → 0/1 (`ERISK_LABELS_FILE`)

`splits/user_split.csv` is the fixed user-level train/test assignment (712 / 179 users) and is tracked.

## Setup

```bash
conda create -n hongyi python=3.12 && conda activate hongyi
pip install -r requirements.txt          # install the CUDA build of torch that matches your driver
git submodule update --init              # extremism/ : PHQ-9 item scoring
```

- **PHQ-9 scoring** uses the `extremism` submodule (our fork, branch `my-changes`, which includes a fix to
  `item-scoring/item_scoring.py`: `--scale` defaults to `['all']` when omitted).
- **LIWC**: `liwc.py` and `liwc_script.py` are not tracked (dictionary-derived, licence restrictions). Put them in this
  directory; `liwc_script.py` is called by `post_features.run_liwc`. NLTK data goes to `nltk_data/` (`NLTK_DATA`).
- Two GPUs were used (RTX 4090); one is enough, the adapter fine-tuning needs about 3.4 GB.

## Running

```bash
# 1) data and features: run data-processing.ipynb top to bottom -> Results/v2/, Features/v2/{train,test}_features.pkl

# 2) frozen MentalRoBERTa embeddings (optional)
python -u embed_posts.py train --device cuda:0 && python -u embed_posts.py test --device cuda:0

# 2b) adapter fine-tuned MentalRoBERTa, cross-fitted over 3 user-level folds, then combined into the feature ft_logit
for k in 0 1 2; do python -u finetune_mental.py fold --fold $k --device cuda:0; done
python finetune_mental.py combine

# 3) single split, 3 seeds (names are keys of feature_cols.EXPERIMENTS_V2; --parent-only for the parent-only subset)
python -u run_experiments.py lean2 lean only_PHQ all --seeds 42 43 44 --device cuda:0

# 4) repeated cross-validation, baseline and significance tests (results in models/v2/cv_all/)
python -u cv_ft.py --repeat 0 --outer 0 --device cuda:0      # one job per (repeat 0..2, outer fold 0..4)
python -u cv_all.py --repeats 3 --shard 0 1 --device cuda:0
python -u cv_tfidf.py --repeats 3
python cv_significance.py --repeats 3
python error_analysis.py                                     # aggregates to docs/figures/, excerpts stay in Results/
python error_analysis.py lean2_without_RE                    # same for another model (outputs get a suffix)

# 5) figures and the PDF report
python docs/make_figures.py && python docs/export_pdf.py
```

Seed 42 keeps the plain experiment name; other seeds are stored as `<name>_seed<seed>`.

## Layout

| Path | Purpose |
|---|---|
| `paths.py`, `data_prep.py` | locations, reading and cleaning threads, user split, time sorting |
| `post_features.py`, `features_history.py`, `feature_cols.py` | post-level features (LIWC, LSM/LSMX, RE, cosine, PHQ), history features, feature groups and all experiment definitions |
| `seq_model.py`, `seq_eval.py`, `experiment.py` | causal Transformer, alarm rule and metrics, 5-fold training and evaluation |
| `run_experiments.py`, `run_grid.py` | command-line entry points |
| `embed_posts.py`, `finetune_mental.py` | MentalRoBERTa baselines |
| `cv_splits.py`, `cv_ft.py`, `cv_all.py`, `cv_tfidf.py`, `cv_significance.py` | repeated cross-validation, TF-IDF baseline, paired tests |
| `error_analysis.py` | false positives and negatives, self-report rule baseline |
| `benchmark.py` | per-post inference cost (`docs/benchmark_results.json`) |
| `docs/` | `SYSTEM_REPORT_EN.md` (full report, with PDF), figure and PDF scripts |
| `tests/` | synthetic-data checks |

Run the tests with `python tests/test_features_history.py` (and the other files in `tests/`).

## Known limitations

- Only 102 depressed users: most differences between feature sets are not resolvable.
- The relative-entropy feature is fitted on labels with cross-fitting and is therefore excluded from the cross-validation.
- The online client (`Test/`) is not tracked and has not been aligned with this pipeline. It reads the team token from the
  environment variable `ERISK_TEAM_TOKEN`; never commit tokens.
