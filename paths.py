"""Every input and output location of the offline pipeline, in one place.

Defaults are relative to this directory; override with environment variables instead of editing code:
  ERISK_THREADS_DIR  folder with the raw thread JSON files   (default Data/all_combined)
  ERISK_LABELS_FILE  user_id -> 0/1 label file               (default Data/shuffled_ground_truth_labels.txt)
  NLTK_DATA          nltk resources used by the LIWC script  (default nltk_data)

Layout written by data-processing.ipynb (version "v2"):
  Results/v2/posts.csv                      cleaned target posts, all users
  Results/v2/{train,test}.csv               time-ordered split tables (row_id, re_fold)
  Results/v2/liwc/{split}_{body,parent}.fullframe.pickle
  Results/v2/parts/{split}_{lsm,re,sim,phq9}.pickle
  Features/v2/{split}_features.pkl          final model input
  models/v2/re_fold{k}_Dep{0,1}_LM          cross-fitted relative-entropy language models
  splits/user_split.csv                     fixed user -> train/test assignment (not distributed: contains labels)
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "test")


def _env(name, default):
    return Path(os.environ.get(name) or default).expanduser()


class Paths:
    def __init__(self, root=None, version="v2"):
        self.root = Path(root) if root else ROOT
        self.threads_dir = _env("ERISK_THREADS_DIR", self.root / "Data" / "all_combined")
        self.labels_file = _env("ERISK_LABELS_FILE", self.root / "Data" / "shuffled_ground_truth_labels.txt")
        self.nltk_data = _env("NLTK_DATA", self.root / "nltk_data")
        self.extremism_dir = self.root / "extremism"
        self.user_split_file = self.root / "splits" / "user_split.csv"
        self.work = self.root / "Results" / version
        self.features_dir = self.root / "Features" / version
        self.models_dir = self.root / "models" / version

    @property
    def posts(self):
        return self.work / "posts.csv"

    def split_csv(self, split):
        return self.work / f"{split}.csv"

    def liwc(self, split, which):
        """LIWC counts for the `body` or `parent` text; the file name is what liwc_script.py produces."""
        return self.work / "liwc" / f"{split}_{which}.fullframe.pickle"

    def part(self, split, name):
        """Post-level feature piece: one of lsm, re, sim, phq9."""
        return self.work / "parts" / f"{split}_{name}.pickle"

    def features(self, split):
        return self.features_dir / f"{split}_features.pkl"

    def embeddings(self, split, name="mental"):
        """Frozen sentence embeddings, one row per post in split-table order (row_id)."""
        return self.features_dir / f"{split}_{name}_emb.npy"

    def make_dirs(self):
        for d in (self.work / "liwc", self.work / "parts", self.features_dir, self.models_dir,
                  self.user_split_file.parent):
            d.mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        rows = [("threads", self.threads_dir), ("labels", self.labels_file), ("nltk_data", self.nltk_data),
                ("user split", self.user_split_file), ("work", self.work), ("features", self.features_dir),
                ("models", self.models_dir)]
        return "Paths(\n" + "\n".join(f"  {k:<10} {v}" for k, v in rows) + "\n)"
