"""User-sequence Transformer: standardisation, length-aware batching, training and prediction.

The model reads one user's posts in time order and outputs a depression probability after every post.
A causal mask lets position t see posts 0..t only, so the probability at t is what could be known after
reading t+1 posts. Training uses the user label at every position.
"""

import copy
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG1P_COLS = ("post_index",)


def set_seed(seed):
    import os
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device():
    """The CUDA device with the most free memory, else cpu."""
    if not torch.cuda.is_available():
        return "cpu"
    free = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
    return f"cuda:{max(free)[1]}"


class Standardizer:
    """Per-column z-score with log1p applied to post_index first; statistics come from the training rows only.
    Values are clipped to +-clip so a one-word post (a LIWC rate of 1.0) cannot dominate a batch."""

    def __init__(self, cols, mean, std, clip=10.0):
        self.cols, self.mean, self.std, self.clip = list(cols), np.asarray(mean, float), np.asarray(std, float), clip

    @staticmethod
    def _raw(df, cols):
        x = df[list(cols)].to_numpy(np.float64, copy=True)
        for c in LOG1P_COLS:
            if c in cols:
                x[:, list(cols).index(c)] = np.log1p(x[:, list(cols).index(c)])
        return x

    @classmethod
    def fit(cls, df, cols, clip=10.0):
        x = cls._raw(df, cols)
        std = x.std(axis=0)
        std[std < 1e-8] = 1.0
        return cls(cols, x.mean(axis=0), std, clip)

    def transform(self, df):
        z = (self._raw(df, self.cols) - self.mean) / self.std
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"cols": self.cols, "mean": self.mean.tolist(), "std": self.std.tolist(), "clip": self.clip}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["cols"], d["mean"], d["std"], d.get("clip", 10.0))


def make_sequences(df, x):
    """Split the row-aligned matrix `x` into one array per user. `df` must be sorted by user, then time."""
    uid = df["user_id"].to_numpy()
    change = np.flatnonzero(uid[1:] != uid[:-1]) + 1
    starts, ends = np.r_[0, change], np.r_[change, len(uid)]
    users = uid[starts]
    if len(set(users)) != len(users):
        raise ValueError("rows of one user are not contiguous; sort with features_history.sort_posts first")
    y = df["Depression"].to_numpy()
    labels = y[starts]
    if any((y[s:e] != y[s]).any() for s, e in zip(starts, ends)):
        raise ValueError("a user has more than one label")
    return users, [x[s:e] for s, e in zip(starts, ends)], labels.astype(int)


def make_batches(lengths, rng=None, max_users=32, attn_budget=8_000_000, chunk=512):
    """Index batches grouped by length. A batch never holds more than max_users users or more than
    attn_budget attention cells (users * longest length squared), except a single very long user."""
    lengths = np.asarray(lengths)
    order = rng.permutation(len(lengths)) if rng is not None else np.arange(len(lengths))
    batches = []
    for s in range(0, len(order), chunk):
        part = order[s:s + chunk]
        part = part[np.argsort(-lengths[part], kind="stable")]
        cur = []
        for i in part:
            if cur and (len(cur) + 1 > max_users or (len(cur) + 1) * lengths[cur[0]] ** 2 > attn_budget):
                batches.append(cur)
                cur = []
            cur.append(int(i))
        if cur:
            batches.append(cur)
    if rng is not None:
        batches = [batches[i] for i in rng.permutation(len(batches))]
    return batches


def collate(seqs, idx, device):
    longest = max(len(seqs[i]) for i in idx)
    x = np.zeros((len(idx), longest, seqs[idx[0]].shape[1]), dtype=np.float32)
    mask = np.zeros((len(idx), longest), dtype=bool)
    for r, i in enumerate(idx):
        x[r, :len(seqs[i])] = seqs[i]
        mask[r, :len(seqs[i])] = True
    return torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device)


class UserSequenceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, mask):
        """x: (B, T, F) features, mask: (B, T) True for real posts. Returns logits of shape (B, T)."""
        t = x.size(1)
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.encoder(self.input_proj(x), mask=causal, src_key_padding_mask=~mask)
        return self.head(h).squeeze(-1)


def sequence_loss(logits, mask, labels, pos_weight):
    """User label at every real position; each user counts once, whatever the number of posts."""
    target = labels[:, None].expand_as(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pos_weight)
    m = mask.float()
    return ((bce * m).sum(1) / m.sum(1)).mean()


@torch.no_grad()
def mean_loss(model, seqs, labels, pos_weight, device, **batch_kw):
    model.eval()
    total, n = 0.0, 0
    for idx in make_batches([len(s) for s in seqs], None, **batch_kw):
        x, mask = collate(seqs, idx, device)
        y = torch.tensor(labels[idx], dtype=torch.float32, device=device)
        total += sequence_loss(model(x, mask), mask, y, pos_weight).item() * len(idx)
        n += len(idx)
    return total / n


def fit(train_seqs, train_y, val_seqs, val_y, device, epochs=30, lr=1e-4, weight_decay=0.01, patience=5,
        seed=0, verbose=True, **model_kw):
    """Train with early stopping on the validation loss and return the model holding its best weights."""
    set_seed(seed)
    n_pos = int(train_y.sum())
    pos_weight = torch.tensor((len(train_y) - n_pos) / max(n_pos, 1), dtype=torch.float32, device=device)
    model = UserSequenceModel(train_seqs[0].shape[1], **model_kw).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    lengths = [len(s) for s in train_seqs]
    best_loss, best_state, best_epoch, history = np.inf, None, -1, []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for idx in make_batches(lengths, rng):
            x, mask = collate(train_seqs, idx, device)
            y = torch.tensor(train_y[idx], dtype=torch.float32, device=device)
            loss = sequence_loss(model(x, mask), mask, y, pos_weight)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        train_loss = total / len(train_seqs)
        val_loss = mean_loss(model, val_seqs, val_y, pos_weight, device)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if verbose:
            print(f"epoch {epoch + 1:2d}  train {train_loss:.4f}  val {val_loss:.4f}")
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    model.best_epoch = best_epoch + 1
    return model, history


@torch.no_grad()
def predict(model, seqs, device, attn_budget=32_000_000):
    """Probability after every post, as one array per user."""
    model.eval()
    lengths = [len(s) for s in seqs]
    out = [None] * len(seqs)
    for idx in make_batches(lengths, None, attn_budget=attn_budget, chunk=len(seqs)):
        x, mask = collate(seqs, idx, device)
        p = torch.sigmoid(model(x, mask)).cpu().numpy()
        for r, i in enumerate(idx):
            out[i] = p[r, :lengths[i]]
    return out
