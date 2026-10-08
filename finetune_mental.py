"""Cross-fitted adapter fine-tuning of MentalRoBERTa: one depression logit per post, leak-free for the sequence model.

    # one process per fold (user-level folds are the RE folds of data_prep), then combine
    nohup conda run -n hongyi python -u finetune_mental.py fold --fold 0 --device cuda:0 > Results/v2/ft_fold0.log 2>&1 &
    conda run -n hongyi python finetune_mental.py combine

A bottleneck adapter ("seq_bn", AdapterHub `adapters`) plus a classification head are trained on the posts of the
other folds' users (user label on every post, at most --posts-per-user random posts per user and epoch, positive
class weighted); the encoder stays frozen. The users of the held-out fold are scored by the model that never saw
them; test users are scored by the mean logit of the folds, exactly like the cross-fitted RE feature. The result
is stored as the post-level feature `ft_logit` (parts/<split>_ft_mental.pickle) and feeds the same sequence model.
"""
import argparse
import copy
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import data_prep as dp
from paths import Paths

MODEL = "mental/mental-roberta-base"


def part_path(P, seed, fold):
    return P.work / "parts" / f"ftmental_s{seed}_fold{fold}.pkl"


def build_model(device):
    from adapters import AutoAdapterModel
    model = AutoAdapterModel.from_pretrained(MODEL)
    model.add_adapter("dep", config="seq_bn")
    model.add_classification_head("dep", num_labels=2, overwrite_ok=True)
    model.train_adapter("dep")  # freezes the encoder, trains the adapter and the head
    model.set_active_adapters("dep")
    print("active adapters:", model.active_adapters, "| active head:", model.active_head, flush=True)
    return model.to(device)


def score(model, tok, texts, device, max_len, batch_size=128):
    """Logit difference (depressed minus control) for every text, in the order of `texts`."""
    model.eval()
    order = np.argsort([len(t) for t in texts])
    out = np.zeros(len(texts), dtype=np.float32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(texts), batch_size):
            idx = order[s:s + batch_size]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt").to(device)
            lg = model(**enc).logits.float()
            out[idx] = (lg[:, 1] - lg[:, 0]).cpu().numpy()
    return out


def epoch_batches(texts, order, batch_size, rng):
    """Batches of similar length inside chunks of 50 batches, then shuffled."""
    batches = []
    chunk = batch_size * 50
    for s in range(0, len(order), chunk):
        part = sorted(order[s:s + chunk], key=lambda i: len(texts[i]))
        batches += [part[b:b + batch_size] for b in range(0, len(part), batch_size)]
    return [batches[i] for i in rng.permutation(len(batches))]


def train_adapter(texts, labels, post_users, by_user, fit_users, val_users, args, device, tok, rng):
    """Train a fresh adapter + head on the posts of `fit_users`; keeps the epoch with the best post-level AUC on a
    sample of 50 posts from each of `val_users`. Returns (model, epoch log, trainable parameters, train seconds, peak GB)."""
    model = build_model(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train_params = sum(p.numel() for p in trainable)
    val_idx = np.concatenate([rng.permutation(by_user[u])[:50] for u in val_users])
    val_text = [texts[i] for i in val_idx]
    val_y = labels[val_idx]
    val_user = post_users[val_idx]

    steps_per_epoch = int(sum(min(len(by_user[u]), args.posts_per_user) for u in fit_users) / args.batch) + 1
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.06 * total_steps))
                                              * max(0.0, (total_steps - s) / total_steps))
    torch.cuda.reset_peak_memory_stats(device)
    best_auc, best_state, log, t_train = -1.0, None, [], 0.0
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        ex = np.concatenate([rng.permutation(by_user[u])[:args.posts_per_user] for u in fit_users])
        n_pos = int(labels[ex].sum())
        weight = torch.tensor([1.0, (len(ex) - n_pos) / max(n_pos, 1)], device=device)
        model.train()
        total, n = 0.0, 0
        for b in epoch_batches(texts, ex, args.batch, rng):
            enc = tok([texts[i] for i in b], padding=True, truncation=True, max_length=args.max_len,
                      return_tensors="pt").to(device)
            y = torch.tensor(labels[b], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**enc).logits
            loss = F.cross_entropy(logits.float(), y, weight=weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(b)
            n += len(b)
        torch.cuda.synchronize(device)
        t_train += time.perf_counter() - t0
        s_val = score(model, tok, val_text, device, args.max_len)
        post_auc = roc_auc_score(val_y, s_val)
        vdf = pd.DataFrame({"u": val_user, "y": val_y, "s": s_val}).groupby("u").agg(y=("y", "max"), s=("s", "mean"))
        user_auc = roc_auc_score(vdf["y"], vdf["s"])
        log.append({"epoch": epoch + 1, "train_loss": total / n, "val_post_auc": post_auc, "val_user_auc": user_auc,
                    "epoch_seconds": time.perf_counter() - t0, "examples": int(len(ex))})
        print(log[-1], flush=True)
        if post_auc > best_auc:
            best_auc = post_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if "adapter" in k or "head" in k}
    assert best_state, "no adapter or head weights found in the state dict"
    model.load_state_dict(best_state, strict=False)
    return model, log, n_train_params, t_train, torch.cuda.max_memory_allocated(device) / 1e9


def run_fold(args):
    P = Paths()
    rng = np.random.default_rng(args.seed + args.fold)
    torch.manual_seed(args.seed + args.fold)
    train = dp.read_table(P.split_csv("train"), usecols=["row_id", "user_id", "Depression", "body", "re_fold"])
    test = dp.read_table(P.split_csv("test"), usecols=["row_id", "body"])
    users = train.groupby("user_id").agg(y=("Depression", "max"), fold=("re_fold", "first"))
    if args.max_users:
        users = users.groupby("y", group_keys=False).apply(lambda g: g.sample(min(len(g), args.max_users), random_state=0))
        train = train[train["user_id"].isin(users.index)].reset_index(drop=True)
    fit_pool = users[users.fold != args.fold]
    held_users = users.index[users.fold == args.fold]
    fit_users, val_users = train_test_split(fit_pool.index, test_size=0.1, stratify=fit_pool["y"],
                                            random_state=args.seed)
    by_user = train.groupby("user_id").indices
    texts = train["body"].astype(str).tolist()
    labels = train["Depression"].to_numpy()
    print(f"fold {args.fold}: fit {len(fit_users)} users, val {len(val_users)}, held out {len(held_users)}", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model, log, n_train_params, t_train, peak_gb = train_adapter(
        texts, labels, train["user_id"].to_numpy(), by_user, fit_users, val_users, args, args.device, tok, rng)
    best_auc = max(e["val_post_auc"] for e in log)
    print(f"trainable parameters: {n_train_params / 1e6:.2f} M", flush=True)

    held_pos = np.flatnonzero(train["user_id"].isin(held_users).to_numpy())
    t0 = time.perf_counter()
    held_scores = score(model, tok, [texts[i] for i in held_pos], args.device, args.max_len)
    test_scores = score(model, tok, test["body"].astype(str).tolist(), args.device, args.max_len)
    print(f"scored {len(held_pos)} held-out and {len(test)} test posts in {time.perf_counter() - t0:.0f} s", flush=True)
    out = pd.concat([
        pd.DataFrame({"split": "train", "row_id": train["row_id"].to_numpy()[held_pos], "ft_logit": held_scores}),
        pd.DataFrame({"split": "test", "row_id": test["row_id"].to_numpy(), "ft_logit": test_scores})])
    part_path(P, args.seed, args.fold).parent.mkdir(parents=True, exist_ok=True)
    out.to_pickle(part_path(P, args.seed, args.fold))
    meta = {"fold": args.fold, "seed": args.seed, "trainable_params_M": n_train_params / 1e6, "train_seconds": t_train,
            "peak_gpu_gb": peak_gb, "best_val_post_auc": best_auc, "epochs": log, "args": vars(args)}
    json.dump(meta, open(str(part_path(P, args.seed, args.fold)).replace(".pkl", ".json"), "w"), indent=2)
    print("saved", part_path(P, args.seed, args.fold), flush=True)


def combine(args):
    P = Paths()
    frames = [pd.read_pickle(part_path(P, args.seed, k)) for k in range(dp.N_RE_FOLDS)]
    allf = pd.concat(frames)
    train = allf[allf.split == "train"].drop(columns="split")
    if train["row_id"].duplicated().any():
        raise ValueError("a train post was scored by more than one fold")
    test = allf[allf.split == "test"].groupby("row_id", as_index=False)["ft_logit"].mean()
    for split, d in (("train", train), ("test", test)):
        expected = len(dp.read_table(P.split_csv(split), usecols=["row_id"]))
        if len(d) != expected:
            raise ValueError(f"{split}: {len(d)} scored posts, expected {expected}")
        d.sort_values("row_id").to_pickle(P.part(split, "ft_mental"))
    for split, d in (("train", train), ("test", test)):
        t = dp.read_table(P.split_csv(split), usecols=["row_id", "user_id", "Depression"]).merge(d, on="row_id")
        g = t.groupby("user_id").agg(y=("Depression", "max"), s=("ft_logit", "mean"))
        print(f"{split}: user-level AUC of the mean ft_logit = {roc_auc_score(g.y, g.s):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fold")
    f.add_argument("--fold", type=int, required=True)
    f.add_argument("--device", default="cuda:0")
    f.add_argument("--epochs", type=int, default=3)
    f.add_argument("--posts-per-user", type=int, default=200)
    f.add_argument("--batch", type=int, default=32)
    f.add_argument("--lr", type=float, default=3e-4)
    f.add_argument("--max-len", type=int, default=256)
    f.add_argument("--seed", type=int, default=42)
    f.add_argument("--max-users", type=int, default=0, help="smoke test: keep this many users per class")
    c = sub.add_parser("combine")
    c.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run_fold(args) if args.cmd == "fold" else combine(args)


if __name__ == "__main__":
    main()
