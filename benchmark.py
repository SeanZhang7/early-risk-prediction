"""Measure the computing cost of every part of the system (per-post inference, sequence model, training).

    conda run -n hongyi python benchmark.py --device cuda:0 --n-gpu 200 --n-cpu 96 --out docs/benchmark_results.json

Per-post times use real test posts (a fixed random sample), discarded warm-up calls and, on GPU,
torch.cuda.synchronize() around every call. Model loading is reported separately and never included in
the per-post times. GPU memory is the peak above what was already allocated before the component was loaded.
On GPU every encoder is measured in fp32 and fp16 so that models are compared at the same precision.
Nothing here changes any experiment result.
"""
import argparse
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from paths import Paths  # noqa: E402

P = Paths()
os.environ.setdefault("NLTK_DATA", str(P.nltk_data))
RESULTS = []
BATCH = 32


def is_cuda(device):
    return str(device).startswith("cuda")


def sync(device):
    if is_cuda(device):
        torch.cuda.synchronize(device)


def timed(fn, items, device, warmup=3):
    """Per-call wall time in ms for fn(item) over items (the first `warmup` calls are discarded)."""
    for it in items[:warmup]:
        fn(it)
    sync(device)
    ts = []
    for it in items:
        sync(device)
        t0 = time.perf_counter()
        fn(it)
        sync(device)
        ts.append((time.perf_counter() - t0) * 1000)
    return np.array(ts)


def batch_ms_per_post(fn, bodies, device):
    """ms per post when posts are processed BATCH at a time; the first batch is a discarded warm-up."""
    fn(bodies[:BATCH])
    sync(device)
    rest = bodies[BATCH:]
    t0 = time.perf_counter()
    for s in range(0, len(rest), BATCH):
        fn(rest[s:s + BATCH])
    sync(device)
    return (time.perf_counter() - t0) * 1000 / len(rest)


class Mem:
    """Peak GPU memory of a component above what was allocated before it was loaded."""

    def __init__(self, device):
        self.device = device
        gc.collect()
        if is_cuda(device):
            torch.cuda.empty_cache()
            self.base = torch.cuda.memory_allocated(device)

    def reset(self):
        if is_cuda(self.device):
            torch.cuda.reset_peak_memory_stats(self.device)

    def peak(self):
        return (torch.cuda.max_memory_allocated(self.device) - self.base) / 1e9 if is_cuda(self.device) else None


def record(component, setting, device, ts=None, n_params=None, per_item_ms=None, peak_gb=None, note="", batch=1):
    row = {"component": component, "setting": setting, "device": str(device), "batch": batch,
           "n_params_M": None if n_params is None else round(n_params / 1e6, 1),
           "peak_gpu_mem_GB": None if peak_gb is None else round(peak_gb, 2), "note": note}
    if ts is not None:
        row.update({"ms_mean": round(float(ts.mean()), 2), "ms_std": round(float(ts.std(ddof=1)), 2),
                    "ms_p50": round(float(np.median(ts)), 2), "ms_p95": round(float(np.percentile(ts, 95)), 2),
                    "n_calls": len(ts)})
    if per_item_ms is not None:
        row["ms_per_post"] = round(float(per_item_ms), 2)
        row["posts_per_s"] = round(1000.0 / float(per_item_ms), 1)
    RESULTS.append(row)
    print(row, flush=True)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def label(device, threads, half):
    if is_cuda(device):
        return "GPU " + ("fp16" if half else "fp32")
    return f"CPU {threads} thread(s) fp32"


def sample_posts(n, seed=0):
    d = pd.read_csv(P.split_csv("test"), usecols=["type", "body", "parent"], keep_default_na=False)
    d = d[d["body"].str.strip().ne("")]
    d = d.sample(n=n, random_state=seed).reset_index(drop=True)
    return d["body"].astype(str).tolist(), d["parent"].astype(str).tolist()


def bench_liwc(bodies, parents):
    t0 = time.perf_counter()
    import liwc
    load_s = time.perf_counter() - t0
    liwc.get_all_liwc_parts(liwc.preprocess("warm up"))

    def one(text):
        liwc.get_all_liwc_parts(liwc.preprocess(text))

    record("LIWC, body only", "CPU 1 thread, batch 1", "cpu", timed(one, bodies, "cpu"),
           note=f"one-time module import {load_s:.1f} s")
    pairs = list(zip(bodies, parents))
    record("LIWC, body + parent (comments)", "CPU 1 thread, batch 1", "cpu",
           timed(lambda bp: (one(bp[0]), one(bp[1])), pairs, "cpu"))


def bench_phq(bodies, device, threads, half, name="mixedbread-ai/mxbai-embed-large-v1"):
    from sentence_transformers import SentenceTransformer, util
    torch.set_num_threads(threads)
    mem = Mem(device)
    t0 = time.perf_counter()
    model = SentenceTransformer(name, device=str(device))
    if half and is_cuda(device):
        model.half()
    load_s = time.perf_counter() - t0
    items = [l.strip() for l in open(ROOT / "extremism" / "data" / "scales" / "PHQ-9_archetype_scale.txt") if l.strip()]
    item_emb = model.encode(items, convert_to_tensor=True)
    mem.reset()

    def encode_and_score(texts):
        e = model.encode(texts, batch_size=BATCH, convert_to_tensor=True)
        util.cos_sim(e, item_emb)

    comp = "PHQ-9 encoder (mxbai-embed-large-v1)"
    ts = timed(lambda t: encode_and_score(t), bodies, device)
    record(comp, "batch 1, " + label(device, threads, half), device, ts, n_params(model), peak_gb=mem.peak(),
           note=f"load {load_s:.1f} s")
    mem.reset()
    record(comp, f"batch {BATCH}, " + label(device, threads, half), device, n_params=n_params(model),
           per_item_ms=batch_ms_per_post(encode_and_score, bodies, device), peak_gb=mem.peak(), batch=BATCH)
    del model, item_emb


def bench_mental(bodies, device, threads, half, name="mental/mental-roberta-base", max_len=256):
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(threads)
    mem = Mem(device)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, torch_dtype=torch.float16 if half and is_cuda(device) else torch.float32)
    model = model.to(device).eval()
    load_s = time.perf_counter() - t0
    mem.reset()

    @torch.no_grad()
    def emb(texts):
        enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1)

    comp = "MentalRoBERTa embedding, frozen"
    ts = timed(lambda t: emb([t]), bodies, device)
    record(comp, "batch 1, " + label(device, threads, half), device, ts, n_params(model), peak_gb=mem.peak(),
           note=f"load {load_s:.1f} s")
    mem.reset()
    record(comp, f"batch {BATCH}, " + label(device, threads, half), device, n_params=n_params(model),
           per_item_ms=batch_ms_per_post(emb, bodies, device), peak_gb=mem.peak(), batch=BATCH)
    del model


def bench_re(bodies, device, threads, half, max_len=256):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(threads)
    mem = Mem(device)
    dirs = [P.models_dir / f"re_fold0_Dep{c}_LM" for c in (0, 1)]
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(dirs[0])
    tok.pad_token = tok.eos_token
    models = [AutoModelForCausalLM.from_pretrained(d).to(device).eval() for d in dirs]
    if half and is_cuda(device):
        models = [m.half() for m in models]
    load_s = time.perf_counter() - t0
    mem.reset()

    @torch.no_grad()
    def pair(texts):
        out = []
        for model in models:
            enc = tok(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits[:, :-1].float()
            target, mask = enc["input_ids"][:, 1:], enc["attention_mask"][:, 1:]
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction="none")
            out.append((ce.view(target.shape) * mask).sum(1) / mask.sum(1).clamp(min=1))
        return out

    comp = "RE: 2 distilgpt2 models (one fold pair)"
    ts = timed(lambda t: pair([t]), bodies, device)
    record(comp, "batch 1, " + label(device, threads, half), device, ts, sum(n_params(m) for m in models),
           peak_gb=mem.peak(), note=f"load {load_s:.1f} s; the offline pipeline averages 3 such pairs (6 models)")
    mem.reset()
    record(comp, f"batch {BATCH}, " + label(device, threads, half), device,
           n_params=sum(n_params(m) for m in models), per_item_ms=batch_ms_per_post(pair, bodies, device),
           peak_gb=mem.peak(), batch=BATCH)
    del models


def bench_history(n_users=30):
    import experiment as ex
    from features_history import build_features
    _, test_df = ex.load_features(P)
    users = test_df["user_id"].drop_duplicates().sample(n=n_users, random_state=0)
    ts, posts = [], 0
    for u in users:
        d = test_df[test_df["user_id"] == u].reset_index(drop=True)
        t0 = time.perf_counter()
        build_features(d)
        ts.append((time.perf_counter() - t0) * 1000 / len(d))
        posts += len(d)
    record("History features (full rebuild of a user, per post)", f"{n_users} users, {posts} posts, CPU", "cpu",
           np.array(ts), note="upper bound: an online client could update incrementally")


def bench_sequence(device, threads, lengths=(10, 100, 1000), exp="lean2_without_RE"):
    import seq_model as sm
    from experiment import experiment_dir
    torch.set_num_threads(threads)
    mem = Mem(device)
    cols = json.load(open(experiment_dir(P, exp) / "scaler.json"))["cols"]
    model = sm.UserSequenceModel(len(cols), hidden_dim=256, n_heads=4, n_layers=3, dropout=0.3).to(device).eval()
    model.load_state_dict(torch.load(experiment_dir(P, exp) / "fold0.pt", map_location=device, weights_only=True))
    for n in lengths:
        x = torch.randn(1, n, len(cols), device=device)
        mask = torch.ones(1, n, dtype=torch.bool, device=device)
        mem.reset()

        @torch.no_grad()
        def fwd(_):
            model(x, mask)

        ts = timed(fwd, list(range(50 if n < 1000 else 20)), device)
        record(f"Sequence model ({exp}, {len(cols)} inputs)",
               f"one forward over {n} posts, " + label(device, threads, False), device, ts, n_params(model),
               peak_gb=mem.peak(), note="no KV cache: the whole history is re-encoded for every new post")
    del model


def bench_training(device, exps=("lean2_without_RE", "mental_emb")):
    import experiment as ex
    import seq_model as sm
    from feature_cols import EXPERIMENTS_V2
    from sklearn.model_selection import StratifiedKFold, train_test_split
    train_df, _ = ex.load_features(P)
    cfg = ex.Config()
    for exp in exps:
        cols = EXPERIMENTS_V2[exp]
        df = ex.attach_embeddings(P, train_df, "train") if exp.startswith("mental") else train_df
        scaler = sm.Standardizer.fit(df, cols)
        _, seqs, y = sm.make_sequences(df, scaler.transform(df))
        tr, _ = next(iter(StratifiedKFold(cfg.n_folds, shuffle=True, random_state=cfg.seed).split(np.zeros(len(y)), y)))
        fit_idx, val_idx = train_test_split(tr, test_size=cfg.inner_val_frac, stratify=y[tr], random_state=cfg.seed)
        kw = dict(hidden_dim=cfg.hidden_dim, n_heads=cfg.n_heads, n_layers=cfg.n_layers, dropout=cfg.dropout)
        args = ([seqs[i] for i in fit_idx], y[fit_idx], [seqs[i] for i in val_idx], y[val_idx], device)
        sm.fit(*args, epochs=1, lr=cfg.lr, patience=99, seed=0, verbose=False, **kw)  # warm-up
        mem = Mem(device)
        mem.reset()
        t0 = time.perf_counter()
        sm.fit(*args, epochs=3, lr=cfg.lr, patience=99, seed=0, verbose=False, **kw)
        sync(device)
        per_epoch = (time.perf_counter() - t0) / 3
        saved = json.load(open(ex.experiment_dir(P, exp) / "results.json"))
        epochs_run = sum(f["epochs_run"] for f in saved["folds"])
        record(f"Training one fold ({exp}, {len(cols)} inputs)", f"{len(fit_idx)} users, GPU fp32", device,
               peak_gb=mem.peak(), note=f"{per_epoch:.2f} s/epoch incl. validation; the saved 5-fold run used "
               f"{epochs_run} epochs in total, about {per_epoch * epochs_run / 60:.1f} min of pure training")
        RESULTS[-1].update({"s_per_epoch": round(per_epoch, 2), "epochs_total_5fold": epochs_run,
                            "est_minutes_5fold": round(per_epoch * epochs_run / 60, 2)})
        del seqs, df
        gc.collect()


def environment(device):
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if is_cuda(device) else None,
            "cpu": next((l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), None),
            "cpu_cores_visible": os.cpu_count()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-gpu", type=int, default=200, help="posts timed one by one on GPU, and for LIWC")
    ap.add_argument("--n-cpu", type=int, default=96, help="posts timed on CPU (must exceed 2 batches of 32)")
    ap.add_argument("--cpu-threads", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--out", default=str(ROOT / "docs" / "benchmark_results.json"))
    ap.add_argument("--skip-training", action="store_true")
    args = ap.parse_args()

    bodies, parents = sample_posts(max(args.n_gpu, args.n_cpu))
    words = np.array([len(b.split()) for b in bodies])
    print(f"sample: {len(bodies)} posts, words mean {words.mean():.0f}, median {np.median(words):.0f}", flush=True)
    bench_liwc(bodies[:args.n_gpu], parents[:args.n_gpu])
    bench_history()
    gpu_threads = torch.get_num_threads()
    for half in (False, True):
        sub = bodies[:args.n_gpu]
        bench_phq(sub, args.device, gpu_threads, half)
        bench_mental(sub, args.device, gpu_threads, half)
        bench_re(sub, args.device, gpu_threads, half)
    bench_sequence(args.device, gpu_threads)
    for thr in args.cpu_threads:
        sub = bodies[:args.n_cpu]
        bench_phq(sub, "cpu", thr, False)
        bench_mental(sub, "cpu", thr, False)
        bench_re(sub, "cpu", thr, False)
        bench_sequence("cpu", thr)
    if not args.skip_training:
        torch.set_num_threads(gpu_threads)
        bench_training(args.device)
    out = {"environment": environment(args.device), "sample_words_mean": float(words.mean()),
           "sample_words_median": float(np.median(words)), "results": RESULTS}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("saved", args.out)


if __name__ == "__main__":
    main()
