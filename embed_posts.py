"""Frozen MentalRoBERTa embeddings of every post body (mean-pooled last layer), saved next to the feature tables.

    nohup conda run -n hongyi python -u embed_posts.py train --device cuda:0 > Results/v2/embed_train.log 2>&1 &

Row i of Features/v2/<split>_mental_emb.npy is the post with row_id i in Results/v2/<split>.csv (float16).
"""

import argparse

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

import data_prep as dp
from paths import Paths

MODEL = "mental/mental-roberta-base"
MAX_LEN = 256


@torch.no_grad()
def embed(texts, device, batch_size=128, model_name=MODEL):
    """Mean-pooled embeddings (n, hidden) as float16; texts are batched by length so padding stays small."""
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device).eval()
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float16)
    for start in range(0, len(texts), batch_size):
        idx = order[start:start + batch_size]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=MAX_LEN,
                  return_tensors="pt").to(device)
        hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        out[idx] = ((hidden * mask).sum(1) / mask.sum(1)).cpu().numpy()
        if (start // batch_size) % 200 == 0:
            print(f"{start}/{len(texts)}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("split", choices=["train", "test"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    P = Paths()
    table = dp.read_table(P.split_csv(args.split), usecols=["row_id", "body"])
    if not (table["row_id"].to_numpy() == np.arange(len(table))).all():
        raise ValueError("row_id must be the row position in the split table")
    emb = embed(table["body"].astype(str).tolist(), args.device)
    if not np.isfinite(emb.astype(np.float32)).all():
        raise ValueError("non-finite values in the embeddings")
    np.save(P.embeddings(args.split), emb)
    print(args.split, emb.shape, "->", P.embeddings(args.split), flush=True)


if __name__ == "__main__":
    main()
