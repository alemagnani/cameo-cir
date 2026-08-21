"""
MagicLens (JAX/Flax) as a Stage-1 composed-image-retrieval model on CAMEO (Approach #2).

Mirrors scripts/run_searle.py: encode the 4,075-product corpus (image + empty text ->
multimodal_embed_norm), encode each query (reference image + translated_question), rank corpus by
cosine, write run.json {qid:[product_ids]}. Run inside the .venv-magiclens env.

Usage:
  python scripts/run_magiclens_cir.py --queries data/queries_dev_full.json \
      --out data/runs/magiclens_large_dev.json --model-size large \
      --model-path related_papers/code/magiclens/models/magic_lens_clip_large.pkl --topk 50
"""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ML = ROOT / "related_papers" / "code" / "magiclens"
import sys
sys.path.insert(0, str(ML))

import cv2
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from PIL import Image
from flax import serialization
from tqdm import tqdm
from model import MagicLens  # noqa: E402
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "processed"
_PLACEHOLDER = Image.new("RGB", (224, 224), (128, 128, 128))


def load_pil(path: Path) -> Image.Image:
    try:
        if path.suffix.lower() == ".mp4":
            cap = cv2.VideoCapture(str(path)); ok, frame = cap.read(); cap.release()
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if ok else _PLACEHOLDER
        return Image.open(path).convert("RGB")
    except Exception:
        return _PLACEHOLDER


def process_img(pil: Image.Image, size: int = 224) -> np.ndarray:
    """Mirror data_utils.process_img: [1,H,W,3] -> /max -> bilinear resize to size."""
    ima = jnp.array(pil)[jnp.newaxis, ...]
    ima = ima / (ima.max() + 1e-12)
    ima = jax.image.resize(ima, (1, size, size, 3), method="bilinear")
    return np.array(ima)


def load_model(model_size, model_path):
    model = MagicLens(model_size)
    params = model.init(jax.random.PRNGKey(0),
                        {"ids": jnp.ones((1, 1, 77), jnp.int32), "image": jnp.ones((1, 224, 224, 3), jnp.float32)})
    with open(model_path, "rb") as f:
        params = serialization.from_bytes(params, pickle.load(f))
    return model, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-size", default="large", choices=["base", "large"])
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N products+queries")
    args = ap.parse_args()

    tokenizer = clip_tokenizer.build_tokenizer()
    null_tok = np.array(tokenizer(""))           # [1,77]
    model, params = load_model(args.model_size, args.model_path)
    print("model loaded")

    products = pd.read_parquet(RAW_DIR / "products__test.parquet").drop_duplicates("product_id").reset_index(drop=True)
    if args.limit:
        products = products.head(args.limit)
    pids = products["product_id"].astype(str).tolist()
    paths = products["image_path"].tolist()

    bb_tag = f"magiclens_{args.model_size}"
    cache = CACHE_DIR / f"{bb_tag}_index.npy"
    if cache.exists() and not args.limit:
        index = np.load(cache)
        print(f"loaded cached index {index.shape}")
    else:
        embs = []
        for i in tqdm(range(0, len(paths), args.batch_size), desc="encode corpus"):
            imgs = np.concatenate([process_img(load_pil(RAW_DIR / p)) for p in paths[i:i+args.batch_size]], 0)
            toks = np.concatenate([null_tok] * imgs.shape[0], 0)
            e = model.apply(params, {"ids": toks, "image": imgs})["multimodal_embed_norm"]
            embs.append(np.array(e))
        index = np.concatenate(embs, 0)
        if not args.limit:
            CACHE_DIR.mkdir(parents=True, exist_ok=True); np.save(cache, index)
        print(f"encoded corpus {index.shape}")

    pid2idx = {p: i for i, p in enumerate(pids)}
    queries = json.loads(Path(args.queries).read_text())
    runs = {}
    for qid, q in tqdm(list(queries.items()), desc="queries"):
        img = process_img(load_pil(RAW_DIR / q["image_path"]))
        cap = q.get("translated_question") or q.get("question") or ""
        tok = np.array(tokenizer(cap))
        qe = np.array(model.apply(params, {"ids": tok, "image": img})["multimodal_embed_norm"])  # [1,D]
        sims = (qe @ index.T)[0]
        ref = str(q.get("product_id"))
        if ref in pid2idx:
            sims[pid2idx[ref]] = -1e9
        top = np.argsort(sims)[-args.topk:][::-1]
        runs[qid] = [pids[i] for i in top]

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(runs))
    print(f"wrote {len(runs)} runs -> {out}")


if __name__ == "__main__":
    main()
