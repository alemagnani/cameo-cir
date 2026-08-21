"""
EXP-#2 — training-free composed image retrieval with a stronger open_clip backbone.

Motivation: the SEARLE-XL pipeline is capped at OpenAI CLIP ViT-L/14 and plateaus
at wide-dev NDCG@10 ~0.162. This script tests whether simply swapping in a stronger
open_clip backbone (ViT-H/14, SigLIP, ...) with a training-free image+text composition
raises the Stage-1 ceiling.

Composition (training-free CIR, no textual inversion):
    q = normalize( w_img * img_emb(ref) + w_txt * txt_emb(caption) )

Optionally fuse corpus *name* text into the index, mirroring run_searle.py's --fuse-text:
    index = normalize( alpha * txt_emb(name) + (1-alpha) * img_emb(product) )

Runs fully self-contained on the GPU box (no SEARLE hub, no OpenAI clip.load).
Output run JSON {qid: [top-K product_ids]} — same format eval_by_rule.py / evaluate.py read.

Usage (on the box):
    pip install open_clip_torch
    python scripts/exp004/run_openclip_cir.py \
        --queries data/queries_dev_full.json --out data/runs/openclip_vith_dev.json \
        --model ViT-H-14 --pretrained laion2b_s32b_b79k
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
PRODUCTS_PARQUET = ROOT / "data" / "raw" / "products__test.parquet"
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "processed"

_PLACEHOLDER = Image.new("RGB", (224, 224), color=(128, 128, 128))


def load_product_image(path: Path) -> Image.Image:
    """Open product asset as RGB. .mp4 -> first frame; corrupt/missing -> gray placeholder."""
    try:
        if path.suffix.lower() == ".mp4":
            cap = cv2.VideoCapture(str(path))
            ok, frame = cap.read()
            cap.release()
            if not ok:
                return _PLACEHOLDER
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return Image.open(path).convert("RGB")
    except Exception:
        return _PLACEHOLDER


@torch.no_grad()
def encode_corpus_images(model, preprocess, products, device, batch_size):
    feats = []
    pids = products["product_id"].astype(str).tolist()
    paths = products["image_path"].tolist()
    for i in tqdm(range(0, len(paths), batch_size), desc="encode corpus (image)"):
        imgs = [preprocess(load_product_image(RAW_DIR / p)) for p in paths[i:i + batch_size]]
        batch = torch.stack(imgs).to(device)
        f = model.encode_image(batch).float()
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu())
    return pids, torch.cat(feats, dim=0)


@torch.no_grad()
def encode_corpus_names(model, tokenizer, products, device, batch_size=64):
    names = products["translated_name"].fillna("").astype(str).tolist()
    feats = []
    for i in tqdm(range(0, len(names), batch_size), desc="encode corpus (name)"):
        tok = tokenizer(names[i:i + batch_size]).to(device)
        f = model.encode_text(tok).float()
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def run_queries(model, preprocess, tokenizer, queries, device,
                index_pids, index_feats, topk, w_img, w_txt, caption_field):
    pid2idx = {pid: i for i, pid in enumerate(index_pids)}
    index_feats = index_feats.to(device)
    runs = {}
    for qid, q in tqdm(list(queries.items()), desc="queries"):
        img = load_product_image(RAW_DIR / q["image_path"])
        img_t = preprocess(img).unsqueeze(0).to(device)
        img_feat = model.encode_image(img_t).float()
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        caption = q.get(caption_field) or q.get("translated_question") or q["question"]
        tok = tokenizer([caption]).to(device)
        txt_feat = model.encode_text(tok).float()
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        q_feat = w_img * img_feat + w_txt * txt_feat
        q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)

        sims = (q_feat @ index_feats.T).squeeze(0).cpu()
        ref_pid = str(q["product_id"])
        if ref_pid in pid2idx:
            sims[pid2idx[ref_pid]] = -float("inf")
        top = torch.topk(sims, k=min(topk, sims.shape[0])).indices.tolist()
        runs[qid] = [index_pids[i] for i in top]
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="ViT-H-14")
    ap.add_argument("--pretrained", default="laion2b_s32b_b79k")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--caption-field", default="translated_question")
    ap.add_argument("--w-img", type=float, default=1.0)
    ap.add_argument("--w-txt", type=float, default=1.0)
    ap.add_argument("--fuse-name-alpha", type=float, default=0.0,
                    help="alpha>0: index = alpha*name_text + (1-alpha)*image, renormalized.")
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[skip] output already exists -> {out_path}")
        return

    device = args.device
    print(f"device={device}  model={args.model}  pretrained={args.pretrained}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device)
    model = model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)

    bb_tag = f"openclip_{args.model}_{args.pretrained}".replace("/", "_").replace("-", "_").lower()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    img_cache = CACHE_DIR / f"{bb_tag}_index.pt"

    if img_cache.exists() and not args.rebuild_index:
        print(f"loading cached image index -> {img_cache.name}")
        blob = torch.load(img_cache, map_location="cpu")
        index_pids, image_feats = blob["pids"], blob["feats"]
    else:
        products = pd.read_parquet(PRODUCTS_PARQUET).drop_duplicates("product_id").reset_index(drop=True)
        print(f"encoding {len(products)} products (image)...")
        index_pids, image_feats = encode_corpus_images(model, preprocess, products, device, args.batch_size)
        torch.save({"pids": index_pids, "feats": image_feats}, img_cache)
        print(f"  cached -> {img_cache.name}")

    if args.fuse_name_alpha > 0.0:
        txt_cache = CACHE_DIR / f"{bb_tag}_index_text_name.pt"
        if txt_cache.exists() and not args.rebuild_index:
            text_feats = torch.load(txt_cache, map_location="cpu")["feats"]
        else:
            products = pd.read_parquet(PRODUCTS_PARQUET).drop_duplicates("product_id").reset_index(drop=True)
            assert products["product_id"].astype(str).tolist() == index_pids
            text_feats = encode_corpus_names(model, tokenizer, products, device)
            torch.save({"pids": index_pids, "feats": text_feats}, txt_cache)
        a = args.fuse_name_alpha
        fused = a * text_feats + (1.0 - a) * image_feats
        index_feats = fused / fused.norm(dim=-1, keepdim=True)
        print(f"fused index with name alpha={a}")
    else:
        index_feats = image_feats

    queries = json.loads(Path(args.queries).read_text())
    print(f"running {len(queries)} queries; w_img={args.w_img} w_txt={args.w_txt}")
    runs = run_queries(model, preprocess, tokenizer, queries, device,
                       index_pids, index_feats, args.topk,
                       args.w_img, args.w_txt, args.caption_field)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(runs, indent=2))
    print(f"wrote {len(runs)} runs -> {out}")


if __name__ == "__main__":
    main()
