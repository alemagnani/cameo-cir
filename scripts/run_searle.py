"""
Run zero-shot iSEARLE inference on the CAMEO corpus.

Pipeline:
  - CLIP ViT-L/14 image encoder over all 4,075 products → index (cached).
  - For each query: ref image → CLIP image feats → SEARLE Phi → pseudo-token S*.
    Build prompt "a photo of $ that {translated_question}" with $ replaced by S*.
    CLIP text encoder → text feature. Rank corpus by cosine.
  - Emit run JSON {qid: [top-K product_ids]} for evaluation.

Defaults: ViT-L/14 ("SEARLE-XL"), English `translated_question`, top-50.

Usage:
  python scripts/run_searle.py --queries data/queries_dev.json --out data/runs/searle_vitl14_dev.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import clip
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
PRODUCTS_PARQUET = ROOT / "data" / "raw" / "products__test.parquet"
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "processed"


_PLACEHOLDER = Image.new("RGB", (224, 224), color=(128, 128, 128))


def load_product_image(path: Path) -> Image.Image:
    """Open product asset as RGB PIL Image.

    .mp4 assets → first frame. Corrupt / empty / missing files → gray placeholder
    (so a single bad row doesn't drop a product from the index; rare in this corpus).
    """
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
def encode_corpus(
    clip_model, preprocess, products: pd.DataFrame, device: str, batch_size: int,
) -> tuple[list[str], torch.Tensor]:
    feats = []
    pids = products["product_id"].astype(str).tolist()
    paths = products["image_path"].tolist()
    for i in tqdm(range(0, len(paths), batch_size), desc="encode corpus (image)"):
        batch_paths = paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            img = load_product_image(RAW_DIR / p)
            imgs.append(preprocess(img))
        batch = torch.stack(imgs).to(device)
        f = clip_model.encode_image(batch).float()
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu())
    return pids, torch.cat(feats, dim=0)


def _build_text_for_fusion(products: pd.DataFrame, source: str) -> list[str]:
    """Return one text per product based on the chosen fusion source.

    Note: the dataset's `translated_brand` column is unreliable (mislabels);
    `brand` is the authoritative source.
    """
    if source == "name":
        return products["translated_name"].fillna("").astype(str).tolist()
    if source == "brand":
        brand = products["brand"].fillna("").astype(str).str.strip()
        # Treat "No_Brand" / blank as no brand info
        return [
            f"a {b} facial cleanser" if b and b.lower() not in ("no_brand", "none", "nan") else "a facial cleanser"
            for b in brand
        ]
    if source == "brand_name":
        # Brand prefix + first 8 words of name — concise brand-aware text
        brand = products["brand"].fillna("").astype(str).str.strip()
        name = products["translated_name"].fillna("").astype(str)
        out = []
        for b, n in zip(brand, name):
            short = " ".join(n.split()[:8])
            if b and b.lower() not in ("no_brand", "none", "nan"):
                out.append(f"{b} {short}")
            else:
                out.append(short)
        return out
    raise ValueError(f"unknown fuse-text-source: {source!r}")


@torch.no_grad()
def encode_corpus_text(
    clip_model, products: pd.DataFrame, device: str, batch_size: int = 64,
    source: str = "name",
) -> torch.Tensor:
    """CLIP text embeddings of one text per product (per `source`). L2-normalized."""
    feats = []
    texts = _build_text_for_fusion(products, source)
    for i in tqdm(range(0, len(texts), batch_size), desc=f"encode corpus (text:{source})"):
        batch = texts[i:i + batch_size]
        tok = clip.tokenize(batch, truncate=True).to(device)
        f = clip_model.encode_text(tok).float()
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def run_queries(
    clip_model, preprocess, phi, encode_with_pseudo_tokens,
    queries: dict, prompt_template: str, device: str,
    index_pids: list[str], index_feats: torch.Tensor, topk: int,
    oti_tokens: dict | None = None,
    caption_field: str = "translated_question",
) -> dict:
    pid2idx = {pid: i for i, pid in enumerate(index_pids)}
    runs: dict[str, list[str]] = {}
    for qid, q in tqdm(list(queries.items()), desc="queries"):
        img = load_product_image(RAW_DIR / q["image_path"])
        img_t = preprocess(img).unsqueeze(0).to(device)
        ref_pid = str(q["product_id"])
        if oti_tokens is not None and ref_pid in oti_tokens:
            pseudo = oti_tokens[ref_pid].to(device).unsqueeze(0).float()           # [1, D]
        else:
            img_feat = clip_model.encode_image(img_t).float()                      # [1, D]
            pseudo = phi(img_feat)                                                 # [1, D]
        caption = q.get(caption_field) or q.get("translated_question") or q["question"]
        prompt = prompt_template.replace("{caption}", caption)
        tok = clip.tokenize([prompt], truncate=True).to(device)
        text_feat = encode_with_pseudo_tokens(clip_model, tok, pseudo).float()    # [1, D]
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        sims = (text_feat @ index_feats.to(device).T).squeeze(0).cpu()             # [N]
        # Exclude the reference itself from results.
        if ref_pid in pid2idx:
            sims[pid2idx[ref_pid]] = -float("inf")
        top = torch.topk(sims, k=min(topk, sims.shape[0])).indices.tolist()
        runs[qid] = [index_pids[i] for i in top]
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--backbone", default="ViT-L/14", choices=["ViT-L/14", "ViT-B/32"])
    parser.add_argument("--prompt", default="a photo of $ that {caption}",
                        help="Prompt template; $ is replaced by the SEARLE pseudo-token; {caption} by the query text.")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--caption-field", default="translated_question",
                        help="Which query field to use as {caption}. "
                             "Falls back to translated_question then question if missing.")
    parser.add_argument("--fuse-text", type=float, default=0.0,
                        help="α ∈ [0,1]: index = α·CLIP-text(fusion_source) + (1-α)·CLIP-image, re-normalized. "
                             "0 reproduces EXP-001.")
    parser.add_argument("--fuse-text-source", default="name", choices=["name", "brand", "brand_name"],
                        help="What text to fuse into the index. name=translated_name, "
                             "brand='a {brand} facial cleanser', brand_name=brand + first 8 words.")
    parser.add_argument("--oti-tokens", default=None,
                        help="Path to a directory containing image_names.pkl + ema_oti_pseudo_tokens.pt produced by SEARLE OTI. "
                             "When set, use the OTI token for the query reference product instead of Phi(image_feat). "
                             "All query reference product_ids must be in image_names.pkl.")
    args = parser.parse_args()

    device = args.device
    print(f"device={device}  backbone={args.backbone}")

    print("loading SEARLE Phi + CLIP...")
    phi, encode_with_pseudo_tokens = torch.hub.load(
        "miccunifi/SEARLE", source="github", model="searle",
        backbone=args.backbone, trust_repo=True,
    )
    phi = phi.to(device).eval()
    clip_model, preprocess = clip.load(args.backbone, device=device)
    clip_model = clip_model.eval()

    # Image-only index cache (shared across α values).
    bb_tag = args.backbone.replace("/", "").lower()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    img_cache = CACHE_DIR / f"clip_{bb_tag}_index.pt"
    txt_cache = CACHE_DIR / f"clip_{bb_tag}_index_text.pt"

    if img_cache.exists() and not args.rebuild_index:
        print(f"loading cached image index → {img_cache.relative_to(ROOT)}")
        blob = torch.load(img_cache, map_location="cpu")
        index_pids, image_feats = blob["pids"], blob["feats"]
        print(f"  loaded {len(index_pids)} entries, dim={image_feats.shape[-1]}")
    else:
        products = pd.read_parquet(PRODUCTS_PARQUET).drop_duplicates("product_id").reset_index(drop=True)
        print(f"encoding {len(products)} products (image)...")
        index_pids, image_feats = encode_corpus(clip_model, preprocess, products, device, args.batch_size)
        torch.save({"pids": index_pids, "feats": image_feats}, img_cache)
        print(f"  cached → {img_cache.relative_to(ROOT)}")

    if args.fuse_text > 0.0:
        # Source-specific text cache so name/brand/brand_name fusion don't clobber each other
        src_tag = args.fuse_text_source
        txt_cache_src = CACHE_DIR / f"clip_{bb_tag}_index_text_{src_tag}.pt"
        # Migrate legacy "name" cache without a suffix
        if src_tag == "name" and txt_cache.exists() and not txt_cache_src.exists():
            import shutil
            shutil.copy(txt_cache, txt_cache_src)
        if txt_cache_src.exists() and not args.rebuild_index:
            print(f"loading cached text index → {txt_cache_src.relative_to(ROOT)}")
            blob = torch.load(txt_cache_src, map_location="cpu")
            text_pids, text_feats = blob["pids"], blob["feats"]
            assert text_pids == index_pids, "pid order mismatch between image and text caches"
        else:
            products = pd.read_parquet(PRODUCTS_PARQUET).drop_duplicates("product_id").reset_index(drop=True)
            # ensure same order
            assert products["product_id"].astype(str).tolist() == index_pids
            print(f"encoding {len(products)} products (text:{src_tag})...")
            text_feats = encode_corpus_text(clip_model, products, device, source=src_tag)
            torch.save({"pids": index_pids, "feats": text_feats}, txt_cache_src)
            print(f"  cached → {txt_cache_src.relative_to(ROOT)}")

        a = args.fuse_text
        print(f"fusing index with α={a}: α·text + (1-α)·image, then renormalize")
        fused = a * text_feats + (1.0 - a) * image_feats
        fused = fused / fused.norm(dim=-1, keepdim=True)
        index_feats = fused
    else:
        index_feats = image_feats

    queries = json.loads(Path(args.queries).read_text())
    print(f"running {len(queries)} queries from {args.queries}")

    oti_lookup = None
    if args.oti_tokens:
        import pickle
        oti_dir = Path(args.oti_tokens)
        names_path = oti_dir / "image_names.pkl"
        tokens_path = oti_dir / "ema_oti_pseudo_tokens.pt"
        with open(names_path, "rb") as f:
            oti_names = pickle.load(f)
        oti_tens = torch.load(tokens_path, map_location="cpu")
        oti_lookup = {str(n): oti_tens[i] for i, n in enumerate(oti_names)}
        missing = [str(q["product_id"]) for q in queries.values() if str(q["product_id"]) not in oti_lookup]
        print(f"OTI tokens loaded: {len(oti_lookup)}; missing refs in queries: {len(missing)}")

    runs = run_queries(
        clip_model, preprocess, phi, encode_with_pseudo_tokens,
        queries, args.prompt, device, index_pids, index_feats, args.topk,
        oti_tokens=oti_lookup, caption_field=args.caption_field,
    )

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(runs, indent=2))
    try:
        print(f"wrote {len(runs)} runs → {out.relative_to(ROOT)}")
    except ValueError:
        print(f"wrote {len(runs)} runs → {out}")


if __name__ == "__main__":
    main()
