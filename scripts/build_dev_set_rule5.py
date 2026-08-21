"""
Generate Rule-5 dev queries: LLM free-form composed-image-retrieval modifications.

For each (ref, target) pair sampled from CLIP-L nearest neighbors at ranks 5–25
(not rank 1; avoids trivial duplicates), Sonnet 4.6 sees both product images
and is asked to:

  1. write a natural English modification text that, given the ref image,
     would lead a shopper to the target product
  2. translate that to natural conversational Vietnamese
  3. score confidence 1–5 that this is a sensible CIR pair at all

We keep only pairs with confidence ≥ 4. Output is appended to a JSON file
shaped like `data/queries_dev.json` (so it merges in directly), with rule="llm_free_form".

Usage:
  python scripts/build_dev_set_rule5.py --target 400 --workers 8
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.models import resolve  # noqa: E402

PRODUCTS_PARQUET = ROOT / "data" / "raw" / "products__test.parquet"
RAW_DIR = ROOT / "data" / "raw"
INDEX_CACHE = ROOT / "data" / "processed" / "clip_vit-l14_index.pt"
OUT_PATH_DEFAULT = ROOT / "data" / "queries_dev_rule5.json"
PROGRESS_PATH_DEFAULT = ROOT / "data" / "queries_dev_rule5.progress.json"

_PLACEHOLDER = Image.new("RGB", (224, 224), color=(128, 128, 128))


def load_image(path: Path, max_side: int = 384) -> Image.Image:
    try:
        if path.suffix.lower() == ".mp4":
            cap = cv2.VideoCapture(str(path))
            ok, frame = cap.read()
            cap.release()
            if not ok:
                return _PLACEHOLDER
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            img = Image.open(path).convert("RGB")
    except Exception:
        return _PLACEHOLDER
    w, h = img.size
    if max(w, h) > max_side:
        if w >= h:
            img = img.resize((max_side, int(h * max_side / w)))
        else:
            img = img.resize((int(w * max_side / h), max_side))
    return img


def b64_jpeg(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode()


PROMPT = """You are constructing a Composed Image Retrieval (CIR) dataset for face cleansers.

You see two product images:
  - REFERENCE: what the shopper is looking at now
  - TARGET: a different product they might want instead

If a real shopper could plausibly say "show me something like REFERENCE but ___",
write the natural English text that fills that blank — the *modification* — such
that retrieval given (REFERENCE image + your text) should rank TARGET high.

The modification must describe a meaningful change (different brand, different
format, different size/price, different skincare claim, etc.). It must NOT
just describe the reference or the target verbatim.

Then translate the English modification to natural conversational Vietnamese
as a Vietnamese e-commerce shopper would write it. Keep brand names in Latin
spelling.

Reply with EXACTLY this JSON shape, no other text:
{
  "en": "english modification text, 1 sentence",
  "vn": "vietnamese translation, 1 sentence",
  "confidence": <integer 1-5; 5 = great CIR pair, 1 = the two products are too similar or too unrelated for CIR>
}

Reference product name (English): "{ref_name}"
Target product name (English):    "{tgt_name}"
"""


def build_messages(ref_img: Image.Image, tgt_img: Image.Image,
                   ref_name: str, tgt_name: str) -> list[dict]:
    content = [
        {"type": "text", "text": "REFERENCE IMAGE:"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                      "data": b64_jpeg(ref_img)}},
        {"type": "text", "text": "TARGET IMAGE:"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                      "data": b64_jpeg(tgt_img)}},
        {"type": "text", "text": PROMPT.replace("{ref_name}", ref_name or "").replace("{tgt_name}", tgt_name or "")},
    ]
    return [{"role": "user", "content": content}]


def parse_reply(text: str) -> dict | None:
    """Pull the JSON object out of the model reply, tolerating prose around it."""
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if not all(k in obj for k in ("en", "vn", "confidence")):
            return None
        obj["confidence"] = int(obj["confidence"])
        return obj
    except Exception:
        return None


def sample_pairs(products: pd.DataFrame, index_pids: list[str], index_feats: torch.Tensor,
                 n_pairs: int, rank_lo: int = 5, rank_hi: int = 25,
                 max_per_ref: int = 1, seed: int = 42) -> list[tuple[str, str]]:
    """Sample (ref_pid, tgt_pid) pairs from CLIP-L neighbor ranks rank_lo..rank_hi."""
    rng = random.Random(seed)
    pid2idx = {pid: i for i, pid in enumerate(index_pids)}
    pids = list(index_pids)
    rng.shuffle(pids)

    # cosine sims on cached normalized features — full corpus matmul (4075×4075×768) is fine.
    print("computing neighbor ranks...")
    sims = (index_feats @ index_feats.T)  # already normalized
    # zero out self-similarity
    diag = torch.arange(sims.shape[0])
    sims[diag, diag] = -1.0
    topk_idx = sims.topk(k=rank_hi + 1, dim=1).indices  # [N, rank_hi+1]
    print(f"  topk shape {tuple(topk_idx.shape)}")

    out: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    seen_refs: dict[str, int] = {}
    for ref_pid in pids:
        if len(out) >= n_pairs:
            break
        if seen_refs.get(ref_pid, 0) >= max_per_ref:
            continue
        ridx = pid2idx[ref_pid]
        neighbors = topk_idx[ridx].tolist()
        # neighbors[0] is the closest non-self; take rank_lo..rank_hi range
        candidates = neighbors[rank_lo:rank_hi]
        if not candidates:
            continue
        tgt_idx = rng.choice(candidates)
        tgt_pid = index_pids[tgt_idx]
        if (ref_pid, tgt_pid) in seen_pairs:
            continue
        seen_pairs.add((ref_pid, tgt_pid))
        seen_refs[ref_pid] = seen_refs.get(ref_pid, 0) + 1
        out.append((ref_pid, tgt_pid))
    return out


def load_progress(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"by_pair": {}, "kept": []}


def save_progress(progress: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(progress, ensure_ascii=False))
    tmp.replace(path)


def call_one(client, model: str, ref_meta: dict, tgt_meta: dict, retries: int = 3) -> dict | None:
    ref_img = load_image(RAW_DIR / ref_meta["image_path"])
    tgt_img = load_image(RAW_DIR / tgt_meta["image_path"])
    msgs = build_messages(ref_img, tgt_img,
                          ref_meta.get("translated_name", "") or "",
                          tgt_meta.get("translated_name", "") or "")
    for attempt in range(retries):
        try:
            r = client.messages.create(model=model, max_tokens=512, messages=msgs)
            text = r.content[0].text
            parsed = parse_reply(text)
            return parsed
        except Exception as e:
            if attempt == retries - 1:
                return {"error": f"{type(e).__name__}: {e}"}
            time.sleep(2 ** attempt)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=400, help="Target number of kept (conf≥4) queries")
    parser.add_argument("--oversample", type=float, default=2.0, help="Sample this many ×target pairs to absorb rejections")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--min-confidence", type=int, default=4)
    parser.add_argument("--rank-lo", type=int, default=5)
    parser.add_argument("--rank-hi", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed; bump to draw a different pair set on a re-run.")
    parser.add_argument("--max-per-ref", type=int, default=1)
    parser.add_argument("--out", default=str(OUT_PATH_DEFAULT), help="Output queries JSON")
    parser.add_argument("--progress", default=str(PROGRESS_PATH_DEFAULT), help="Progress checkpoint JSON")
    parser.add_argument("--qid-prefix", default="DEV-R5", help="QID prefix for output entries (use distinct prefixes when generating multiple passes)")
    args = parser.parse_args()

    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = resolve(args.model)
    print(f"using model {model}")

    # Load corpus + CLIP index
    products = pd.read_parquet(PRODUCTS_PARQUET).drop_duplicates("product_id").reset_index(drop=True)
    blob = torch.load(INDEX_CACHE, map_location="cpu")
    index_pids, index_feats = blob["pids"], blob["feats"]
    products = products.set_index("product_id")
    products.index = products.index.astype(str)

    n_to_sample = int(args.target * args.oversample)
    pairs = sample_pairs(products.reset_index(), index_pids, index_feats,
                         n_pairs=n_to_sample, rank_lo=args.rank_lo, rank_hi=args.rank_hi,
                         max_per_ref=args.max_per_ref, seed=args.seed)
    print(f"sampled {len(pairs)} candidate pairs")

    progress_path = Path(args.progress)
    out_path = Path(args.out)
    progress = load_progress(progress_path)
    by_pair: dict[str, dict] = progress["by_pair"]
    kept: list[dict] = progress["kept"]
    print(f"progress: {len(by_pair)} previous calls, {len(kept)} kept so far")

    def task(pair):
        ref_pid, tgt_pid = pair
        key = f"{ref_pid}||{tgt_pid}"
        if key in by_pair:
            return key, by_pair[key]
        ref_meta = products.loc[ref_pid].to_dict()
        tgt_meta = products.loc[tgt_pid].to_dict()
        result = call_one(client, model, ref_meta, tgt_meta)
        return key, result

    new_calls = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(task, p) for p in pairs]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="LLM pair generation"):
            key, result = fut.result()
            by_pair[key] = result
            if isinstance(result, dict) and result.get("confidence", 0) >= args.min_confidence and "en" in result:
                ref_pid, tgt_pid = key.split("||")
                ref_meta = products.loc[ref_pid].to_dict()
                kept.append({
                    "ref_pid": ref_pid, "tgt_pid": tgt_pid,
                    "ref_image_path": ref_meta["image_path"],
                    "en": result["en"], "vn": result["vn"],
                    "confidence": result["confidence"],
                })
            new_calls += 1
            if new_calls % 25 == 0:
                save_progress({"by_pair": by_pair, "kept": kept}, progress_path)
                print(f"  checkpoint: {new_calls} new calls, kept={len(kept)}")
            if len(kept) >= args.target:
                # Cancel remaining futures
                for f in futs:
                    f.cancel()
                break

    save_progress({"by_pair": by_pair, "kept": kept}, progress_path)

    # Emit queries in the same shape as data/queries_dev.json
    queries: dict[str, dict] = {}
    for i, k in enumerate(kept[: args.target]):
        qid = f"{args.qid_prefix}-{i:04d}"
        queries[qid] = {
            "image_path": k["ref_image_path"],
            "product_id": str(k["ref_pid"]),
            "question": k["vn"],
            "translated_question": k["en"],
            "answer_product_id": [str(k["tgt_pid"])],
            "rule": "llm_free_form",
            "confidence": k["confidence"],
        }
    out_path.write_text(json.dumps(queries, indent=2, ensure_ascii=False))
    try:
        print(f"wrote {len(queries)} queries → {out_path.relative_to(ROOT)}")
    except ValueError:
        print(f"wrote {len(queries)} queries → {out_path}")
    print(f"acceptance rate: {len(kept) / max(new_calls,1):.1%}  (target {args.target}, kept {len(kept)})")


if __name__ == "__main__":
    main()
