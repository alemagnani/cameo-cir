"""
Phase A — build a self-contained rerank-job manifest + a small resized image store.

Everything the GPU box needs to rerank, prepared locally so the box only does
VLM forward passes:
  - manifest.jsonl : one job per query {qid, modification, ref_pid, cand_pids, cand_names}
  - images/<pid>.jpg : every referenced product image, downscaled to <=max_side

Candidate sets come from a fixed Stage-1 run (the same one Haiku reranked), so the
VLM comparison is apples-to-apples. Multiple runs can be merged into one manifest
(dedup by (qid, candidate-set)) so a single GPU session covers the whole backlog.

Usage:
  python scripts/gpu_rerank/build_manifest.py \
      --run data/runs/searle_phi_a05_official.json --queries data/queries_official_cir.json \
      --run data/runs/searle_phi_a05_dev.json:429   --queries data/queries_dev.json \
      --topk 20 --out-dir data/gpu_rerank/validation
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "data" / "raw"
_PLACEHOLDER = Image.new("RGB", (384, 384), color=(128, 128, 128))


def cand_hash(cand_ids: list[str]) -> str:
    return hashlib.sha1("|".join(cand_ids).encode()).hexdigest()[:12]


def load_resized(path: Path, max_side: int) -> Image.Image:
    try:
        if path.suffix.lower() == ".mp4":
            cap = cv2.VideoCapture(str(path))
            ok, frame = cap.read()
            cap.release()
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if ok else _PLACEHOLDER
        else:
            img = Image.open(path).convert("RGB")
    except Exception:
        img = _PLACEHOLDER
    w, h = img.size
    if max(w, h) > max_side:
        if w >= h:
            img = img.resize((max_side, max(1, int(h * max_side / w))))
        else:
            img = img.resize((max(1, int(w * max_side / h)), max_side))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", required=True,
                    help="run_json::queries_json  (repeatable). Candidate source + query metadata.")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--max-side", type=int, default=384)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    products = (pd.read_parquet(RAW_DIR / "products__test.parquet")
                .drop_duplicates("product_id").set_index("product_id"))
    products.index = products.index.astype(str)

    jobs: list[dict] = []
    seen_keys: set[str] = set()
    needed_pids: set[str] = set()

    for pair in args.pair:
        run_path, q_path = pair.split("::")
        run = json.loads(Path(run_path).read_text())
        queries = json.loads(Path(q_path).read_text())
        for qid in run:
            if qid not in queries:
                continue
            ranked = run[qid]
            head = ranked[:args.topk]
            key = f"{qid}||{cand_hash([str(p) for p in head])}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            q = queries[qid]
            ref_pid = str(q["product_id"])
            modification = q.get("translated_question") or q["question"]
            cand_pids, cand_names = [], []
            for pid in head:
                pid = str(pid)
                cand_pids.append(pid)
                nm = products.loc[pid]["translated_name"] if pid in products.index else "(unknown product)"
                cand_names.append(nm or "")
            ref_name = products.loc[ref_pid]["translated_name"] if ref_pid in products.index else ""
            jobs.append({"qid": qid, "cand_hash": key.split("||")[1],
                         "modification": modification, "ref_pid": ref_pid, "ref_name": ref_name,
                         "cand_pids": cand_pids, "cand_names": cand_names})
            needed_pids.add(ref_pid)
            needed_pids.update(cand_pids)

    # Resize + write every referenced image once.
    written = 0
    for pid in tqdm(sorted(needed_pids), desc="resize images"):
        dst = img_dir / f"{pid}.jpg"
        if dst.exists():
            continue
        if pid not in products.index:
            _PLACEHOLDER.save(dst, format="JPEG", quality=85)
            continue
        img = load_resized(RAW_DIR / products.loc[pid]["image_path"], args.max_side)
        img.save(dst, format="JPEG", quality=85)
        written += 1

    manifest_path = out_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")

    size_mb = sum(p.stat().st_size for p in img_dir.glob("*.jpg")) / 1e6
    print(f"jobs: {len(jobs)}  unique images: {len(needed_pids)} ({size_mb:.1f} MB)")
    print(f"manifest -> {manifest_path}")
    print(f"images   -> {img_dir}")


if __name__ == "__main__":
    main()
