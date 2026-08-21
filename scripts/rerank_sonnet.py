"""
Listwise vision reranker over a stage-1 run.

For each query, send to the model:
  - reference product image
  - modification text (English `translated_question`)
  - top-K candidate images, labeled A, B, C, ...
and ask for a best-to-worst ranking of those K letters.

The returned ordering is spliced into positions 0..K-1 of the original run;
positions K..end are kept as-is. Caching is per-(query, candidate-set, model)
so re-running with the same inputs costs nothing.

Usage:
  python scripts/rerank_sonnet.py \
      --run data/runs/searle_xl_official.json \
      --queries data/queries_official_cir.json \
      --out data/runs/searle_xl_official.rerank.json \
      --topk 20 --model sonnet
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import string
import sys
import time
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.models import resolve  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "eval_cache"

LETTERS = string.ascii_uppercase  # A-Z (up to 26 candidates)


def _label(i: int) -> str:
    """Two-character label that scales beyond 26 candidates. A..Z, AA..AZ, BA..BZ ..."""
    if i < 26:
        return LETTERS[i]
    return LETTERS[i // 26 - 1] + LETTERS[i % 26]


def _labels(n: int) -> list[str]:
    return [_label(i) for i in range(n)]


def load_image_bytes(path: Path, max_side: int = 384) -> bytes:
    """Open image (or first frame of mp4), downscale to <= max_side, return JPEG bytes."""
    if path.suffix.lower() == ".mp4":
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            img = Image.new("RGB", (max_side, max_side), color=(128, 128, 128))
        else:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (max_side, max_side), color=(128, 128, 128))
    w, h = img.size
    if max(w, h) > max_side:
        if w >= h:
            img = img.resize((max_side, int(h * max_side / w)))
        else:
            img = img.resize((int(w * max_side / h), max_side))
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def b64(b: bytes) -> str:
    return base64.standard_b64encode(b).decode()


def candidate_set_hash(cand_ids: list[str]) -> str:
    return hashlib.sha1(("|".join(cand_ids)).encode()).hexdigest()[:12]


def build_prompt(modification: str, ref_name: str, cand_names: list[str]) -> str:
    n = len(cand_names)
    labels = _labels(n)
    listing = "\n".join(
        f"  ({L}) {cand_names[i]}" for i, L in enumerate(labels)
    )
    return f"""You are reranking products for a Composed Image Retrieval task.

The user shows a REFERENCE product image and says:
  MODIFICATION: "{modification}"

You will then see {n} CANDIDATE product images, labeled ({labels[0]})..({labels[-1]}).

For each candidate, you also have its English product name (description noisy, prefer the image):
{listing}

Rank the {n} candidates from BEST match for "reference + modification" to WORST.
A candidate is a better match when its visual+textual identity matches the *reference* in
every way the modification does NOT change, AND satisfies the change the modification asks for.

Reply with ONLY the {n} labels separated by single spaces, best first, no other text.
Labels are one or two letters (A, B, ..., Z, AA, AB, ...).
"""


def parse_ranking(text: str, n: int) -> list[int] | None:
    """Parse model reply into a permutation of indices 0..n-1. Tolerates 1- or 2-letter labels."""
    label_to_idx = {_label(i): i for i in range(n)}
    # Greedy: prefer two-letter matches when valid.
    tokens = re.findall(r"[A-Z]{1,2}", text.upper())
    seen: list[int] = []
    for t in tokens:
        if t in label_to_idx and label_to_idx[t] not in seen:
            seen.append(label_to_idx[t])
        elif len(t) == 2 and t[0] in label_to_idx and label_to_idx[t[0]] not in seen:
            seen.append(label_to_idx[t[0]])
    if len(seen) < n:
        for i in range(n):
            if i not in seen:
                seen.append(i)
    return seen[:n]


class SonnetReranker:
    def __init__(self, model: str, topk: int):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = resolve(model)
        self.topk = topk
        safe = self.model.replace("/", "_").replace(":", "_")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_path = CACHE_DIR / f"rerank_{safe}_top{topk}.json"
        self.cache: dict[str, list[int]] = self._load()
        print(f"reranker model: {self.model}  topk={topk}  cache={self.cache_path.relative_to(ROOT)} ({len(self.cache)} entries)")

    def _load(self) -> dict:
        if self.cache_path.exists():
            return json.loads(self.cache_path.read_text())
        return {}

    def _save(self) -> None:
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cache, indent=2))
        tmp.replace(self.cache_path)

    def rerank_one(
        self, qid: str, modification: str, ref_path: Path, ref_name: str,
        cand_ids: list[str], cand_paths: list[Path], cand_names: list[str],
        retries: int = 3,
    ) -> list[int]:
        cache_key = f"{qid}||{candidate_set_hash(cand_ids)}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt = build_prompt(modification, ref_name, cand_names)
        # Build multimodal content: ref image, then candidate images in order.
        content = [
            {"type": "text", "text": "REFERENCE PRODUCT IMAGE:"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                          "data": b64(load_image_bytes(ref_path))}},
        ]
        for i, p in enumerate(cand_paths):
            content.append({"type": "text", "text": f"CANDIDATE ({_label(i)}):"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                         "data": b64(load_image_bytes(p))}})
        content.append({"type": "text", "text": prompt})

        for attempt in range(retries):
            try:
                msg = self.client.messages.create(
                    model=self.model, max_tokens=128,
                    messages=[{"role": "user", "content": content}],
                )
                reply = msg.content[0].text.strip()
                order = parse_ranking(reply, len(cand_ids))
                if order is not None:
                    self.cache[cache_key] = order
                    self._save()
                    return order
            except Exception as e:
                if attempt == retries - 1:
                    print(f"\n  [{qid}] rerank failed after {retries} retries: {e}", file=sys.stderr)
                    fallback = list(range(len(cand_ids)))
                    return fallback
                time.sleep(2 ** attempt)
        return list(range(len(cand_ids)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--products", default=str(ROOT / "data" / "raw" / "products__test.parquet"))
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--model", default="sonnet")
    args = parser.parse_args()

    queries = json.loads(Path(args.queries).read_text())
    run = json.loads(Path(args.run).read_text())
    products = (pd.read_parquet(args.products)
                .drop_duplicates("product_id")
                .set_index("product_id"))
    products.index = products.index.astype(str)

    reranker = SonnetReranker(model=args.model, topk=args.topk)

    out_run: dict[str, list[str]] = {}
    for qid in tqdm(list(run.keys()), desc="rerank"):
        ranked = run[qid]
        head = ranked[:args.topk]
        tail = ranked[args.topk:]
        if qid not in queries:
            out_run[qid] = ranked
            continue
        q = queries[qid]
        modification = q.get("translated_question") or q["question"]
        ref_path = RAW_DIR / q["image_path"]
        ref_meta = products.loc[str(q["product_id"])] if str(q["product_id"]) in products.index else None
        ref_name = (ref_meta["translated_name"] if ref_meta is not None else "")

        cand_paths, cand_names = [], []
        for pid in head:
            if str(pid) not in products.index:
                cand_paths.append(ref_path)  # safe fallback
                cand_names.append("(unknown product)")
                continue
            row = products.loc[str(pid)]
            cand_paths.append(RAW_DIR / row["image_path"])
            cand_names.append(row["translated_name"] or "")

        order = reranker.rerank_one(
            qid, modification, ref_path, ref_name, head, cand_paths, cand_names,
        )
        new_head = [head[i] for i in order]
        out_run[qid] = new_head + tail

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_run, indent=2))
    print(f"wrote reranked run → {out}")


if __name__ == "__main__":
    main()
