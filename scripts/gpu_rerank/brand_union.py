"""
Brand-aware candidate augmentation for brand-swap queries.

brand_swap queries name the target brand ("...from X instead of Y"). CLIP embeddings
match brand names poorly, so we parse the target brand and UNION the corpus products of
that brand (matched on the `brand` field) into the front of the Stage-1 candidate list.
Free, symbolic, CPU. No-op for queries that don't name a brand (e.g. natural official ones)
unless --extractor is extended.

Usage:
  python brand_union.py --run stage1.json --queries q.json --out augmented.json
"""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


def norm(s): return re.sub(r'[^a-z0-9 ]', '', str(s).lower()).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--products", default=str(ROOT / "data/raw/products__test.parquet"))
    args = ap.parse_args()

    prod = pd.read_parquet(args.products).drop_duplicates("product_id")
    prod["product_id"] = prod["product_id"].astype(str)
    brand2pids = defaultdict(set)
    for pid, b in zip(prod["product_id"], prod["brand"].fillna("")):
        brand2pids[norm(b)].add(pid)

    def brand_pool(txt):
        m = re.search(r'from (.+?) instead of', txt or "", re.I)
        if not m:
            return []
        t = norm(m.group(1))
        if t in brand2pids:
            return list(brand2pids[t])
        c = set()
        for nb, pids in brand2pids.items():
            if nb and (nb in t or t in nb or SequenceMatcher(None, nb, t).ratio() > 0.85):
                c |= pids
        return list(c)

    run = json.loads(Path(args.run).read_text())
    queries = json.loads(Path(args.queries).read_text())
    out = {}
    n_aug = 0
    for qid, ranked in run.items():
        q = queries.get(qid, {})
        bp = [p for p in brand_pool(q.get("translated_question", "")) if p not in ranked]
        if bp:
            out[qid] = bp + ranked
            n_aug += 1
        else:
            out[qid] = ranked
    Path(args.out).write_text(json.dumps(out))
    print(f"brand-union: augmented {n_aug}/{len(run)} queries -> {args.out}")


if __name__ == "__main__":
    main()
