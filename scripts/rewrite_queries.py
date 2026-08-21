"""
Add a `translated_question_rewritten` field to every query in dev / official
files. Sonnet rewrites the original EN modification text into a richer
description with brand/format/category context — helpful for CLIP-grounding
on niche e-commerce brands.

Idempotent: re-running only writes for queries that don't already have the field.
Costs ~$0.5–1 over the full dev+official set.

Usage:
  python scripts/rewrite_queries.py --files data/queries_dev_full.json data/queries_official_cir.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.models import resolve  # noqa: E402


PROMPT = """You are augmenting a shopper's modification text for a Composed Image Retrieval system over face cleansers.

The shopper wants to retrieve a face cleanser product *similar to their reference image* with the modification described below.

REFERENCE PRODUCT NAME (English): {ref_name}
ORIGINAL MODIFICATION: "{caption}"

Rewrite the modification into 1–2 natural English sentences that:
  - keep the original intent literally
  - add concrete, retrieval-relevant detail (product category if obvious, brand
    descriptors if a brand is named, ingredient/format/skin-type hints if implied)
  - stay grounded in what the shopper actually said — don't invent products

Reply with ONLY the rewritten text — no quotes, no preamble, no explanation."""


def rewrite_one(client, model: str, ref_name: str, caption: str, retries: int = 3) -> str | None:
    msgs = [
        {
            "role": "user",
            "content": PROMPT.replace("{ref_name}", ref_name or "(unknown)").replace("{caption}", caption),
        }
    ]
    for attempt in range(retries):
        try:
            r = client.messages.create(model=model, max_tokens=256, messages=msgs)
            text = r.content[0].text.strip().strip('"').strip()
            if text:
                return text
        except Exception as e:
            if attempt == retries - 1:
                return None
            time.sleep(2 ** attempt)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--field", default="translated_question_rewritten")
    args = parser.parse_args()

    import anthropic
    import pandas as pd

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = resolve(args.model)
    products = pd.read_parquet(ROOT / "data" / "raw" / "products__test.parquet").drop_duplicates("product_id")
    products = products.set_index("product_id")
    products.index = products.index.astype(str)

    for path in args.files:
        p = Path(path)
        queries = json.loads(p.read_text())
        todo = [qid for qid, q in queries.items() if not q.get(args.field)]
        print(f"\n{p.name}: {len(queries)} queries, {len(todo)} need rewriting")
        if not todo:
            continue

        def task(qid):
            q = queries[qid]
            caption = q.get("translated_question") or q.get("question") or ""
            ref_pid = str(q.get("product_id", ""))
            ref_name = products.loc[ref_pid]["translated_name"] if ref_pid in products.index else ""
            return qid, rewrite_one(client, model, ref_name, caption)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(task, qid) for qid in todo]
            done = 0
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"rewrite {p.name}"):
                qid, rewritten = fut.result()
                if rewritten:
                    queries[qid][args.field] = rewritten
                done += 1
                if done % 100 == 0:
                    tmp = p.with_suffix(".tmp")
                    tmp.write_text(json.dumps(queries, indent=2, ensure_ascii=False))
                    tmp.replace(p)

        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(queries, indent=2, ensure_ascii=False))
        tmp.replace(p)
        kept = sum(1 for q in queries.values() if q.get(args.field))
        print(f"  done; {kept}/{len(queries)} have {args.field}")


if __name__ == "__main__":
    main()
