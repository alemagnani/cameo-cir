"""
LLM query-attribute extractor (Approach #1c).

Extracts the structured target fields {brand, tgt_price, tgt_format, tgt_claim} from a query's
natural-language modification text, so metadata_retrieve.py can run on free-form / official queries
that have no gold structure. Cached per (model, qid) to data/eval_cache/.

Usage:
  python llm_extract.py --queries data/queries_dev_full.json --model haiku --out extracted.json
"""
from __future__ import annotations
import argparse, json, os, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = ROOT / "data" / "eval_cache"

PROMPT = """You extract structured search constraints from a shopping request about facial cleansers.
The user shows a reference product and asks for a DIFFERENT product. Extract the TARGET product's constraints.

Reply with ONLY a JSON object with these keys (use null if not specified):
- "brand": the brand the desired product should be (string). If they want a different brand, that new brand; if "same brand", that brand.
- "tgt_price": target price in VND as an integer (e.g. "around 49k" -> 49000, "350k" -> 350000), or null.
- "tgt_format": desired product format if mentioned (e.g. "gel cleanser", "foaming", "micellar / makeup-removing", "cream", "oil", "clay"), or null.
- "tgt_claim": desired benefit/skin-concern if mentioned (e.g. "brightening / whitening", "anti-aging", "for acne-prone skin", "for sensitive skin", "hydrating"), or null.

Request: "{q}"
JSON:"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--field", default="translated_question")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(ROOT))
    from src.eval.models import resolve
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = resolve(args.model)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"extract_{model.replace('/', '_')}.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    queries = json.loads(Path(args.queries).read_text())
    out = {}
    n_api = 0
    for qid, q in queries.items():
        if qid in cache:
            out[qid] = cache[qid]; continue
        text = q.get(args.field) or q.get("translated_question") or q.get("question") or ""
        try:
            msg = client.messages.create(model=model, max_tokens=256,
                                         messages=[{"role": "user", "content": PROMPT.format(q=text)}])
            raw = msg.content[0].text
            m = re.search(r"\{.*\}", raw, re.S)
            fields = json.loads(m.group(0)) if m else {}
        except Exception as e:
            print(f"  [{qid}] extract failed: {e}")
            fields = {}
        # normalize tgt_price to int
        tp = fields.get("tgt_price")
        if isinstance(tp, str):
            d = re.sub(r"[^0-9]", "", tp)
            fields["tgt_price"] = int(d) if d else None
        cache[qid] = fields; out[qid] = fields; n_api += 1
        if n_api % 50 == 0:
            cache_path.write_text(json.dumps(cache)); print(f"  ...{n_api} extracted")

    cache_path.write_text(json.dumps(cache))
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"llm_extract: {len(out)} queries ({n_api} new API calls) -> {args.out}")


if __name__ == "__main__":
    main()
