"""
Structured metadata retrieval (Approach #1).

For each query we have (gold or LLM-extracted) structured fields describing the *target* product:
the relevant brand, a target price, and optionally a target format/claim. We retrieve corpus
products by symbolic metadata matching — brand match + price proximity + format/claim keyword hits
in name+description — exploiting the corpus columns (brand, numeric price VND, translated_description)
that visual CLIP ignores. Output is a ranked run.json {qid:[product_ids]} (high-precision head),
which can be fused with a visual run (rrf_fuse.py) and reranked.

Fields per query (schema, same for gold and extracted):
  rule, brand (the brand to match: tgt_brand for brand_swap else ref_brand),
  tgt_price (int VND, optional), tgt_format (str, optional), tgt_claim (str, optional).

Usage:
  # gold fields straight from the dev queries:
  python metadata_retrieve.py --queries data/queries_dev.json --use-gold --out run.json
  # extracted fields from llm_extract.py:
  python metadata_retrieve.py --queries data/queries_dev_full.json --fields extracted.json --out run.json
"""
from __future__ import annotations
import argparse, json, math, re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

FORMAT_KW = {
    "gel": ["gel"], "foam": ["foam"], "foaming": ["foam"], "cream": ["cream"],
    "oil": ["oil"], "micellar": ["micellar", "makeup", "make-up", "cleansing water"],
    "makeup": ["micellar", "makeup", "make-up"], "clay": ["clay", "mud"], "powder": ["powder"],
    "balm": ["balm"], "milk": ["milk"], "bar": ["bar", "soap"], "soap": ["bar", "soap"],
    "water": ["water", "micellar"], "mousse": ["mousse", "foam"], "scrub": ["scrub", "exfoliat"],
}
CLAIM_KW = {
    "brighten": ["brighten", "whiten", "glow", "radiance"], "whiten": ["whiten", "brighten"],
    "anti-aging": ["anti aging", "anti-aging", "wrinkle", "aging", "firm"],
    "aging": ["anti aging", "anti-aging", "wrinkle", "aging"], "acne": ["acne", "blemish", "pimple"],
    "hydrat": ["hydrat", "moistur"], "dry": ["hydrat", "moistur", "dry"],
    "sensitive": ["sensitive", "gentle"], "oil control": ["oil control", "oily", "sebum"],
    "oily": ["oil control", "oily", "sebum"], "pore": ["pore"], "exfoliat": ["exfoliat", "scrub"],
}


def norm_brand(s):
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()


def phrase_keywords(phrase, kwmap):
    p = str(phrase or "").lower()
    out = []
    for key, kws in kwmap.items():
        if key in p:
            out += kws
    return list(dict.fromkeys(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--products", default=str(ROOT / "data/raw/products__test.parquet"))
    ap.add_argument("--use-gold", action="store_true", help="derive fields from the dev gold query fields")
    ap.add_argument("--fields", default=None, help="json {qid: {brand,tgt_price,tgt_format,tgt_claim,...}}")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--price-sigma", type=float, default=0.4, help="relative price proximity width")
    ap.add_argument("--terms", default="brand,price,format,claim",
                    help="comma list of scoring terms to enable (ablation); brand is always the gate")
    args = ap.parse_args()
    terms = {t.strip() for t in args.terms.split(",") if t.strip()}

    prod = pd.read_parquet(args.products).drop_duplicates("product_id")
    prod["product_id"] = prod["product_id"].astype(str)
    pid_list = prod["product_id"].tolist()
    brand_n = {p: norm_brand(b) for p, b in zip(prod["product_id"], prod["brand"].fillna(""))}
    price = {p: (int(v) if pd.notna(v) else None) for p, v in zip(prod["product_id"], prod["price"])}
    text = {p: (str(n) + " " + str(d)).lower() for p, n, d in
            zip(prod["product_id"], prod["translated_name"].fillna(""), prod["translated_description"].fillna(""))}
    b2p = defaultdict(set)
    for p, b in brand_n.items():
        b2p[b].add(p)

    def brand_match(brand):
        t = norm_brand(brand)
        if not t:
            return None  # no brand constraint -> signal "no confident metadata candidates"
        if t in b2p:
            return set(b2p[t])
        c = set()
        for nb, ps in b2p.items():
            if nb and (nb in t or t in nb or SequenceMatcher(None, nb, t).ratio() > 0.85):
                c |= ps
        return c

    queries = json.loads(Path(args.queries).read_text())
    fields = json.loads(Path(args.fields).read_text()) if args.fields else {}

    def get_fields(qid, q):
        if args.use_gold:
            rule = q.get("rule", "")
            brand = q.get("tgt_brand") if rule == "brand_swap" else q.get("ref_brand")
            return {"brand": brand, "tgt_price": q.get("tgt_price"),
                    "tgt_format": q.get("tgt_format"), "tgt_claim": q.get("tgt_claim")}
        return fields.get(qid, {})

    runs = {}
    for qid, q in queries.items():
        f = get_fields(qid, q)
        cands = brand_match(f.get("brand"))
        if cands is None or len(cands) > 200:  # no/too-weak brand constraint -> defer to visual
            runs[qid] = []
            continue
        ref = str(q.get("product_id"))
        cands.discard(ref)
        fmt_kw = phrase_keywords(f.get("tgt_format"), FORMAT_KW)
        clm_kw = phrase_keywords(f.get("tgt_claim"), CLAIM_KW)
        tp = f.get("tgt_price")
        scored = []
        for p in cands:
            s = 1.0  # brand-matched base
            if "format" in terms and fmt_kw:
                s += 2.0 if any(k in text[p] for k in fmt_kw) else 0.0
            if "claim" in terms and clm_kw:
                s += 2.0 if any(k in text[p] for k in clm_kw) else 0.0
            if "price" in terms and tp and price[p]:
                rel = abs(price[p] - tp) / max(tp, 1)
                s += math.exp(-(rel ** 2) / (2 * args.price_sigma ** 2))  # 0..1, peak at exact price
            scored.append((p, s))
        scored.sort(key=lambda x: (-x[1], x[0]))  # deterministic tie-break by product_id
        runs[qid] = [p for p, _ in scored[:args.topk]]

    Path(args.out).write_text(json.dumps(runs))
    n_nonempty = sum(1 for v in runs.values() if v)
    print(f"metadata_retrieve: {len(runs)} queries, {n_nonempty} non-empty -> {args.out}")


if __name__ == "__main__":
    main()
