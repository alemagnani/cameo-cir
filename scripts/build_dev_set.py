"""
Build data/queries_dev.json — a synthetic development set with hard labels.

Rationale (see README §7.1): the official test set ships without
answer_product_id, so we cannot iterate against organizer relevance
judgments. This dev set lets us compute NDCG@k / MAP with hard labels
during development.

Rules
-----
- cheaper / pricier  : within-brand price-swap (original v0)
- format_swap        : within-brand swap of product format (foam/gel/cream/...)
- brand_swap         : cross-brand pair at similar price, same format+claim
- claim_swap         : within-brand swap of skin-concern claim
- free_form          : (deferred) LLM-judged pairs sampled from CLIP neighbors

Each rule emits a uniqueness predicate so NDCG@1 is meaningful.

Schema matches data/queries_official_cir.json but adds answer_product_id,
rule, and per-rule diagnostic fields.

Output: data/queries_dev.json (merged) + data/queries_dev_by_rule/{rule}.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PRODUCTS_PARQUET = ROOT / "data" / "raw" / "products__test.parquet"
OUT_PATH = ROOT / "data" / "queries_dev.json"
OUT_DIR = ROOT / "data" / "queries_dev_by_rule"
SEED = 42

# Keyword tables for format / claim mining. Patterns are regex, matched
# case-insensitively against the lowercased name+description in both
# English and Vietnamese.
FORMATS = {
    "foam":     [r"\bfoam", r"\bfoaming", r"\bbọt\b", r"sủi bọt"],
    "gel":      [r"\bgel\b"],
    "cream":    [r"\bcream\b", r"creamy", r"\bkem rửa", r"\bkem tẩy"],
    "oil":      [r"cleansing oil", r"oil cleanser", r"dầu tẩy", r"dầu rửa"],
    "micellar": [r"micellar", r"tẩy trang"],
    "powder":   [r"\bpowder\b", r"bột rửa", r"enzyme powder"],
    "clay":     [r"\bclay\b", r"mud mask", r"đất sét"],
}

CLAIMS = {
    "oily":      [r"oily skin", r"oil control", r"kiềm dầu", r"da dầu"],
    "dry":       [r"dry skin", r"da khô", r"hydrating", r"cấp ẩm"],
    "acne":      [r"\bacne\b", r"anti-acne", r"\bmụn\b"],
    "sensitive": [r"sensitive skin", r"da nhạy cảm", r"nhạy cảm"],
    "whitening": [r"whitening", r"brightening", r"trắng da", r"sáng da"],
    "antiaging": [r"anti.?aging", r"anti.?wrinkle", r"chống lão hóa"],
}

FORMAT_NL = {
    "foam":     "a foaming cleanser",
    "gel":      "a gel cleanser",
    "cream":    "a cream cleanser",
    "oil":      "a cleansing oil",
    "micellar": "a micellar / makeup-removing cleanser",
    "powder":   "a powder cleanser",
    "clay":     "a clay-based cleanser",
}

CLAIM_NL = {
    "oily":      "for oily skin / oil control",
    "dry":       "for dry skin / hydrating",
    "acne":      "for acne-prone skin",
    "sensitive": "for sensitive skin",
    "whitening": "for brightening / whitening",
    "antiaging": "anti-aging",
}


# ----- shared helpers -----

def load_products() -> pd.DataFrame:
    df = pd.read_parquet(PRODUCTS_PARQUET)
    df = df.drop_duplicates("product_id", keep="first").reset_index(drop=True)
    df["brand_norm"] = df["translated_brand"].fillna(df["brand"]).str.strip()
    df["text_en"] = (
        df["translated_name"].fillna("") + " ||| " + df["translated_description"].fillna("")
    ).str.lower()
    df["text_vn"] = (
        df["product_name"].fillna("") + " ||| " + df["processed_description"].fillna("")
    ).str.lower()
    df["formats"] = [_tags(e, v, FORMATS) for e, v in zip(df["text_en"], df["text_vn"])]
    df["claims"]  = [_tags(e, v, CLAIMS)  for e, v in zip(df["text_en"], df["text_vn"])]
    return df


def _tags(en: str, vn: str, table: dict[str, list[str]]) -> frozenset[str]:
    out = set()
    for k, pats in table.items():
        if any(re.search(p, en) for p in pats) or any(re.search(p, vn) for p in pats):
            out.add(k)
    return frozenset(out)


def _entry(rule: str, ref, tgt, question_en: str, **extra) -> dict:
    return {
        "rule": rule,
        "ref": ref,
        "tgt": tgt,
        "question_en": question_en,
        "extra": extra,
    }


# ----- Rule 1: price swap (existing) -----

def cheaper_pairs(df: pd.DataFrame, max_pairs: int) -> list[dict]:
    out = []
    for brand, g in df.groupby("brand_norm"):
        if not brand or len(g) < 3:
            continue
        g = g.sort_values("price").reset_index(drop=True)
        ref, tgt = g.iloc[-1], g.iloc[0]
        if ref["price"] / max(tgt["price"], 1) < 1.5:
            continue
        others = g.iloc[1:-1]
        lo, hi = tgt["price"] * 0.75, tgt["price"] * 1.25
        if ((others["price"] >= lo) & (others["price"] <= hi)).any():
            continue
        price_k = round(tgt["price"] / 1000)
        out.append(_entry(
            "cheaper_same_brand", ref, tgt,
            f"Looking for the same brand ({brand}) but cheaper, around {price_k}k VND.",
            ref_price=int(ref["price"]), tgt_price=int(tgt["price"]),
            ref_brand=brand,
        ))
    random.Random(SEED).shuffle(out)
    return out[:max_pairs]


def pricier_pairs(df: pd.DataFrame, max_pairs: int) -> list[dict]:
    out = []
    for brand, g in df.groupby("brand_norm"):
        if not brand or len(g) < 3:
            continue
        g = g.sort_values("price").reset_index(drop=True)
        ref, tgt = g.iloc[0], g.iloc[-1]
        if tgt["price"] / max(ref["price"], 1) < 1.5:
            continue
        others = g.iloc[1:-1]
        lo, hi = tgt["price"] * 0.75, tgt["price"] * 1.25
        if ((others["price"] >= lo) & (others["price"] <= hi)).any():
            continue
        price_k = round(tgt["price"] / 1000)
        out.append(_entry(
            "pricier_same_brand", ref, tgt,
            f"I'd like a more premium option from the same brand ({brand}), around {price_k}k VND.",
            ref_price=int(ref["price"]), tgt_price=int(tgt["price"]),
            ref_brand=brand,
        ))
    random.Random(SEED + 1).shuffle(out)
    return out[:max_pairs]


# ----- Rule 2: format swap (within brand) -----

def format_swap_pairs(df: pd.DataFrame, max_pairs: int) -> list[dict]:
    # Only consider products with exactly one format tag — ambiguity kills uniqueness.
    df1 = df[df["formats"].apply(lambda s: len(s) == 1)].copy()
    df1["format"] = df1["formats"].apply(lambda s: next(iter(s)))
    out = []
    rng = random.Random(SEED + 2)
    for brand, g in df1.groupby("brand_norm"):
        if not brand or len(g) < 2:
            continue
        by_fmt: dict[str, list] = {}
        for _, row in g.iterrows():
            by_fmt.setdefault(row["format"], []).append(row)
        fmts = list(by_fmt)
        if len(fmts) < 2:
            continue
        # Pick ordered (ref_fmt, tgt_fmt) where tgt is a unique exemplar in brand.
        for ref_fmt in fmts:
            for tgt_fmt in fmts:
                if ref_fmt == tgt_fmt:
                    continue
                if len(by_fmt[tgt_fmt]) != 1:
                    continue  # uniqueness in brand for target format
                tgt = by_fmt[tgt_fmt][0]
                # pick any ref of ref_fmt; prefer the one closest in price to tgt
                refs = sorted(by_fmt[ref_fmt], key=lambda r: abs(r["price"] - tgt["price"]))
                ref = refs[0]
                if ref["product_id"] == tgt["product_id"]:
                    continue
                out.append(_entry(
                    "format_swap", ref, tgt,
                    f"Same brand ({brand}), but I want {FORMAT_NL[tgt_fmt]} instead of {FORMAT_NL[ref_fmt]}.",
                    ref_brand=brand, ref_format=ref_fmt, tgt_format=tgt_fmt,
                ))
    rng.shuffle(out)
    return out[:max_pairs]


# ----- Rule 3: brand swap at similar price -----

def brand_swap_pairs(df: pd.DataFrame, max_pairs: int) -> list[dict]:
    # Pair (brand_A, brand_B) with same single-format tag and same single-claim tag,
    # within ±15% price. Uniqueness: target is the *only* product in brand_B matching
    # that (format, claim) within ±25% of ref price.
    df1 = df[df["formats"].apply(lambda s: len(s) == 1)
             & df["claims"].apply(lambda s: len(s) == 1)].copy()
    df1["format"] = df1["formats"].apply(lambda s: next(iter(s)))
    df1["claim"]  = df1["claims"].apply(lambda s: next(iter(s)))
    out = []
    rng = random.Random(SEED + 3)
    # bucket by (format, claim) for speed
    for (fmt, claim), g in df1.groupby(["format", "claim"]):
        if len(g) < 2:
            continue
        rows = list(g.itertuples(index=False))
        for i, ref in enumerate(rows):
            lo, hi = ref.price * 0.85, ref.price * 1.15
            cand = [r for r in rows
                    if r.brand_norm and r.brand_norm != ref.brand_norm
                    and lo <= r.price <= hi]
            if len(cand) != 1:
                continue
            tgt = cand[0]
            # Uniqueness across whole corpus: only one product in tgt brand with this (fmt, claim) within ±25%
            lo2, hi2 = ref.price * 0.75, ref.price * 1.25
            same_brand_fc = df1[(df1["brand_norm"] == tgt.brand_norm)
                                & (df1["format"] == fmt)
                                & (df1["claim"] == claim)
                                & (df1["price"] >= lo2) & (df1["price"] <= hi2)]
            if len(same_brand_fc) != 1:
                continue
            out.append(_entry(
                "brand_swap",
                pd.Series(ref._asdict()), pd.Series(tgt._asdict()),
                f"I want something similar — {FORMAT_NL[fmt]} {CLAIM_NL[claim]} — but from {tgt.brand_norm} instead of {ref.brand_norm}.",
                ref_brand=ref.brand_norm, tgt_brand=tgt.brand_norm,
                format=fmt, claim=claim,
            ))
    rng.shuffle(out)
    # Dedup by (ref_id, tgt_id)
    seen = set()
    uniq = []
    for p in out:
        key = (str(p["ref"]["product_id"]), str(p["tgt"]["product_id"]))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq[:max_pairs]


# ----- Rule 4: claim swap (within brand) -----

def claim_swap_pairs(df: pd.DataFrame, max_pairs: int) -> list[dict]:
    df1 = df[df["claims"].apply(lambda s: len(s) == 1)].copy()
    df1["claim"] = df1["claims"].apply(lambda s: next(iter(s)))
    out = []
    rng = random.Random(SEED + 4)
    for brand, g in df1.groupby("brand_norm"):
        if not brand or len(g) < 2:
            continue
        by_claim: dict[str, list] = {}
        for _, row in g.iterrows():
            by_claim.setdefault(row["claim"], []).append(row)
        if len(by_claim) < 2:
            continue
        for ref_claim in by_claim:
            for tgt_claim in by_claim:
                if ref_claim == tgt_claim:
                    continue
                if len(by_claim[tgt_claim]) != 1:
                    continue
                tgt = by_claim[tgt_claim][0]
                refs = sorted(by_claim[ref_claim], key=lambda r: abs(r["price"] - tgt["price"]))
                ref = refs[0]
                if ref["product_id"] == tgt["product_id"]:
                    continue
                out.append(_entry(
                    "claim_swap", ref, tgt,
                    f"Same brand ({brand}), but I need one {CLAIM_NL[tgt_claim]} instead of {CLAIM_NL[ref_claim]}.",
                    ref_brand=brand, ref_claim=ref_claim, tgt_claim=tgt_claim,
                ))
    rng.shuffle(out)
    return out[:max_pairs]


# ----- VN translation -----

def translate_to_vietnamese(en_questions: list[str]) -> list[str]:
    """Batched Sonnet call: render EN modifications as natural VN."""
    import anthropic

    from src.eval.models import resolve

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = resolve("sonnet")
    # Chunk to keep prompts manageable.
    out: list[str] = []
    CHUNK = 40
    for i in range(0, len(en_questions), CHUNK):
        batch = en_questions[i:i + CHUNK]
        numbered = "\n".join(f"{j+1}. {q}" for j, q in enumerate(batch))
        prompt = (
            "Translate each numbered shopping query below into natural conversational "
            "Vietnamese as a Vietnamese customer would type it on an e-commerce site. "
            "Keep brand names in their original Latin spelling. Reply with the same "
            "numbered list, one translation per line, no extra commentary.\n\n"
            f"{numbered}"
        )
        msg = client.messages.create(
            model=model, max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        got = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if "." in line[:4]:
                line = line.split(".", 1)[1].strip()
            got.append(line)
        if len(got) != len(batch):
            raise RuntimeError(f"Translation count mismatch in chunk {i}: got {len(got)}, expected {len(batch)}")
        out.extend(got)
    return out


# ----- main -----

RULES = {
    "cheaper":     cheaper_pairs,
    "pricier":     pricier_pairs,
    "format":      format_swap_pairs,
    "brand":       brand_swap_pairs,
    "claim":       claim_swap_pairs,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", default="cheaper,pricier,format,brand,claim",
                        help="Comma-separated subset of: " + ",".join(RULES))
    parser.add_argument("--max-per-rule", type=int, default=100)
    parser.add_argument("--translate", action="store_true")
    args = parser.parse_args()

    rules = [r.strip() for r in args.rules.split(",") if r.strip()]
    unknown = [r for r in rules if r not in RULES]
    if unknown:
        raise SystemExit(f"Unknown rules: {unknown}. Available: {list(RULES)}")

    df = load_products()
    print(f"Loaded {len(df)} unique products across {df['brand_norm'].nunique()} brands.")

    all_pairs: list[dict] = []
    counts: dict[str, int] = {}
    for r in rules:
        # price-swap rules use the original 20-per-rule cap by default for backwards-compat,
        # the rest take --max-per-rule.
        cap = 20 if r in ("cheaper", "pricier") else args.max_per_rule
        pairs = RULES[r](df, cap)
        counts[r] = len(pairs)
        all_pairs.extend(pairs)
        print(f"  rule={r:8s} pairs={len(pairs)}")

    if args.translate:
        print(f"Translating {len(all_pairs)} queries to Vietnamese via Sonnet...")
        en = [p["question_en"] for p in all_pairs]
        vn = translate_to_vietnamese(en)
    else:
        vn = [p["question_en"] for p in all_pairs]

    queries: dict[str, dict] = {}
    by_rule: dict[str, dict[str, dict]] = {}
    for i, p in enumerate(all_pairs):
        ref, tgt = p["ref"], p["tgt"]
        qid = f"DEV-{i:04d}"
        entry = {
            "image_path": ref["image_path"],
            "product_id": str(ref["product_id"]),
            "question": vn[i],
            "translated_question": p["question_en"],
            "answer_product_id": [str(tgt["product_id"])],
            "rule": p["rule"],
            "ref_price": int(ref["price"]),
            "tgt_price": int(tgt["price"]),
            **{k: (int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
               for k, v in p["extra"].items()},
        }
        queries[qid] = entry
        by_rule.setdefault(p["rule"], {})[qid] = entry

    OUT_PATH.write_text(json.dumps(queries, indent=2, ensure_ascii=False))
    print(f"Wrote {len(queries)} dev queries → {OUT_PATH.relative_to(ROOT)}")

    OUT_DIR.mkdir(exist_ok=True)
    for rule, entries in by_rule.items():
        path = OUT_DIR / f"{rule}.json"
        path.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
        print(f"  slice: {rule:24s} {len(entries):4d} → {path.relative_to(ROOT)}")

    print("\nSummary:")
    for r in rules:
        print(f"  {r:8s} requested ≤{20 if r in ('cheaper','pricier') else args.max_per_rule:4d}  built {counts[r]:4d}")


if __name__ == "__main__":
    main()
