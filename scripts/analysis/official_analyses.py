"""
Official-test-set analyses for the NTCIR-19 CAMEO paper (20 CIR topics, pooled human 3-point).
Complements scripts/analysis/paper_analyses.py (which covers the synthetic dev set).

Inputs (read-only):
  - data/submission/final/retrank_CIR_Run{1..5}.json  (our 5 submitted runs; official CIR-0..19)
  - /tmp/retrank_CIR.csv                               (organizers' official per-topic nDCG@5/@10)
  - overview Table 4 pool stats transcribed inline (Pool, Partial, Relevant per topic)
  - data/raw/products__test.parquet                    (brand list, for metadata-resolvability tagging)

VERIFIED run provenance (content-matched by candidate-SET Jaccard@50 vs official Stage-1 runs;
SEARLE-Phi and bigG pools are distinct at Jaccard 0.22, so the 1.000 matches are decisive):
  Run1  Stage-1 = FUSED (bigG a=0.2 + brand-union + symbolic metadata) -> Qwen2.5-VL-72B  [matches fused_official 1.00]
  Run2  Stage-1 = SEARLE-Phi -> Qwen2.5-VL-72B    | Run2/3/4 share the SEARLE-Phi pool (Jaccard 1.00)
  Run3  Stage-1 = SEARLE-Phi -> InternVL3-38B     |
  Run4  Stage-1 = SEARLE-Phi -> Qwen2.5-VL-32B    |
  Run5  Stage-1 = bigG + brand-union -> Qwen2.5-VL-32B  [matches official_bigg_a02_bu1 1.00]
What is / isn't a clean comparison on OFFICIAL data:
  * VALID  Reranker-model ablation : Run2 vs Run3 vs Run4 (IDENTICAL SEARLE-Phi pool, reranker varies)
  * VALID  Stage-1 backbone/pool   : both bigG-family runs (Run1 0.714, Run5 0.709) beat all three
                                     SEARLE-Phi runs (Run2/3/4 0.606-0.636) under matched rerankers
  * NOT a clean metadata ablation  : Run2->Run1 and Run4->Run5 change the whole Stage-1 FAMILY
                                     (SEARLE-Phi -> bigG+metadata), not just the metadata term.
                                     The clean metadata isolation lives in the DEV-SET ablation
                                     (paper_analyses.py: metadata fusion R@50 0.436->0.527).
"""
import csv, json, re, sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent.parent
SUB = ROOT / "data/submission/final"
CSV = Path("/tmp/retrank_CIR.csv")

RUN_CFG = {
    "Run1": ("FUSED (bigG+brand-union+meta)", "Qwen2.5-VL-72B", "bigG-meta"),
    "Run2": ("SEARLE-Phi", "Qwen2.5-VL-72B", "searle"),
    "Run3": ("SEARLE-Phi", "InternVL3-38B", "searle"),
    "Run4": ("SEARLE-Phi", "Qwen2.5-VL-32B", "searle"),
    "Run5": ("bigG + brand-union", "Qwen2.5-VL-32B", "bigG-meta"),
}

# Overview Table 4: topic -> (Pool, Partial(grade1), Relevant(grade2))
POOL = {
    "CIR-0": (63, 24, 17), "CIR-1": (78, 5, 31), "CIR-2": (87, 33, 18), "CIR-3": (70, 8, 54),
    "CIR-4": (101, 29, 40), "CIR-5": (81, 29, 9), "CIR-6": (84, 20, 49), "CIR-7": (78, 23, 35),
    "CIR-8": (89, 13, 13), "CIR-9": (114, 6, 66), "CIR-10": (92, 14, 5), "CIR-11": (87, 50, 7),
    "CIR-12": (103, 32, 38), "CIR-13": (100, 14, 10), "CIR-14": (92, 3, 62), "CIR-15": (107, 26, 31),
    "CIR-16": (74, 6, 47), "CIR-17": (91, 24, 17), "CIR-18": (107, 10, 13), "CIR-19": (72, 20, 26),
}


def load_run(n):
    d = json.loads((SUB / f"retrank_CIR_Run{n}.json").read_text())
    return {k: [str(x) for x in (v["answer_product_id"] if isinstance(v, dict) else v)] for k, v in d.items()}


def load_csv():
    """official per-topic nDCG@5/@10 keyed by (run, topic); plus n_rel_returned/num_rel_pool."""
    d = {}
    for row in csv.DictReader(open(CSV)):
        if row["topic"].startswith("ALL"):
            continue
        d[(row["run"], row["topic"])] = {
            "ndcg5": float(row["ndcg@5"]), "ndcg10": float(row["ndcg@10"]),
            "n_ret": int(row["n_returned"]), "n_rel_ret": int(row["n_rel_returned"]),
            "n_rel_pool": int(row["num_rel_pool"]),
        }
    return d


def macro(csvd, run, key):
    return mean(csvd[(run, t)][key] for t in POOL)


def pcorr(xs, ys):
    n = len(xs); mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** .5
    vy = sum((y - my) ** 2 for y in ys) ** .5
    return cov / (vx * vy) if vx and vy else float("nan")


def main():
    csvd = load_csv()
    topics = sorted(POOL, key=lambda t: int(t.split("-")[1]))
    runs = {n: load_run(n) for n in range(1, 6)}

    print("=" * 78)
    print("VERIFIED RUN CONFIG + OFFICIAL MACRO SCORES")
    print("=" * 78)
    print(f"{'Run':5s}{'Stage-1':40s}{'reranker':16s}{'nDCG@5':>8s}{'nDCG@10':>9s}")
    for r, (s1, rr, _) in RUN_CFG.items():
        print(f"{r:5s}{s1:40s}{rr:16s}{macro(csvd, r, 'ndcg5'):>8.4f}{macro(csvd, r, 'ndcg10'):>9.4f}")

    print("\n" + "=" * 78)
    print("A. RERANKER-MODEL ABLATION (Run2/3/4 = IDENTICAL SEARLE-Phi pool; reranker varies)")
    print("=" * 78)
    for r in ("Run2", "Run3", "Run4"):
        print(f"  {r} [{RUN_CFG[r][1]:14s}]  nDCG@5={macro(csvd,r,'ndcg5'):.4f}  nDCG@10={macro(csvd,r,'ndcg10'):.4f}")
    spread10 = max(macro(csvd, r, "ndcg10") for r in ("Run2", "Run3", "Run4")) - \
        min(macro(csvd, r, "ndcg10") for r in ("Run2", "Run3", "Run4"))
    print(f"  -> nDCG@10 spread across 3 rerankers on identical candidates: {spread10:.4f}")

    print("\n" + "=" * 78)
    print("B. STAGE-1 FAMILY EFFECT under matched rerankers (NOT a clean metadata ablation:")
    print("   SEARLE-Phi pool -> bigG+metadata pool; whole Stage-1 family changes). Per-topic W/T/L nDCG@10")
    print("=" * 78)
    for lo, hi, tag in [("Run2", "Run1", "Qwen-72B: SEARLE-Phi -> bigG-fused"),
                        ("Run4", "Run5", "Qwen-32B: SEARLE-Phi -> bigG+brand-union")]:
        w = t = l = 0
        for tp in topics:
            a, b = csvd[(lo, tp)]["ndcg10"], csvd[(hi, tp)]["ndcg10"]
            if b > a + 1e-9:
                w += 1
            elif b < a - 1e-9:
                l += 1
            else:
                t += 1
        print(f"  {tag}")
        print(f"    {lo} {macro(csvd,lo,'ndcg10'):.4f} -> {hi} {macro(csvd,hi,'ndcg10'):.4f}"
              f"  (delta {macro(csvd,hi,'ndcg10')-macro(csvd,lo,'ndcg10'):+.4f});"
              f" per-topic W/T/L = {w}/{t}/{l}")

    print("\n" + "=" * 78)
    print("C. PER-TOPIC METADATA-RESOLVABILITY vs SCORE (tests overview sec 6.1)")
    print("=" * 78)
    # brand list from corpus for objective brand-mention detection
    try:
        import pandas as pd
        prod = pd.read_parquet(ROOT / "data/raw/products__test.parquet")
        brands = sorted({re.sub(r"[^a-z0-9]", "", str(b).lower()) for b in prod["brand"].dropna()
                         if len(str(b)) >= 3}, key=len, reverse=True)
    except Exception as e:
        print("  (brand list unavailable:", e, "-- price-only tagging)")
        brands = []
    price_re = re.compile(r"\d[\d.,]*\s*(k|nghìn|ngàn|đ|vnd|d\b)", re.I)
    tags = {}
    for tp in topics:
        q = (runs[1][tp := tp] and "") or ""  # placeholder to keep tp
    # use the question text stored in the submission file
    subj = json.loads((SUB / "retrank_CIR_Run1.json").read_text())
    for tp in topics:
        q = str(subj[tp].get("question", "")).lower()
        has_price = bool(price_re.search(q))
        qn = re.sub(r"[^a-z0-9]", "", q)
        has_brand = any(b in qn for b in brands if len(b) >= 4)
        tags[tp] = {"price": has_price, "brand": has_brand, "meta": has_price or has_brand}
    meta_t = [tp for tp in topics if tags[tp]["meta"]]
    vis_t = [tp for tp in topics if not tags[tp]["meta"]]
    for label, group in [("metadata-resolvable (price or brand in query)", meta_t),
                         ("visual/other", vis_t)]:
        if group:
            print(f"  {label}: n={len(group)}  "
                  f"Run1 nDCG@10={mean(csvd[('Run1',tp)]['ndcg10'] for tp in group):.4f}  "
                  f"Run1 nDCG@5={mean(csvd[('Run1',tp)]['ndcg5'] for tp in group):.4f}")
    print(f"  topics tagged metadata-resolvable: {sorted(meta_t, key=lambda t:int(t.split('-')[1]))}")
    print(f"  topics tagged visual/other:        {sorted(vis_t, key=lambda t:int(t.split('-')[1]))}")

    print("\n" + "=" * 78)
    print("D. RECALL vs SCORE (recall of graded-relevant into our pool drives nDCG)")
    print("=" * 78)
    rec = [csvd[("Run1", tp)]["n_rel_ret"] / csvd[("Run1", tp)]["n_rel_pool"] for tp in topics]
    nd = [csvd[("Run1", tp)]["ndcg10"] for tp in topics]
    frac_full = [POOL[tp][2] / (POOL[tp][1] + POOL[tp][2]) for tp in topics]  # grade2 / (grade1+grade2)
    print(f"  corr( Run1 recall-of-relevant , Run1 nDCG@10 ) = {pcorr(rec, nd):+.3f}")
    print(f"  corr( frac fully-relevant(g2) , Run1 nDCG@10 ) = {pcorr(frac_full, nd):+.3f}")
    print(f"  corr( pool size               , Run1 nDCG@10 ) = {pcorr([POOL[tp][0] for tp in topics], nd):+.3f}")

    print("\n" + "=" * 78)
    print("E. PER-TOPIC BEST-OF-OUR-5 (how much a per-topic run oracle would gain)")
    print("=" * 78)
    best = mean(max(csvd[(f"Run{n}", tp)]["ndcg10"] for n in range(1, 6)) for tp in topics)
    print(f"  mean nDCG@10: Run1={macro(csvd,'Run1','ndcg10'):.4f}  best-of-5-per-topic={best:.4f}"
          f"  (headroom {best-macro(csvd,'Run1','ndcg10'):+.4f})")

    print("\n" + "=" * 78)
    print("F. WORST OFFICIAL TOPICS (Run1 nDCG@10) with pool context")
    print("=" * 78)
    for tp in sorted(topics, key=lambda t: csvd[("Run1", t)]["ndcg10"])[:6]:
        c = csvd[("Run1", tp)]
        p = POOL[tp]
        print(f"  {tp:7s} nDCG@10={c['ndcg10']:.3f}  recall={c['n_rel_ret']}/{c['n_rel_pool']}"
              f"  pool={p[0]} (g1={p[1]},g2={p[2]})  meta={tags[tp]['meta']}  q={subj[tp].get('question','')[:52]!r}")


if __name__ == "__main__":
    main()
