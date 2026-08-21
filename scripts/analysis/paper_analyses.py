"""
Offline analyses for the NTCIR-19 CAMEO paper (Codex-recommended A-D).
All operate on saved dev runs + gold fields; no GPU.

A: metadata component ablation (brand / +price / +format / +claim / all), gold fields, per rule.
B: gold-vs-extracted metadata retrieval + field-level extraction accuracy.
C: per-rule gold-rank shift under visual reranking (representative visual pool pair).
D: Stage-1 candidate-recall / oracle-NDCG@10 upper bound vs actual NDCG@10.

Ablation-run provenance (all with --use-gold --topk 50 --price-sigma 0.4):
  abl_brand.json                     : --terms brand
  abl_brand_price.json               : --terms brand,price
  abl_brand_format.json              : --terms brand,format
  abl_brand_claim.json               : --terms brand,claim
  abl_brand_price_format_claim.json  : = meta_gold_dev.json (--terms brand,price,format,claim)
Runs sort by (-score, product_id) for deterministic tie-breaking.
"""
import json, sys
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.eval.metrics import evaluate

Q = json.loads((ROOT / "data/queries_dev.json").read_text())
RULES = ["cheaper_same_brand", "pricier_same_brand", "format_swap", "brand_swap", "claim_swap"]
GOLD = {qid: q["answer_product_id"] for qid, q in Q.items() if q.get("answer_product_id")}
RULE = {qid: q.get("rule", "_") for qid, q in Q.items()}


def load(run):
    d = json.loads((ROOT / run).read_text())
    return {k: (v if isinstance(v, list) else v) for k, v in d.items()}


def per_rule_ndcg(run, k=10):
    out = {}
    for rule in RULES + ["_overall"]:
        qids = [q for q in GOLD if (rule == "_overall" or RULE[q] == rule)]
        qrels = {q: {p: 2 for p in GOLD[q]} for q in qids}
        slice_run = {q: run.get(q, []) for q in qids}
        res = evaluate(qrels, slice_run, ks=[k])
        out[rule] = res[f"ndcg@{k}"]
    return out


def recall_at(run, k):
    qids = list(GOLD)
    qrels = {q: {p: 2 for p in GOLD[q]} for q in qids}
    slice_run = {q: run.get(q, []) for q in qids}
    return evaluate(qrels, slice_run, ks=[k])[f"recall@{k}"]


def fmt_row(name, d, cols):
    return f"{name:<26s}" + "".join(f"{d[c]:>8.3f}" for c in cols)


print("\n===== A. METADATA COMPONENT ABLATION (gold fields, metadata-only, NDCG@10 per rule) =====")
abl = {
    "brand only": "data/runs/analysis/abl_brand.json",
    "brand+price": "data/runs/analysis/abl_brand_price.json",
    "brand+format": "data/runs/analysis/abl_brand_format.json",
    "brand+claim": "data/runs/analysis/abl_brand_claim.json",
    "brand+price+format+claim": "data/runs/analysis/abl_brand_price_format_claim.json",
}
cols = ["cheaper_same_brand", "pricier_same_brand", "format_swap", "claim_swap", "brand_swap", "_overall"]
short = {"cheaper_same_brand": "cheap", "pricier_same_brand": "pricier", "format_swap": "format",
         "claim_swap": "claim", "brand_swap": "brand", "_overall": "OVER"}
print(f"{'variant':<26s}" + "".join(f"{short[c]:>8s}" for c in cols))
for name, path in abl.items():
    print(fmt_row(name, per_rule_ndcg(load(path)), cols))

print("\n===== B. GOLD vs EXTRACTED metadata retrieval (metadata-only, NDCG@10 per rule) =====")
print(f"{'fields':<26s}" + "".join(f"{short[c]:>8s}" for c in cols))
print(fmt_row("gold", per_rule_ndcg(load("data/runs/metadata/meta_gold_dev.json")), cols))
print(fmt_row("extracted (Haiku)", per_rule_ndcg(load("data/runs/metadata/meta_ext_dev.json")), cols))

print("\n----- B2. field-level extraction accuracy (extracted vs gold, dev-429) -----")
ext = json.loads((ROOT / "data/eval_cache/extract_claude-haiku-4-5-20251001.json").read_text())


def norm(s):
    import re
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


# "price stated" is defined by rule type (only cheaper/pricier state a price in the text),
# not by the presence of a gold tgt_price (which is populated for all synthetic queries).
PRICE_RULES = {"cheaper_same_brand", "pricier_same_brand"}
brand_ok = brand_n = price_ok = price_n = fmt_cov = fmt_n = clm_cov = clm_n = 0
for qid, q in Q.items():
    e = ext.get(qid)
    if not e:
        continue
    rule = q.get("rule", "")
    gbrand = q.get("tgt_brand") if rule == "brand_swap" else q.get("ref_brand")
    if gbrand:
        brand_n += 1
        if norm(e.get("brand")) and (norm(e.get("brand")) in norm(gbrand) or norm(gbrand) in norm(e.get("brand"))):
            brand_ok += 1
    if rule in PRICE_RULES:  # queries that actually state a price
        price_n += 1
        ep, gprice = e.get("tgt_price"), q.get("tgt_price")
        if ep and gprice and abs(ep - gprice) / max(gprice, 1) < 0.05:
            price_ok += 1
    if q.get("tgt_format"):
        fmt_n += 1
        if e.get("tgt_format"):
            fmt_cov += 1
    if q.get("tgt_claim"):
        clm_n += 1
        if e.get("tgt_claim"):
            clm_cov += 1
print(f"  brand match (normalized):        {brand_ok}/{brand_n} = {brand_ok/max(brand_n,1):.3f}")
print(f"  price within 5% (price-stated):  {price_ok}/{price_n} = {price_ok/max(price_n,1):.3f}")
print(f"  format COVERAGE (non-null|gold): {fmt_cov}/{fmt_n} = {fmt_cov/max(fmt_n,1):.3f}")
print(f"  claim COVERAGE (non-null|gold):  {clm_cov}/{clm_n} = {clm_cov/max(clm_n,1):.3f}")

print("\n===== D. STAGE-1 CANDIDATE-RECALL / ORACLE UPPER BOUND vs ACTUAL NDCG@10 =====")
stage1 = {
    "SEARLE-Phi": "data/runs/searle_phi_a05_dev.json",
    "bigG+fusion (visual)": "data/runs/rerank_eval/encode_runs/oc_biggf2_dev.json",
    "metadata-fused (ext)": "data/runs/metadata/fuse_ext_prio_dev.json",
}
print(f"{'Stage-1':<22s}{'R@20':>8s}{'R@50':>8s}{'oracleNDCG@10':>15s}{'actualNDCG@10':>15s}")
for name, path in stage1.items():
    r = load(path)
    r20, r50 = recall_at(r, 20), recall_at(r, 50)
    # oracle NDCG@10 = 1.0 whenever gold is anywhere in the (<=50) retrieved list = recall@50
    print(f"{name:<22s}{r20:>8.3f}{r50:>8.3f}{r50:>15.3f}{per_rule_ndcg(r)['_overall']:>15.3f}")
print("  (oracle NDCG@10 = best achievable by reordering the fixed pool = candidate recall@pool-depth)")

print("\n===== C. GOLD-RANK SHIFT UNDER VISUAL RERANK (representative bigG pool + Qwen rerank) =====")
pre = load("data/runs/ar_6c9503931f22_dev.json")
post = load("data/runs/ar_6c9503931f22_dev.rerank.json")
print(f"  pool NDCG@10: pre-rerank {per_rule_ndcg(pre)['_overall']:.3f} -> "
      f"post-rerank {per_rule_ndcg(post)['_overall']:.3f}")
K = 20
agg = defaultdict(lambda: [0, 0, 0, 0])  # rule -> [n_gold_in_preK, improved, worsened, exited_topK]
for qid, gold in GOLD.items():
    if qid not in pre or qid not in post:
        continue
    g = gold[0]
    pl, ql = pre[qid][:K], post[qid]
    if g not in pl:
        continue
    rule = RULE[qid]
    agg[rule][0] += 1
    rp = pl.index(g)
    rq = ql.index(g) if g in ql[:K] else None
    if rq is None:
        agg[rule][3] += 1
    elif rq < rp:
        agg[rule][1] += 1
    elif rq > rp:
        agg[rule][2] += 1
print(f"{'rule':<20s}{'gold_in_preTop20':>17s}{'moved_up':>10s}{'moved_down':>12s}{'exited_top20':>14s}")
for rule in RULES + ["_overall"]:
    if rule == "_overall":
        tot = [sum(agg[r][i] for r in RULES) for i in range(4)]
    else:
        tot = agg[rule]
    if tot[0] == 0:
        continue
    print(f"{rule:<20s}{tot[0]:>17d}{tot[1]:>10d}{tot[2]:>12d}{tot[3]:>14d}")
print("  (down+exited = demotions of the gold item by the visual reranker)")
