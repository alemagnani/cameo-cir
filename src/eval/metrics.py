"""
Standard IR metrics for CAMEO CIR evaluation.

Relevance scale (matching task rubric):
  0 — not relevant
  1 — partially relevant
  2 — fully relevant

A qrel is a dict:  {query_id: {product_id: relevance_score}}
A run  is a dict:  {query_id: [product_id, ...]}   (ranked, best first)
"""

import math
from typing import Dict, List


Qrels = Dict[str, Dict[str, int]]   # query_id → {product_id → rel}
Run   = Dict[str, List[str]]        # query_id → [product_id, ...]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _dcg(rels: List[int], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    return sum(
        rel / math.log2(i + 2)          # i+2 because i is 0-based
        for i, rel in enumerate(rels[:k])
    )


def _ideal_rels(qrel: Dict[str, int], k: int) -> List[int]:
    """Sorted relevance list for a single query (ideal ranking)."""
    return sorted(qrel.values(), reverse=True)[:k]


# ---------------------------------------------------------------------------
# Per-query metrics
# ---------------------------------------------------------------------------

def ndcg_at_k(qrel: Dict[str, int], ranked: List[str], k: int) -> float:
    """NDCG@k for a single query."""
    rels   = [qrel.get(pid, 0) for pid in ranked[:k]]
    ideal  = _ideal_rels(qrel, k)
    idcg   = _dcg(ideal, k)
    return _dcg(rels, k) / idcg if idcg > 0 else 0.0


def average_precision(qrel: Dict[str, int], ranked: List[str]) -> float:
    """Average Precision for a single query (binary: rel >= 1 counts as relevant)."""
    relevant = {pid for pid, r in qrel.items() if r >= 1}
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, pid in enumerate(ranked, 1):
        if pid in relevant:
            hits += 1
            precision_sum += hits / i
    return precision_sum / len(relevant)


def recall_at_k(qrel: Dict[str, int], ranked: List[str], k: int) -> float:
    """Recall@k for a single query (binary relevance)."""
    relevant = {pid for pid, r in qrel.items() if r >= 1}
    if not relevant:
        return 0.0
    retrieved = set(ranked[:k])
    return len(relevant & retrieved) / len(relevant)


# ---------------------------------------------------------------------------
# Aggregate over all queries
# ---------------------------------------------------------------------------

def evaluate(qrels: Qrels, run: Run, ks: List[int] = (5, 10, 20)) -> Dict:
    """
    Compute NDCG@k, MAP, and Recall@k across all queries.

    Returns a dict with keys like 'ndcg@10', 'map', 'recall@10'.
    Also includes per-query breakdowns under 'per_query'.
    """
    per_query = {}
    ndcg_sums  = {k: 0.0 for k in ks}
    recall_sums = {k: 0.0 for k in ks}
    ap_sum = 0.0
    n = 0

    for qid, qrel in qrels.items():
        ranked = run.get(qid, [])
        per_query[qid] = {
            **{f"ndcg@{k}": ndcg_at_k(qrel, ranked, k) for k in ks},
            **{f"recall@{k}": recall_at_k(qrel, ranked, k) for k in ks},
            "ap": average_precision(qrel, ranked),
        }
        ap_sum += per_query[qid]["ap"]
        for k in ks:
            ndcg_sums[k]   += per_query[qid][f"ndcg@{k}"]
            recall_sums[k] += per_query[qid][f"recall@{k}"]
        n += 1

    if n == 0:
        return {"error": "no queries evaluated"}

    results = {
        "num_queries": n,
        "map": ap_sum / n,
        **{f"ndcg@{k}": ndcg_sums[k] / n for k in ks},
        **{f"recall@{k}": recall_sums[k] / n for k in ks},
        "per_query": per_query,
    }
    return results


def print_results(results: Dict, title: str = "Evaluation Results") -> None:
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"  Queries evaluated: {results['num_queries']}")
    print(f"{'='*50}")
    for key, val in results.items():
        if key in ("per_query", "num_queries", "error"):
            continue
        print(f"  {key:<15} {val:.4f}")
    print()
    if "per_query" in results:
        print("  Per-query breakdown:")
        for qid, scores in results["per_query"].items():
            ndcg10 = scores.get("ndcg@10", scores.get("ndcg@5", "?"))
            ap     = scores.get("ap", "?")
            print(f"    {qid}: NDCG@10={ndcg10:.4f}  AP={ap:.4f}")
