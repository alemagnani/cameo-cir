"""
Compute per-rule + overall NDCG@k, Recall@k, MAP on a CIR run.

Returns a dict shaped:
  {
    "overall": {"ndcg@1": ..., "ndcg@5": ..., "map": ..., "n": 429, ...},
    "by_rule": {
      "cheaper_same_brand": {...same shape},
      "pricier_same_brand": {...},
      ...
    }
  }
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.metrics import evaluate  # noqa: E402

DEFAULT_KS = (1, 5, 10, 20)


def per_rule_metrics(run: dict, queries: dict, ks=DEFAULT_KS) -> dict:
    by_rule_q: dict[str, dict] = defaultdict(dict)
    for qid, q in queries.items():
        by_rule_q[q.get("rule", "_unknown")][qid] = q

    out: dict = {"overall": _slice_metrics(queries, run, ks), "by_rule": {}}
    for rule, slice_q in by_rule_q.items():
        out["by_rule"][rule] = _slice_metrics(slice_q, run, ks)
    return out


def _slice_metrics(slice_queries: dict, run: dict, ks) -> dict:
    qrels = {qid: {pid: 2 for pid in q["answer_product_id"]}
             for qid, q in slice_queries.items() if q.get("answer_product_id")}
    slice_run = {qid: run[qid] for qid in qrels if qid in run}
    if not slice_run:
        return {"n": 0}
    res = evaluate(qrels, slice_run, ks=ks)
    res["n"] = len(slice_run)
    return res
