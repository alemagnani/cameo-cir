"""
Per-rule evaluation of a CIR run against data/queries_dev.json.

Splits the run by the `rule` field on each dev query, computes NDCG@k,
Recall@k, and MAP per rule, and prints one row per rule plus an overall row.

Usage:
  python scripts/eval_by_rule.py --run data/runs/searle_xl_dev.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.metrics import evaluate  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--queries", default=str(ROOT / "data" / "queries_dev.json"))
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10, 20])
    args = parser.parse_args()

    queries = json.loads(Path(args.queries).read_text())
    run = json.loads(Path(args.run).read_text())

    by_rule: dict[str, dict[str, dict]] = defaultdict(dict)
    for qid, q in queries.items():
        by_rule[q.get("rule", "_unknown")][qid] = q
    by_rule["_overall"] = queries

    header = f"{'rule':<22s} {'N':>4s}  " + "  ".join(f"NDCG@{k:<3d}" for k in args.ks) + "  " + "  ".join(f"R@{k:<4d}" for k in args.ks) + f"  {'MAP':>6s}"
    print(header)
    print("-" * len(header))

    for rule, slice_queries in by_rule.items():
        qrels = {qid: {pid: 2 for pid in q["answer_product_id"]}
                 for qid, q in slice_queries.items() if q.get("answer_product_id")}
        slice_run = {qid: run[qid] for qid in qrels if qid in run}
        if not slice_run:
            continue
        res = evaluate(qrels, slice_run, ks=args.ks)
        ndcg = [f"{res[f'ndcg@{k}']:.4f}" for k in args.ks]
        rec = [f"{res[f'recall@{k}']:.4f}" for k in args.ks]
        print(f"{rule:<22s} {len(slice_run):>4d}  " + "  ".join(f"{v:>7s}" for v in ndcg) + "  " + "  ".join(f"{v:>6s}" for v in rec) + f"  {res['map']:.4f}")


if __name__ == "__main__":
    main()
