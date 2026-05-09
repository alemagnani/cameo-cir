"""
Compare multiple Claude judge models on the same run.

For each model:
  - Scores the top-k retrieved products per query
  - Computes NDCG@K and MAP using those scores as qrels
  - Shows per-product score agreement vs gold standard and vs other models

Usage:
    python scripts/compare_judges.py --run path/to/run.json
                                     [--models haiku sonnet opus]
                                     [--queries data/queries_sample.json]
                                     [--k 10] [--judge-k 20]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.models import resolve, display_name, DEFAULT_JUDGE_MODELS
from src.eval.metrics import evaluate, ndcg_at_k, average_precision


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_gt_qrels(queries):
    return {
        qid: {str(pid): 2 for pid in q.get("answer_product_id", [])}
        for qid, q in queries.items()
        if q.get("answer_product_id")
    }


def run_judge(model_alias, queries, run, products_df, k):
    from src.eval.judge import ClaudeJudge
    judge = ClaudeJudge(model=model_alias)
    return judge.score_run(queries, run, products_df, k=k, verbose=False)


def main():
    parser = argparse.ArgumentParser(description="Compare Claude judge models on a CIR run")
    parser.add_argument("--run",     required=True)
    parser.add_argument("--queries", default=str(ROOT / "data" / "queries_sample.json"))
    parser.add_argument("--models",  nargs="+", default=DEFAULT_JUDGE_MODELS,
                        help="Model aliases or full IDs to compare (default: haiku sonnet opus)")
    parser.add_argument("--k",       type=int, default=10, help="k for NDCG@k (default 10)")
    parser.add_argument("--judge-k", type=int, default=20, help="Top-k products to judge per query")
    parser.add_argument("--products", default=str(ROOT / "data" / "raw" / "products__test.parquet"))
    args = parser.parse_args()

    import pandas as pd
    queries  = load_json(args.queries)
    run      = load_json(args.run)
    products = pd.read_parquet(args.products)
    prod_lookup = (
        products.drop_duplicates(subset="product_id")
        .set_index("product_id")[["translated_name"]]
        .to_dict("index")
    )

    gt_qrels = build_gt_qrels(queries)
    print(f"Queries: {len(queries)}  |  GT queries: {len(gt_qrels)}")
    print(f"Models : {args.models}")
    print(f"Judge-k: {args.judge_k}  |  Eval-k: {args.k}\n")

    # Collect qrels from each model
    all_qrels = {}
    for model_alias in args.models:
        mid = resolve(model_alias)
        dname = display_name(mid)
        print(f"=== {dname} ({mid}) ===")
        all_qrels[model_alias] = run_judge(model_alias, queries, run, products, args.judge_k)
        print()

    # --- Summary table: aggregate metrics per model ---
    print("=" * 65)
    print(f"  {'Model':<10}  {'NDCG@'+str(args.k):<12}  {'MAP':<10}  {'# judged'}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*8}")

    for model_alias in args.models:
        qrels = all_qrels[model_alias]
        # Only evaluate queries where this model produced scores
        r = evaluate(qrels, run, ks=[args.k])
        n_judged = sum(len(v) for v in qrels.values())
        dname = display_name(resolve(model_alias))
        print(f"  {dname:<10}  {r[f'ndcg@{args.k}']:<12.4f}  {r['map']:<10.4f}  {n_judged}")

    # --- Per-product score table for each query ---
    print()
    for qid, query in queries.items():
        ranked = [str(p) for p in run.get(qid, [])[:args.judge_k]]
        gt_pids = {str(p) for p in query.get("answer_product_id", [])}

        print(f"  Query: {qid}")
        print(f"  Q: \"{query['question'][:90]}\"")
        print()

        col_w = 10
        header = f"  {'Rank':<5}  {'product_id':<14}  {'GT':<4}" + \
                 "".join(f"  {display_name(resolve(m)):<{col_w}}" for m in args.models) + \
                 "  Name"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for rank, pid in enumerate(ranked, 1):
            gt_mark = "✓" if pid in gt_pids else ""
            scores = "".join(
                f"  {all_qrels[m].get(qid, {}).get(pid, '-'):<{col_w}}"
                for m in args.models
            )
            name = prod_lookup.get(pid, {}).get("translated_name", "?")[:45]
            print(f"  {rank:<5}  {pid:<14}  {gt_mark:<4}{scores}  {name}")
        print()

    # --- Inter-model agreement on non-GT products ---
    if len(args.models) >= 2:
        print("=" * 65)
        print("  Inter-model score agreement (non-GT products)")
        print("=" * 65)
        from itertools import combinations
        for m1, m2 in combinations(args.models, 2):
            d1 = display_name(resolve(m1))
            d2 = display_name(resolve(m2))
            matches = total = 0
            for qid in queries:
                pids = set(all_qrels[m1].get(qid, {})) & set(all_qrels[m2].get(qid, {}))
                gt = {str(p) for p in queries[qid].get("answer_product_id", [])}
                for pid in pids - gt:
                    total += 1
                    if all_qrels[m1][qid][pid] == all_qrels[m2][qid][pid]:
                        matches += 1
            pct = 100 * matches / total if total else 0
            print(f"  {d1} vs {d2}: {matches}/{total} agree ({pct:.0f}%)")


if __name__ == "__main__":
    main()
