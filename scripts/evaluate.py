"""
Evaluate a CIR system run against the CAMEO queries.

Usage:
    python scripts/evaluate.py --run path/to/run.json [--queries data/queries_sample.json]
                               [--k 5 10 20] [--use-judge] [--judge-k 20]

Run file format (JSON):
    {
        "CIR-1": ["product_id_1", "product_id_2", ...],
        ...
    }

Queries file format (JSON) — same as the official sample_submission.json:
    {
        "CIR-1": {
            "image_path": "...",
            "product_id": "1869",
            "question": "...",
            "answer_product_id": ["1654"]   // optional ground truth
        },
        ...
    }

Evaluation modes:
  1. Ground truth only  — if answer_product_id is present, use it directly (score=2).
  2. Claude-as-judge    — pass --use-judge to score non-ground-truth retrieved products
                          using Claude (requires ANTHROPIC_API_KEY env var).
  3. Mixed              — always uses ground truth when available, judges the rest.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.metrics import evaluate, print_results


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_qrels_from_ground_truth(queries: dict) -> dict:
    """
    Build qrels from the answer_product_id fields in the query file.
    Only works for queries that have ground truth.
    """
    qrels = {}
    for qid, q in queries.items():
        gts = q.get("answer_product_id", [])
        if gts:
            qrels[qid] = {str(pid): 2 for pid in gts}
    return qrels


def main():
    parser = argparse.ArgumentParser(description="Evaluate a CAMEO CIR run")
    parser.add_argument("--run",     required=True, help="Path to run JSON file")
    parser.add_argument("--queries", default=str(ROOT / "data" / "queries_sample.json"),
                        help="Path to queries JSON file")
    parser.add_argument("--k",       nargs="+", type=int, default=[5, 10, 20],
                        help="Values of k for NDCG@k and Recall@k")
    parser.add_argument("--use-judge", action="store_true",
                        help="Use Claude to judge relevance of non-ground-truth results")
    parser.add_argument("--use-images", action="store_true",
                        help="Pass reference and retrieved product images to the judge")
    parser.add_argument("--judge-model", default="haiku",
                        help="Model alias or full ID for the judge (default: haiku). "
                             "Aliases: haiku, sonnet, opus")
    parser.add_argument("--judge-k",   type=int, default=20,
                        help="How many top results to judge per query (default 20)")
    parser.add_argument("--products",  default=str(ROOT / "data" / "raw" / "products__test.parquet"),
                        help="Path to products parquet (needed for judge mode)")
    args = parser.parse_args()

    # Load inputs
    queries = load_json(args.queries)
    run     = load_json(args.run)
    print(f"Queries loaded : {len(queries)}")
    print(f"Run loaded     : {len(run)} queries, "
          f"avg {sum(len(v) for v in run.values()) / max(len(run),1):.0f} results each")

    # Build qrels
    if args.use_judge:
        import pandas as pd
        from src.eval.judge import ClaudeJudge

        products = pd.read_parquet(args.products)
        judge = ClaudeJudge(model=args.judge_model, use_images=args.use_images)
        print(f"\nRunning Claude-as-judge (top-{args.judge_k} per query) ...")
        qrels = judge.score_run(queries, run, products, k=args.judge_k, verbose=True)
    else:
        qrels = build_qrels_from_ground_truth(queries)
        n_with_gt = sum(1 for q in queries.values() if q.get("answer_product_id"))
        n_without = len(queries) - n_with_gt
        if n_without:
            print(f"WARNING: {n_without} queries have no ground truth and --use-judge "
                  f"is off — they will be skipped.")
        qrels = {qid: v for qid, v in qrels.items() if v}

    if not qrels:
        print("ERROR: No qrels to evaluate against. "
              "Add answer_product_id to queries or use --use-judge.")
        sys.exit(1)

    # Evaluate
    results = evaluate(qrels, run, ks=args.k)
    print_results(results, title=f"Results — {Path(args.run).stem}")

    # Save results alongside the run file
    out_path = Path(args.run).with_suffix(".eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
