#!/usr/bin/env bash
# Phase C — eval one reranked model (449-job json) vs ground truth + cached judge.
# Usage: bash eval_model.sh <tag>   (reads data/runs/rerank_eval/gpu/reranked_<tag>.json)
set -uo pipefail
cd "$(dirname "$0")/../.."
TAG=$1
RR=data/runs/rerank_eval/gpu/reranked_$TAG.json
OUT=data/runs/rerank_eval
mkdir -p "$OUT"

# official-20: splice into Phi official Stage-1, judge with cached Sonnet vision
python scripts/gpu_rerank/splice.py --stage1 data/runs/searle_phi_a05_official.json \
  --reranked "$RR" --out "$OUT/${TAG}_official.json" >/dev/null
# dev-429: splice into Phi dev Stage-1, restricted to the 429 rule qids
python scripts/gpu_rerank/splice.py --stage1 data/runs/searle_phi_a05_dev.json \
  --reranked "$RR" --only-qids data/queries_dev.json --out "$OUT/${TAG}_dev429.json" >/dev/null

echo "########## $TAG ##########"
echo "--- official-20 (Sonnet vision judge, cached) ---"
PYTHONPATH=. python scripts/evaluate.py --run "$OUT/${TAG}_official.json" \
  --queries data/queries_official_cir.json --use-judge --judge-model sonnet --use-images \
  --judge-k 20 --k 10 2>&1 | grep -aE "ndcg@10|recall@10" | head -3
echo "--- dev-429 (rule ground truth) ---"
PYTHONPATH=. python scripts/eval_by_rule.py --run "$OUT/${TAG}_dev429.json" 2>&1 | grep -aE "_overall|brand_swap"
