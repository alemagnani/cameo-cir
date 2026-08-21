#!/usr/bin/env bash
# Phase B — run candidate VLM rerankers over the validation manifest on a GPU box.
# Reads model specs from $WORK/models.txt (one per line):
#     tag|hf_id|tensor_parallel|extra_args
# Resilient: each model gets a 3-job smoke test first; on failure we skip to the
# next so one broken integration doesn't waste the session. Results written per-model.
set -uo pipefail

WORK=${WORK:-/workspace/gpu_rerank}
MANIFEST=${MANIFEST:-$WORK/manifest.jsonl}
IMAGES=${IMAGES:-$WORK/images}
OUT=$WORK/out
SCRIPT=$WORK/vllm_rerank.py
MODELS_FILE=${MODELS_FILE:-$WORK/models.txt}
mkdir -p "$OUT"

run_model () {
  local tag=$1 hf=$2 tp=$3 extra=$4
  echo "############################################################"
  echo "### MODEL $tag  ($hf)  tp=$tp  extra='$extra'"
  echo "############################################################"
  echo ">>> smoke test (3 jobs)"
  if ! python "$SCRIPT" --manifest "$MANIFEST" --images-dir "$IMAGES" \
        --model "$hf" --tensor-parallel "$tp" $extra \
        --out "$OUT/smoke_$tag.json" --limit 3 2>&1; then
    echo "!!! $tag SMOKE FAILED — skipping"
    return 1
  fi
  echo ">>> full run ($tag)"
  if ! python "$SCRIPT" --manifest "$MANIFEST" --images-dir "$IMAGES" \
        --model "$hf" --tensor-parallel "$tp" $extra \
        --out "$OUT/reranked_$tag.json" 2>&1; then
    echo "!!! $tag FULL RUN FAILED"
    return 1
  fi
  echo "<<< $tag DONE -> $OUT/reranked_$tag.json"
}

while IFS='|' read -r tag hf tp extra; do
  [ -z "${tag:-}" ] && continue
  case "$tag" in \#*) continue;; esac
  run_model "$tag" "$hf" "$tp" "${extra:-}" || echo "=== continuing past $tag ==="
done < "$MODELS_FILE"

echo "=== ALL MODELS ATTEMPTED ==="
ls -la "$OUT"/reranked_*.json 2>/dev/null
