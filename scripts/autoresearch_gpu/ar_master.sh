#!/usr/bin/env bash
# GPU-native autoresearch loop. For each config in ar_roadmap.txt (resumable, skip-done):
#   Stage-1 encode (shared/cached) -> optional brand-union -> top-K manifest -> VLM rerank
#   -> free dev-429 eval (per-rule) -> append ledger.jsonl. Official-20 reranked run saved per
#   config for local judging of dev-winners. Always reaches ALL_DONE.
set -uo pipefail
GR=/workspace/gpu_rerank; CAM=/workspace/cameo; AR=/workspace/autoresearch_gpu
export HF_HOME=/workspace/hf VLLM_WORKER_MULTIPROC_METHOD=spawn
log(){ echo "[$(date +%H:%M:%S)] AR: $*"; }
mkdir -p "$AR/out" "$CAM/data/runs"
LEDGER="$AR/ledger.jsonl"

log "install deps"
pip install -q open_clip_torch pandas pyarrow opencv-python-headless 2>&1 | tail -1 || true
pip install -q -U vllm qwen-vl-utils 2>&1 | tail -1 || true
python -c "import torch;torch.zeros(1).cuda();print('CUDA_OK')" 2>&1 | tail -1 || log "CUDA FAIL"
until [ "$(find $CAM/data/raw/images -type f 2>/dev/null|wc -l)" -ge 4070 ]; do log "wait corpus"; sleep 15; done
cd "$CAM"

grep -v '^#' "$AR/ar_roadmap.txt" | while IFS='|' read -r tag bb pre alpha bu mt mhf topk; do
  [ -z "${tag:-}" ] && continue
  if [ -s "$LEDGER" ] && grep -q "\"tag\": \"$tag\"" "$LEDGER"; then log "[resume] $tag"; continue; fi
  bbt=$(echo "$bb" | tr 'A-Z/' 'a-z_'); at=$(echo "$alpha" | tr -d '.')
  log "=== CONFIG $tag ==="
  # Stage-1 encode (shared across bu/model; skip-existing)
  for qp in "queries_dev:dev" "queries_official_cir:official"; do
    qf=${qp%%:*}; qtag=${qp##*:}; S1="data/runs/s1_${bbt}_a${at}_${qtag}.json"
    [ -s "$S1" ] || python scripts/exp004/run_openclip_cir.py --queries "data/$qf.json" --out "$S1" \
      --model "$bb" --pretrained "$pre" --fuse-name-alpha "$alpha" --topk 50 \
      > "$AR/enc_${bbt}_a${at}_${qtag}.log" 2>&1 || log "ENC FAIL $tag $qtag"
  done
  S1DEV="data/runs/s1_${bbt}_a${at}_dev.json"; S1OFF="data/runs/s1_${bbt}_a${at}_official.json"
  if [ "$bu" = "1" ]; then
    BUDEV="data/runs/s1_${bbt}_a${at}_bu_dev.json"; BUOFF="data/runs/s1_${bbt}_a${at}_bu_official.json"
    [ -s "$BUDEV" ] || python scripts/gpu_rerank/brand_union.py --run "$S1DEV" --queries data/queries_dev.json --out "$BUDEV" >/dev/null 2>&1
    [ -s "$BUOFF" ] || python scripts/gpu_rerank/brand_union.py --run "$S1OFF" --queries data/queries_official_cir.json --out "$BUOFF" >/dev/null 2>&1
    S1DEV="$BUDEV"; S1OFF="$BUOFF"
  fi
  MDIR="$CAM/man_${bbt}_a${at}_bu${bu}_k${topk}"
  [ -s "$MDIR/manifest.jsonl" ] || python scripts/gpu_rerank/build_manifest.py \
    --pair "$S1OFF::data/queries_official_cir.json" --pair "$S1DEV::data/queries_dev.json" \
    --topk "$topk" --out-dir "$MDIR" > "$AR/man_${tag}.log" 2>&1 || { log "MAN FAIL $tag"; continue; }
  RR="$AR/out/reranked_${tag}.json"; mi=$((topk+1))
  if [ ! -s "$RR" ]; then
    python "$GR/vllm_rerank.py" --manifest "$MDIR/manifest.jsonl" --images-dir "$MDIR/images" \
      --model "$mhf" --tensor-parallel 2 --max-images "$mi" --max-model-len 16384 --out "$RR" \
      > "$AR/rr_${tag}.log" 2>&1 || { log "RR FAIL $tag"; continue; }
  fi
  python scripts/gpu_rerank/splice.py --stage1 "$S1DEV" --reranked "$RR" --only-qids data/queries_dev.json --out "$AR/out/dev_${tag}.json" >/dev/null 2>&1
  python scripts/gpu_rerank/splice.py --stage1 "$S1OFF" --reranked "$RR" --out "$AR/out/official_${tag}.json" >/dev/null 2>&1
  python "$AR/ar_eval.py" "$AR/out/dev_${tag}.json" data/queries_dev.json "$AR/out/metrics_${tag}.json" || log "EVAL FAIL $tag"
  python - "$tag" "$bb" "$alpha" "$bu" "$mt" "$topk" "$AR/out/metrics_${tag}.json" "$LEDGER" <<'PY'
import json,sys
tag,bb,alpha,bu,mt,topk,mpath,ledger=sys.argv[1:9]
m=json.load(open(mpath)); ov=m.get('_overall',{})
row={"tag":tag,"backbone":bb,"alpha":float(alpha),"brand_union":int(bu),"model":mt,"topk":int(topk),
     "dev_ndcg10":ov.get('ndcg@10'),"dev_map":ov.get('map'),
     "brand_swap_ndcg10":m.get('brand_swap',{}).get('ndcg@10'),
     "per_rule_ndcg10":{k:v.get('ndcg@10') for k,v in m.items()}}
open(ledger,'a').write(json.dumps(row)+"\n")
print("LEDGER",tag,"dev_ndcg10=",row['dev_ndcg10'])
PY
  log "DONE $tag"
done
log "ALL_DONE"
