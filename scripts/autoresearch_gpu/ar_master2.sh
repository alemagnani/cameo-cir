#!/usr/bin/env bash
# GPU autoresearch v2: per-model args (InternVL3 patch-cap, Qwen-72B mem-tuning), top-K aware,
# seed roadmap + LLM-proposer rounds. Resumable (skip configs already in ledger). Reaches ALL_DONE.
set -uo pipefail
GR=/workspace/gpu_rerank; CAM=/workspace/cameo; AR=/workspace/autoresearch_gpu
export HF_HOME=/workspace/hf VLLM_WORKER_MULTIPROC_METHOD=spawn
log(){ echo "[$(date +%H:%M:%S)] AR2: $*"; }
mkdir -p "$AR/out" "$CAM/data/runs"
LEDGER="$AR/ledger.jsonl"; ROADMAP="$AR/ar_roadmap2.txt"
START=$(date +%s); SOFT_BUDGET=$((5*3600))   # stop proposing after ~5h; reaper hard-stops at 6h
MAX_ROUNDS=3

log "install deps"
pip install -q open_clip_torch pandas pyarrow opencv-python-headless 2>&1 | tail -1 || true
pip install -q -U vllm qwen-vl-utils 2>&1 | tail -1 || true
python -c "import torch;torch.zeros(1).cuda();print('CUDA_OK')" 2>&1 | tail -1 || log "CUDA FAIL"
until [ "$(find $CAM/data/raw/images -type f 2>/dev/null|wc -l)" -ge 4070 ]; do log "wait corpus"; sleep 15; done
cd "$CAM"

model_args(){ case "$1" in
  qwen72b) echo "--tensor-parallel 2 --gpu-mem-util 0.97 --max-model-len 8192 --enforce-eager";;
  internvl3) echo "--tensor-parallel 2 --max-model-len 16384 --mm-max-patches 2";;
  *) echo "--tensor-parallel 2 --max-model-len 16384";; esac; }

run_one(){
  local tag=$1 bb=$2 pre=$3 alpha=$4 bu=$5 mt=$6 mhf=$7 topk=$8
  if [ -s "$LEDGER" ] && grep -q "\"tag\": \"$tag\"" "$LEDGER"; then return 0; fi
  local bbt at; bbt=$(echo "$bb"|tr 'A-Z/' 'a-z_'); at=$(echo "$alpha"|tr -d '.')
  log "=== CONFIG $tag ==="
  for qp in "queries_dev:dev" "queries_official_cir:official"; do
    qf=${qp%%:*}; qtag=${qp##*:}; S1="data/runs/s1_${bbt}_a${at}_${qtag}.json"
    [ -s "$S1" ] || python scripts/exp004/run_openclip_cir.py --queries "data/$qf.json" --out "$S1" \
      --model "$bb" --pretrained "$pre" --fuse-name-alpha "$alpha" --topk 50 \
      > "$AR/enc_${bbt}_a${at}_${qtag}.log" 2>&1 || log "ENC FAIL $tag $qtag"
  done
  local S1DEV="data/runs/s1_${bbt}_a${at}_dev.json" S1OFF="data/runs/s1_${bbt}_a${at}_official.json"
  if [ "$bu" = "1" ]; then
    local BUDEV="data/runs/s1_${bbt}_a${at}_bu_dev.json" BUOFF="data/runs/s1_${bbt}_a${at}_bu_official.json"
    [ -s "$BUDEV" ] || python scripts/gpu_rerank/brand_union.py --run "$S1DEV" --queries data/queries_dev.json --out "$BUDEV" >/dev/null 2>&1
    [ -s "$BUOFF" ] || python scripts/gpu_rerank/brand_union.py --run "$S1OFF" --queries data/queries_official_cir.json --out "$BUOFF" >/dev/null 2>&1
    S1DEV="$BUDEV"; S1OFF="$BUOFF"
  fi
  local MDIR="$CAM/man_${bbt}_a${at}_bu${bu}_k${topk}"
  [ -s "$MDIR/manifest.jsonl" ] || python scripts/gpu_rerank/build_manifest.py \
    --pair "$S1OFF::data/queries_official_cir.json" --pair "$S1DEV::data/queries_dev.json" \
    --topk "$topk" --out-dir "$MDIR" > "$AR/man_${tag}.log" 2>&1 || { log "MAN FAIL $tag"; return 0; }
  local RR="$AR/out/reranked_${tag}.json" mi=$((topk+1))
  if [ ! -s "$RR" ]; then
    python "$GR/vllm_rerank.py" --manifest "$MDIR/manifest.jsonl" --images-dir "$MDIR/images" \
      --model "$mhf" --max-images "$mi" $(model_args "$mt") --out "$RR" \
      > "$AR/rr_${tag}.log" 2>&1 || { log "RR FAIL $tag"; return 0; }
  fi
  python scripts/gpu_rerank/splice.py --stage1 "$S1DEV" --reranked "$RR" --only-qids data/queries_dev.json --out "$AR/out/dev_${tag}.json" >/dev/null 2>&1
  python scripts/gpu_rerank/splice.py --stage1 "$S1OFF" --reranked "$RR" --out "$AR/out/official_${tag}.json" >/dev/null 2>&1
  python "$AR/ar_eval.py" "$AR/out/dev_${tag}.json" data/queries_dev.json "$AR/out/metrics_${tag}.json" || { log "EVAL FAIL $tag"; return 0; }
  python - "$tag" "$bb" "$alpha" "$bu" "$mt" "$topk" "$AR/out/metrics_${tag}.json" "$LEDGER" <<'PY'
import json,sys
tag,bb,alpha,bu,mt,topk,mpath,ledger=sys.argv[1:9]
m=json.load(open(mpath)); ov=m.get('_overall',{})
open(ledger,'a').write(json.dumps({"tag":tag,"backbone":bb,"alpha":float(alpha),"brand_union":int(bu),
  "model":mt,"topk":int(topk),"dev_ndcg10":ov.get('ndcg@10'),"dev_map":ov.get('map'),
  "brand_swap_ndcg10":m.get('brand_swap',{}).get('ndcg@10')})+"\n")
print("LEDGER",tag,"dev_ndcg10=",ov.get('ndcg@10'))
PY
  log "DONE $tag"
}

run_pending(){
  while IFS='|' read -r tag bb pre alpha bu mt mhf topk; do
    [ -z "${tag:-}" ] && continue; case "$tag" in \#*) continue;; esac
    run_one "$tag" "$bb" "$pre" "$alpha" "$bu" "$mt" "$mhf" "$topk"
  done < "$ROADMAP"
}

# Pure GPU consumer: run all roadmap configs, then poll for new ones pushed by the LOCAL
# proposer-driver (keeps the API key off this box). Exit on STOP sentinel or long idle.
log "=== CONSUMER LOOP START ==="
IDLE=0
while true; do
  before=$( [ -s "$LEDGER" ] && wc -l < "$LEDGER" || echo 0 )
  run_pending
  after=$( [ -s "$LEDGER" ] && wc -l < "$LEDGER" || echo 0 )
  if [ "$after" -gt "$before" ]; then IDLE=0; else IDLE=$((IDLE+1)); fi
  [ -f "$AR/STOP" ] && { log "STOP sentinel"; break; }
  [ $(( $(date +%s) - START )) -ge $SOFT_BUDGET ] && { log "soft budget"; break; }
  [ "$IDLE" -ge 40 ] && { log "idle timeout (no new configs)"; break; }
  sleep 60
done
log "ALL_DONE"
