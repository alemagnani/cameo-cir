#!/usr/bin/env bash
# STEP 2 — broad eval of the FROZEN pipeline, fully resumable.
#   Stage-1: ViT-bigG-14 + name-fusion(0.2) over full-dev (2083) + official (20)
#   Stage-2: local VLM rerank (Qwen-32B) of top-20
# Produces Stage-1 runs + reranked runs; final NDCG/MAP eval is done LOCALLY after pull
# (exact ground truth for dev = free; official = cached Sonnet judge).
# Every output is skip-if-exists, so a crash/relaunch resumes where it left off.
set -uo pipefail
GR=/workspace/gpu_rerank
CAM=/workspace/cameo
export HF_HOME=/workspace/hf VLLM_WORKER_MULTIPROC_METHOD=spawn
log(){ echo "[$(date +%H:%M:%S)] STEP2: $*"; }

log "install deps (open_clip + vllm)"
pip install -q open_clip_torch pandas pyarrow opencv-python-headless 2>&1 | tail -1 || true
pip install -q -U vllm qwen-vl-utils 2>&1 | tail -1 || true
python -c "import torch;x=torch.zeros(1).cuda();print('CUDA_OK')" 2>&1 | tail -1 || log "CUDA FAIL"
until [ "$(find $CAM/data/raw/images -type f 2>/dev/null | wc -l)" -ge 4070 ]; do log "waiting for corpus"; sleep 15; done

MODEL="ViT-bigG-14"; PRE="laion2b_s39b_b160k"; A=0.2
cd "$CAM"
# ---- Stage-1 encode (skip-existing) ----
for qpair in "queries_dev_full:devfull" "queries_official_cir:official"; do
  qf=${qpair%%:*}; qtag=${qpair##*:}
  OUT="data/runs/oc_biggf2_${qtag}.json"
  if [ -s "$OUT" ]; then log "[resume] $OUT exists"; continue; fi
  log "Stage-1 encode bigG+fusion on $qtag"
  python scripts/exp004/run_openclip_cir.py --queries "data/$qf.json" --out "$OUT" \
    --model "$MODEL" --pretrained "$PRE" --fuse-name-alpha "$A" --topk 50 \
    > "step2_encode_${qtag}.log" 2>&1 || log "ENCODE FAIL $qtag"
done

# ---- brand-union augmentation (symbolic, free) of Stage-1 (skip-existing) ----
for qpair in "queries_dev_full:devfull" "queries_official_cir:official"; do
  qf=${qpair%%:*}; qtag=${qpair##*:}
  if [ ! -s "data/runs/oc_biggf2bu_${qtag}.json" ]; then
    log "brand-union augment $qtag"
    python "$CAM/scripts/gpu_rerank/brand_union.py" --run "data/runs/oc_biggf2_${qtag}.json" \
      --queries "data/$qf.json" --out "data/runs/oc_biggf2bu_${qtag}.json" 2>&1 || log "BRANDUNION FAIL $qtag"
  fi
done

# ---- build top-20 manifest from brand-union Stage-1 (skip-existing) ----
if [ ! -s "$CAM/step2_manifest/manifest.jsonl" ]; then
  log "build top-20 manifest"
  python "$CAM/scripts/gpu_rerank/build_manifest.py" \
    --pair "$CAM/data/runs/oc_biggf2bu_official.json::$CAM/data/queries_official_cir.json" \
    --pair "$CAM/data/runs/oc_biggf2bu_devfull.json::$CAM/data/queries_dev_full.json" \
    --topk 20 --out-dir "$CAM/step2_manifest" > "$CAM/step2_build.log" 2>&1 || log "BUILD FAIL"
fi

# ---- Stage-2 rerank (skip-existing inside run_all_models) ----
if [ ! -s "$GR/out/reranked_qwen32b_step2.json" ]; then
  log "Stage-2 rerank (Qwen-32B top-20, $(wc -l < $CAM/step2_manifest/manifest.jsonl) jobs)"
  MANIFEST=$CAM/step2_manifest/manifest.jsonl IMAGES=$CAM/step2_manifest/images \
    MODELS_FILE=$GR/models_step2.txt bash "$GR/run_all_models.sh" > "$GR/run_step2.log" 2>&1 || true
fi
log "ALL_DONE"
