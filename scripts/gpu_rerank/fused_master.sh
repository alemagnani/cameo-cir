#!/usr/bin/env bash
set -uo pipefail
GR=/workspace/gpu_rerank
export HF_HOME=/workspace/hf VLLM_WORKER_MULTIPROC_METHOD=spawn
log(){ echo "[$(date +%H:%M:%S)] FUSE: $*"; }
log "install vllm"; pip install -q -U vllm qwen-vl-utils 2>&1 | tail -1 || true
python -c "import torch;torch.zeros(1).cuda();print('CUDA_OK')" 2>&1 | tail -1 || log "CUDA FAIL"
[ -s "$GR/out/reranked_qwen72b_fused30.json" ] && { log "[resume] done"; echo ALL_DONE; exit 0; }
MANIFEST=$GR/fused30/manifest.jsonl IMAGES=$GR/fused30/images MODELS_FILE=$GR/models_fused72.txt \
  bash $GR/run_all_models.sh > $GR/run_fused.log 2>&1 || true
log ALL_DONE
