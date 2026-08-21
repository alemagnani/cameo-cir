#!/usr/bin/env bash
set -uo pipefail
GR=/workspace/gpu_rerank
export HF_HOME=/workspace/hf VLLM_WORKER_MULTIPROC_METHOD=spawn
log(){ echo "[$(date +%H:%M:%S)] RR: $*"; }
log "install vllm"
pip install -q -U vllm qwen-vl-utils 2>&1 | tail -1 || true
python -c "import torch;x=torch.zeros(1).cuda();print('CUDA_OK')" 2>&1 | tail -1 || log "CUDA FAIL"
[ -s "$GR/out/reranked_qwen32b_bigg20.json" ] && { log "[resume] output exists, skip"; echo ALL_DONE; exit 0; }
log "rerank bigg20 top-20 (Qwen-32B)"
MANIFEST=$GR/bigg20/manifest.jsonl IMAGES=$GR/bigg20/images MODELS_FILE=$GR/models_bigg20.txt \
  bash $GR/run_all_models.sh > $GR/run_bigg20.log 2>&1 || true
log "ALL_DONE"
