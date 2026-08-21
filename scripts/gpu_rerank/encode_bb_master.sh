#!/usr/bin/env bash
# Encode-only backbone shootout: explicit strong models vs bigG (recall on dev-429). Resumable.
set -uo pipefail
CAM=/workspace/cameo; export HF_HOME=/workspace/hf
log(){ echo "[$(date +%H:%M:%S)] BB: $*"; }
pip install -q open_clip_torch pandas pyarrow opencv-python-headless 2>&1 | tail -1 || true
until [ "$(find $CAM/data/raw/images -type f 2>/dev/null|wc -l)" -ge 4070 ]; do log "wait corpus"; sleep 15; done
cd "$CAM"
# tag|model|pretrained
MODELS=(
 "siglip2so|ViT-SO400M-16-SigLIP2-384|webli"
 "siglip2so14|ViT-SO400M-14-SigLIP2-384|webli"
 "eva_e|EVA02-E-14|laion2b_s4b_b115k"
)
for spec in "${MODELS[@]}"; do
  IFS='|' read -r tag mdl pre <<< "$spec"
  for qp in "queries_dev:dev"; do
    qf=${qp%%:*}; qt=${qp##*:}; OUT="data/runs/bb_${tag}_${qt}.json"
    [ -s "$OUT" ] && { log "[skip] $OUT"; continue; }
    log "encode $tag ($mdl/$pre)"
    python scripts/exp004/run_openclip_cir.py --queries "data/$qf.json" --out "$OUT" \
      --model "$mdl" --pretrained "$pre" --topk 50 > "bb_${tag}.log" 2>&1 || log "FAIL $tag (bad model id?)"
  done
done
log "=== recall@K (dev-429) ==="
python - <<'PY'
import json,glob,os
q=json.load(open('/workspace/cameo/data/queries_dev.json'))
gt={k:set(map(str,v.get('answer_product_id',[]))) for k,v in q.items() if v.get('answer_product_id')}
for f in sorted(glob.glob('/workspace/cameo/data/runs/bb_*_dev.json')):
    run=json.load(open(f)); tag=os.path.basename(f)[3:-9]
    def r(k):
        n=h=0
        for qid,g in gt.items():
            if qid in run: n+=1; h+= 1 if set(map(str,run[qid][:k]))&g else 0
        return h/max(n,1)
    print(f"{tag:14s} R@20={r(20):.4f} R@50={r(50):.4f}  (bigG: 0.359/0.436)")
PY
log "ALL_DONE"
