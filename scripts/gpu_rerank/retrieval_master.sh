#!/usr/bin/env bash
# Focused retrieval session master — runs unattended on the box:
#   B  ViT-H top-50 rerank (Qwen-32B + InternVL3-38B)
#   C  encode corpus with SigLIP-2 / ViT-bigG / EVA-CLIP x {plain, name-fusion 0.2} -> top-50
#   D  auto-pick highest recall@50 backbone, build top-50 manifest, rerank with Qwen-32B
# No 100B models, no pass A. Non-fatal steps; always reaches ALL_DONE.
set -uo pipefail
GR=/workspace/gpu_rerank
CAM=/workspace/cameo
export HF_HOME=/workspace/hf VLLM_WORKER_MULTIPROC_METHOD=spawn
log(){ echo "[$(date +%H:%M:%S)] MASTER: $*"; }

log "PASS B start (ViT-H top-50 rerank)"
MANIFEST=$GR/vith50/manifest.jsonl IMAGES=$GR/vith50/images MODELS_FILE=$GR/models_vith.txt \
  bash "$GR/run_all_models.sh" > "$GR/run_vith.log" 2>&1 || true
log "PASS B done"

log "PASS C start (backbone encodes); installing open_clip"
pip install -q open_clip_torch pandas pyarrow 2>&1 | tail -1 || true
until [ "$(find $CAM/data/raw/images -type f 2>/dev/null | wc -l)" -ge 4070 ]; do log "waiting for corpus"; sleep 15; done
python - > "$CAM/picks.txt" <<'PY'
import open_clip
pt = open_clip.list_pretrained()
def pick(*subs):
    for m,p in pt:
        s=(m+":"+p).lower()
        if all(x.lower() in s for x in subs): return f"{m}|{p}"
    return "|"
print("siglip2|"+(pick('siglip2','384') or pick('siglip2') or "|"))
print("bigg|"+(pick('bigg-14','laion2b') or pick('bigg') or "|"))
print("eva02l|"+(pick('eva02-l-14') or pick('eva02-l') or "|"))
PY
log "resolved backbones:"; cat "$CAM/picks.txt"
cd "$CAM"
while IFS='|' read -r tag model pre; do
  [ -z "${model:-}" ] && { log "no match for $tag, skip"; continue; }
  for av in "0:" "0.2:f2"; do
    a=${av%%:*}; asfx=${av##*:}
    for qpair in "queries_dev:dev" "queries_official_cir:official"; do
      qf=${qpair%%:*}; qtag=${qpair##*:}
      log "encode ${tag}${asfx} ($model / $pre, fuse=$a) on $qtag"
      python scripts/exp004/run_openclip_cir.py --queries "data/$qf.json" \
        --out "data/runs/oc_${tag}${asfx}_${qtag}.json" --model "$model" --pretrained "$pre" \
        --fuse-name-alpha "$a" --topk 50 > "encode_${tag}${asfx}_${qtag}.log" 2>&1 \
        || log "ENCODE FAIL ${tag}${asfx} $qtag"
    done
  done
done < "$CAM/picks.txt"
log "PASS C done"

log "recall@50 on dev (pick best)"
python - > "$CAM/best.txt" <<'PY'
import json, glob, os
q = json.load(open('/workspace/cameo/data/queries_dev.json'))
gt = {k:set(map(str,v.get('answer_product_id',[]))) for k,v in q.items()}
def recall(run,k=50):
    n=hit=0
    for qid,g in gt.items():
        if not g or qid not in run: continue
        n+=1; hit+= 1 if (set(map(str,run[qid][:k])) & g) else 0
    return hit/max(n,1)
best=None
for f in sorted(glob.glob('/workspace/cameo/data/runs/oc_*_dev.json')):
    tag=os.path.basename(f)[3:-9]
    r=recall(json.load(open(f)))
    print(f"{tag}\tR@50={r:.4f}")
    if best is None or r>best[1]: best=(tag,r)
print("BEST "+(best[0] if best else "none"))
PY
cat "$CAM/best.txt"
BEST=$(awk '/^BEST/{print $2}' "$CAM/best.txt")
log "best backbone = $BEST"

if [ -n "${BEST:-}" ] && [ "$BEST" != "none" ]; then
  log "PASS D start (manifest from $BEST, rerank Qwen-32B)"
  python "$CAM/scripts/gpu_rerank/build_manifest.py" \
    --pair "$CAM/data/runs/oc_${BEST}_official.json::$CAM/data/queries_official_cir.json" \
    --pair "$CAM/data/runs/oc_${BEST}_dev.json::$CAM/data/queries_dev.json" \
    --topk 50 --out-dir "$CAM/best50" > "$CAM/build_best.log" 2>&1 || log "BUILD FAIL"
  if [ -f "$CAM/best50/manifest.jsonl" ]; then
    MANIFEST=$CAM/best50/manifest.jsonl IMAGES=$CAM/best50/images MODELS_FILE=$GR/models_best.txt \
      bash "$GR/run_all_models.sh" > "$GR/run_best.log" 2>&1 || true
  fi
  echo "BEST_BACKBONE=$BEST" > "$GR/out/best_backbone.txt"
  log "PASS D done"
fi
log "ALL_DONE"
