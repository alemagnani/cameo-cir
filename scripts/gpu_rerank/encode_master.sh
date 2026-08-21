#!/usr/bin/env bash
# Minimal, resumable, encode-ONLY retrieval comparison (no vLLM, no reranking).
# Encodes the corpus with SigLIP-2 / ViT-bigG / EVA-CLIP x {plain, name-fusion}, writes
# top-50 runs (skipping any that already exist), then prints recall@K vs ground truth.
# Each (backbone x split) run is an independent checkpoint; a continuous puller saves them.
set -uo pipefail
CAM=/workspace/cameo
export HF_HOME=/workspace/hf
log(){ echo "[$(date +%H:%M:%S)] ENC: $*"; }

log "installing deps (open_clip only)"
pip install -q open_clip_torch pandas pyarrow opencv-python-headless 2>&1 | tail -1 || true
until [ "$(find $CAM/data/raw/images -type f 2>/dev/null | wc -l)" -ge 4070 ]; do log "waiting for corpus"; sleep 15; done
log "corpus present"

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
      OUT="data/runs/oc_${tag}${asfx}_${qtag}.json"
      if [ -s "$OUT" ]; then log "[resume] $OUT exists, skip"; continue; fi
      log "encode ${tag}${asfx} ($model / $pre, fuse=$a) on $qtag"
      python scripts/exp004/run_openclip_cir.py --queries "data/$qf.json" \
        --out "$OUT" --model "$model" --pretrained "$pre" --fuse-name-alpha "$a" \
        --topk 50 > "encode_${tag}${asfx}_${qtag}.log" 2>&1 || log "ENCODE FAIL ${tag}${asfx} $qtag"
    done
  done
done < "$CAM/picks.txt"
log "encodes done; recall@50 on dev:"

python - > "$CAM/best.txt" <<'PY'
import json, glob, os
q = json.load(open('/workspace/cameo/data/queries_dev.json'))
gt = {k:set(map(str,v.get('answer_product_id',[]))) for k,v in q.items()}
def recall(run,k):
    n=hit=0
    for qid,g in gt.items():
        if not g or qid not in run: continue
        n+=1; hit+= 1 if (set(map(str,run[qid][:k])) & g) else 0
    return hit/max(n,1)
best=None
for f in sorted(glob.glob('/workspace/cameo/data/runs/oc_*_dev.json')):
    tag=os.path.basename(f)[3:-9]; run=json.load(open(f))
    print(f"{tag}\tR@10={recall(run,10):.4f}\tR@20={recall(run,20):.4f}\tR@50={recall(run,50):.4f}")
    r=recall(run,50)
    if best is None or r>best[1]: best=(tag,r)
print("BEST "+(best[0] if best else "none"))
PY
cat "$CAM/best.txt"
log "ALL_DONE"
