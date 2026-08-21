#!/usr/bin/env bash
# Step-2 continuous puller + reaper. Saves Stage-1 runs (expensive 2083 encode) AND
# rerank output continuously, then destroys on ALL_DONE or 90-min cap.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source /tmp/retr_box.env
H=root@$IP
SSHE="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p $PORT"
LOCAL=/home/alessandro/workspace/cameo-retrank/data/runs/rerank_eval/step2
mkdir -p "$LOCAL"
DEADLINE=$(( $(date +%s) + 5400 ))   # 90-min cap
pull(){
  scp -o StrictHostKeyChecking=accept-new -P "$PORT" "$H:/workspace/cameo/data/runs/oc_biggf2_*.json" "$LOCAL/" 2>/dev/null || true
  scp -o StrictHostKeyChecking=accept-new -P "$PORT" "$H:/workspace/gpu_rerank/out/reranked_*step2*.json" "$LOCAL/" 2>/dev/null || true
  scp -o StrictHostKeyChecking=accept-new -P "$PORT" "$H:/workspace/gpu_rerank/master.log" "$LOCAL/step2_master.log" 2>/dev/null || true
}
reason=""
while true; do
  pull
  $SSHE $H "grep -q ALL_DONE /workspace/gpu_rerank/master.log" 2>/dev/null && { reason=ALL_DONE; break; }
  [ "$(date +%s)" -ge "$DEADLINE" ] && { reason=DEADLINE; break; }
  sleep 45
done
pull
echo "REAP reason=$reason $(date)"
echo y | vastai destroy instance "$INSTANCE" 2>&1 | tr '\n' ' '; echo
sleep 5
vastai show instances 2>/dev/null | grep -q "$INSTANCE" && { echo retry; echo y | vastai destroy instance "$INSTANCE"; } || echo "CONFIRMED destroyed $INSTANCE"
