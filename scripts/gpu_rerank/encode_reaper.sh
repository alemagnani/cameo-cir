#!/usr/bin/env bash
# Continuous puller + guaranteed reaper for the encode-only session.
# Pulls every result file as it appears (incremental save), then destroys the box
# on ALL_DONE or a 60-min hard cap. Reads box coords from /tmp/retr_box.env.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source /tmp/retr_box.env   # sets IP, PORT, INSTANCE
H=root@$IP
SSHE="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p $PORT"
LOCAL=/home/alessandro/workspace/cameo-retrank/data/runs/rerank_eval/encode_runs
mkdir -p "$LOCAL"
DEADLINE=$(( $(date +%s) + 3600 ))   # 60-min cap

pull(){
  scp -o StrictHostKeyChecking=accept-new -P "$PORT" "$H:/workspace/cameo/data/runs/oc_*.json" "$LOCAL/" 2>/dev/null; scp -o StrictHostKeyChecking=accept-new -P "$PORT" "$H:/workspace/cameo/data/runs/bb_*.json" "$LOCAL/" 2>/dev/null || true
  scp -o StrictHostKeyChecking=accept-new -P "$PORT" "$H:/workspace/cameo/best.txt" "$H:/workspace/cameo/picks.txt" "$LOCAL/" 2>/dev/null || true
  scp -o StrictHostKeyChecking=accept-new -P "$PORT" "$H:/workspace/gpu_rerank/master.log" "$LOCAL/encode_master.log" 2>/dev/null || true
}

reason=""
while true; do
  pull
  if $SSHE $H "grep -q ALL_DONE /workspace/gpu_rerank/master.log" 2>/dev/null; then reason=ALL_DONE; break; fi
  [ "$(date +%s)" -ge "$DEADLINE" ] && { reason=DEADLINE; break; }
  sleep 45
done
pull
echo "REAP reason=$reason $(date)"
echo y | vastai destroy instance "$INSTANCE" 2>&1 | tr '\n' ' '; echo
sleep 5
if vastai show instances 2>/dev/null | grep -q "$INSTANCE"; then
  echo "retrying destroy"; echo y | vastai destroy instance "$INSTANCE" 2>&1
else
  echo "CONFIRMED destroyed $INSTANCE"
fi
