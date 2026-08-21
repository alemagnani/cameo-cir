#!/usr/bin/env bash
# Autoresearch continuous puller + reaper. Pulls ledger + official runs + logs continuously;
# destroys box on ALL_DONE or 6h hard cap.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
source /tmp/retr_box.env
H=root@$IP
SSHE="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p $PORT"
LOCAL=/home/alessandro/workspace/cameo-retrank/data/runs/autoresearch_gpu
mkdir -p "$LOCAL/out"
DEADLINE=$(( $(date +%s) + 21600 ))   # 6h cap
pull(){
  scp -o StrictHostKeyChecking=accept-new -P "$PORT" "$H:/workspace/autoresearch_gpu/ledger.jsonl" "$LOCAL/" 2>/dev/null || true
  scp -o StrictHostKeyChecking=accept-new -P "$PORT" "$H:/workspace/autoresearch_gpu/out/official_*.json" "$LOCAL/out/" 2>/dev/null || true
  scp -o StrictHostKeyChecking=accept-new -P "$PORT" "$H:/workspace/autoresearch_gpu/out/metrics_*.json" "$LOCAL/out/" 2>/dev/null || true
  scp -o StrictHostKeyChecking=accept-new -P "$PORT" "$H:/workspace/autoresearch_gpu/master.log" "$LOCAL/ar_master.log" 2>/dev/null || true
}
reason=""
while true; do
  pull
  $SSHE $H "grep -q ALL_DONE /workspace/autoresearch_gpu/master.log" 2>/dev/null && { reason=ALL_DONE; break; }
  [ "$(date +%s)" -ge "$DEADLINE" ] && { reason=DEADLINE; break; }
  sleep 60
done
pull
echo "REAP reason=$reason $(date)"
echo y | vastai destroy instance "$INSTANCE" 2>&1 | tr '\n' ' '; echo
sleep 5
vastai show instances 2>/dev/null | grep -q "$INSTANCE" && { echo retry; echo y | vastai destroy instance "$INSTANCE"; } || echo "CONFIRMED destroyed $INSTANCE"
