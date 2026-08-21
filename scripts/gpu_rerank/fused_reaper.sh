#!/usr/bin/env bash
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"; source /tmp/retr_box.env; H=root@$IP
SSHE="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p $PORT"
LOCAL=/home/alessandro/workspace/cameo-retrank/data/runs/rerank_eval/fused; mkdir -p "$LOCAL"
DEADLINE=$(( $(date +%s) + 3600 ))
pull(){ scp -o StrictHostKeyChecking=accept-new -P "$PORT" "$H:/workspace/gpu_rerank/out/reranked_*fused*.json" "$LOCAL/" 2>/dev/null || true
  scp -o StrictHostKeyChecking=accept-new -P "$PORT" "$H:/workspace/gpu_rerank/master.log" "$LOCAL/fused_master.log" 2>/dev/null || true; }
reason=""
while true; do pull; $SSHE $H "grep -q ALL_DONE /workspace/gpu_rerank/master.log" 2>/dev/null && { reason=ALL_DONE; break; }
  [ "$(date +%s)" -ge "$DEADLINE" ] && { reason=DEADLINE; break; }; sleep 45; done
pull; echo "REAP $reason $(date)"; echo y | vastai destroy instance "$INSTANCE" 2>&1 | tr '\n' ' '; echo
sleep 5; vastai show instances 2>/dev/null | grep -q "$INSTANCE" && { echo retry; echo y | vastai destroy instance "$INSTANCE"; } || echo "CONFIRMED destroyed $INSTANCE"
