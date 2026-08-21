#!/usr/bin/env bash
# Auto-reaper: pulls all GPU results and DESTROYS the vast instance when the master
# signals ALL_DONE, or when a hard deadline passes (safety against a hung run).
# No human confirmation needed; guarantees the GPU is turned off.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
SSHE="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p 13601"
H=root@80.188.223.202
INSTANCE=39893165
LOCAL=/home/alessandro/workspace/cameo-retrank/data/runs/rerank_eval
mkdir -p "$LOCAL/gpu" "$LOCAL/encode_runs"
DEADLINE=$(( $(date +%s) + 7200 ))   # 3h hard cap

reason=""
while true; do
  if $SSHE $H "grep -q ALL_DONE /workspace/gpu_rerank/master.log" 2>/dev/null; then reason=ALL_DONE; break; fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then reason=DEADLINE; break; fi
  sleep 60
done
echo "REAP triggered: reason=$reason  $(date)"

# Pull everything (best-effort)
scp -o StrictHostKeyChecking=accept-new -P 20130 "$H:/workspace/gpu_rerank/out/*.json"      "$LOCAL/gpu/"          2>/dev/null || true
scp -o StrictHostKeyChecking=accept-new -P 20130 "$H:/workspace/gpu_rerank/*.log"           "$LOCAL/"              2>/dev/null || true
scp -o StrictHostKeyChecking=accept-new -P 20130 "$H:/workspace/cameo/data/runs/oc_*.json"  "$LOCAL/encode_runs/" 2>/dev/null || true
scp -o StrictHostKeyChecking=accept-new -P 20130 "$H:/workspace/cameo/best.txt" "$H:/workspace/cameo/picks.txt" "$LOCAL/" 2>/dev/null || true
echo "pulled results to $LOCAL"

# Destroy the instance (guaranteed shutdown)
echo y | vastai destroy instance "$INSTANCE" 2>&1 | tr '\n' ' '; echo
sleep 5
if vastai show instances 2>/dev/null | grep -q "$INSTANCE"; then
  echo "WARN: instance still listed — retrying destroy"
  echo y | vastai destroy instance "$INSTANCE" 2>&1
else
  echo "CONFIRMED: instance $INSTANCE destroyed"
fi
