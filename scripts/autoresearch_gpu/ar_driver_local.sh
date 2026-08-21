#!/usr/bin/env bash
# LOCAL proposer-driver: keeps ANTHROPIC_API_KEY on this machine. Each round: pull the box's
# ledger, run the LLM proposer locally to extend a local roadmap, push the roadmap to the box
# (the box consumer picks up new configs). After the last round, push a STOP sentinel.
set -uo pipefail
source /tmp/retr_box.env
H=root@$IP
SSHE="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p $PORT"
SCP="scp -o StrictHostKeyChecking=accept-new -P $PORT"
REPO=/home/alessandro/workspace/cameo-retrank
LROAD=/tmp/ar_roadmap_local.txt
LLED=/tmp/ar_ledger_local.jsonl
cp "$REPO/scripts/autoresearch_gpu/ar_roadmap2.txt" "$LROAD"
ROUNDS=${ROUNDS:-3}; GAP=${GAP:-1800}   # 30 min between proposer rounds

for R in $(seq 1 "$ROUNDS"); do
  sleep "$GAP"
  $SCP "$H:/workspace/autoresearch_gpu/ledger.jsonl" "$LLED" 2>/dev/null || { echo "round $R: ledger pull failed"; continue; }
  before=$(grep -vc '^#' "$LROAD")
  ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" python "$REPO/scripts/autoresearch_gpu/ar_propose.py" "$LLED" "$LROAD" 4 2>&1 | tail -1
  after=$(grep -vc '^#' "$LROAD")
  echo "round $R: roadmap $before -> $after configs"
  $SCP "$LROAD" "$H:/workspace/autoresearch_gpu/ar_roadmap2.txt" 2>/dev/null && echo "round $R: pushed roadmap"
done
# signal the box consumer to finish after it drains the queue
touch /tmp/AR_STOP; $SCP /tmp/AR_STOP "$H:/workspace/autoresearch_gpu/STOP" 2>/dev/null && echo "pushed STOP"
