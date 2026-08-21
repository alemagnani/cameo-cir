"""
Autoresearch 12-hour block daemon.

Workflow per block:
  1. Load roadmap.yaml + ledger.
  2. For each roadmap config not yet in ledger: run it (subject to budgets).
  3. When roadmap is exhausted (or budgets near limit), call the LLM proposer
     to generate 1-3 new configs based on what's worked. (Proposer is best-effort;
     budget-capped; failures fall back to "stop early".)
  4. At end-of-block, write reports/block_<timestamp>.md.

Budgets (defaults; override via CLI):
  --duration       43200 s  (12 hours)
  --max-experiments 200
  --max-api-usd    20.00
  --proposer-max-calls 3

Resilient: each experiment is in a try/except. Per-row flush of the ledger.
Safe to kill and resume.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.autoresearch.ledger import LEDGER_PATH_DEFAULT, completed_ids, load_rows
from scripts.autoresearch.run_one import run_one
from scripts.autoresearch.schema import experiment_id, merge_defaults, short_label

# Driver metric is dev.overall.ndcg@10 minus PARETO_LAMBDA × largest per-rule
# regression vs the best-so-far per-rule baseline. Penalizes "improvements" that
# crash a single rule (the EXP-003 failure mode).
PARETO_LAMBDA = 0.5
DRIVER_METRIC = "ndcg@10"

AR_DIR = Path(__file__).resolve().parent
ROADMAP_PATH = AR_DIR / "roadmap.yaml"
REPORTS_DIR = AR_DIR / "reports"

# Crude per-call cost model (rough order-of-magnitude estimates for budgeting,
# NOT for billing). Update when Anthropic pricing changes.
COST_PER_CALL_USD = {
    "haiku":  {"rerank_listwise": 0.005, "judge_text": 0.0005},
    "sonnet": {"rerank_listwise": 0.05,  "judge_text": 0.003},
    "opus":   {"rerank_listwise": 0.20,  "judge_text": 0.012},
}


def load_roadmap(path: Path = ROADMAP_PATH) -> list[dict]:
    raw = path.read_text()
    items = yaml.safe_load(raw) or []
    return [merge_defaults(it) for it in items]


def best_per_rule(rows: list[dict]) -> dict[str, float]:
    """Across all OK rows, the best dev NDCG@10 seen for each rule."""
    out: dict[str, float] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        by_rule = (r.get("metrics") or {}).get("dev", {}).get("by_rule", {})
        for rule, m in by_rule.items():
            v = m.get(DRIVER_METRIC)
            if v is None:
                continue
            if v > out.get(rule, -1.0):
                out[rule] = v
    return out


def driver_score(row: dict, per_rule_baseline: dict[str, float]) -> float | None:
    """Composite: overall NDCG@10 - λ × max(per-rule regression vs best baseline)."""
    dev = (row.get("metrics") or {}).get("dev")
    if not dev:
        return None
    overall = dev.get("overall", {}).get(DRIVER_METRIC)
    if overall is None:
        return None
    max_reg = 0.0
    for rule, m in dev.get("by_rule", {}).items():
        v = m.get(DRIVER_METRIC)
        b = per_rule_baseline.get(rule)
        if v is None or b is None:
            continue
        reg = max(0.0, b - v)
        if reg > max_reg:
            max_reg = reg
    return overall - PARETO_LAMBDA * max_reg


def estimate_cost(cfg: dict, n_queries_dev: int = 429, n_queries_official: int = 20) -> float:
    """Rough $ for the rerank + judge calls this experiment will make (before caching)."""
    cost = 0.0
    if cfg["rerank"]["enabled"]:
        m = cfg["rerank"]["model"]
        cost += COST_PER_CALL_USD.get(m, {}).get("rerank_listwise", 0.05) * n_queries_dev
        if cfg["holdout_judge"]:
            cost += COST_PER_CALL_USD.get(m, {}).get("rerank_listwise", 0.05) * n_queries_official
    if cfg["holdout_judge"]:
        m = cfg["judge_model"]
        cost += COST_PER_CALL_USD.get(m, {}).get("judge_text", 0.0005) * n_queries_official * 20
    return cost


def write_report(block_start: float, rows_this_block: list[dict], budgets_state: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = REPORTS_DIR / f"block_{ts}.md"

    all_rows = load_rows()
    ok_rows = [r for r in all_rows if r.get("status") == "ok" and r["metrics"].get("dev")]
    per_rule_baseline = best_per_rule(ok_rows)
    scored = [(driver_score(r, per_rule_baseline), r) for r in ok_rows]
    scored = [(s, r) for s, r in scored if s is not None]
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1] if scored else None
    best_score = scored[0][0] if scored else None

    block_ok = [r for r in rows_this_block if r.get("status") == "ok"]
    block_err = [r for r in rows_this_block if r.get("status") == "error"]

    lines = []
    lines.append(f"# Autoresearch block report — {ts}")
    lines.append("")
    lines.append(f"- block duration:  {(time.time()-block_start)/3600:.2f} h")
    lines.append(f"- experiments run: {len(rows_this_block)} (ok={len(block_ok)}, err={len(block_err)})")
    lines.append(f"- est. API spend:  ${budgets_state['spent_usd']:.2f} / ${budgets_state['max_usd']:.2f}")
    lines.append(f"- ledger total:    {len(all_rows)} rows")
    lines.append("")

    if best:
        m = best["metrics"]["dev"]["overall"]
        lines.append(f"## Best by Pareto-aware driver score so far  (λ={PARETO_LAMBDA})")
        lines.append(f"- id: `{best['id']}`  label: {best['label']}")
        lines.append(f"- driver_score = **{best_score:.4f}**  (overall - λ·max_per_rule_regression)")
        lines.append(f"- raw overall: NDCG@10 = {m.get('ndcg@10',0):.4f}, MAP = {m.get('map',0):.4f}, "
                     f"R@10 = {m.get('recall@10',0):.4f}, n={m.get('n','?')}")
        if best["metrics"].get("judge"):
            j = best["metrics"]["judge"]
            lines.append(f"- judge NDCG@10 = {j.get('ndcg@10',0):.4f}, MAP = {j.get('map',0):.4f}")
        lines.append("")

    if block_ok:
        lines.append("## This block — results (ordered by driver score)")
        lines.append("")
        lines.append("| id | driver | NDCG@10 | MAP | R@10 | n | label |")
        lines.append("|---|---|---|---|---|---|---|")
        scored_block = [(driver_score(r, per_rule_baseline), r) for r in block_ok
                        if driver_score(r, per_rule_baseline) is not None]
        scored_block.sort(key=lambda x: x[0], reverse=True)
        for sc, r in scored_block:
            m = r["metrics"]["dev"]["overall"]
            lines.append(f"| `{r['id']}` | {sc:.4f} | {m.get('ndcg@10',0):.4f} | {m.get('map',0):.4f} | "
                         f"{m.get('recall@10',0):.4f} | {m.get('n','?')} | {r['label']} |")
        lines.append("")

    if block_err:
        lines.append("## Errors this block")
        for r in block_err:
            lines.append(f"- `{r['id']}`  {r['label']}  — {r['error']}")
        lines.append("")

    # Per-rule best summary across all OK rows (useful when picking the next sweep)
    if per_rule_baseline:
        lines.append("## Best NDCG@10 per dev-rule (across all OK rows)")
        lines.append("")
        lines.append("| rule | best NDCG@10 |")
        lines.append("|---|---|")
        for rule, v in sorted(per_rule_baseline.items(), key=lambda x: -x[1]):
            lines.append(f"| `{rule}` | {v:.4f} |")
        lines.append("")

    lines.append("## Suggested next moves")
    if best and best["metrics"]["dev"]["overall"].get("ndcg@10", 0) > 0.029:
        lines.append("- Best > EXP-001 baseline (0.029). Worth a holdout-judge run on this config.")
    else:
        lines.append("- No config has beaten EXP-001 (NDCG@10=0.029). Consider GPU-side moves "
                     "(EXP-004 OTI, multilingual backbone) — see gpu_queue.yaml.")
    lines.append("- Consider extending roadmap.yaml with: brand-only fusion, structured-tag fusion, "
                 "M-CLIP backbone, query-rewriting via LLM.")

    path.write_text("\n".join(lines))
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=12 * 3600,
                        help="Block duration in seconds (default 12h).")
    parser.add_argument("--max-experiments", type=int, default=200)
    parser.add_argument("--max-api-usd", type=float, default=20.00)
    parser.add_argument("--ledger", default=str(LEDGER_PATH_DEFAULT))
    args = parser.parse_args()

    ledger_path = Path(args.ledger)
    block_start = time.time()
    deadline = block_start + args.duration
    budgets = {"spent_usd": 0.0, "max_usd": args.max_api_usd}

    rows_this_block: list[dict] = []
    print(f"[autoresearch] block start  duration={args.duration}s  "
          f"max_exp={args.max_experiments}  max_usd=${args.max_api_usd}", flush=True)

    iter_idx = 0
    while True:
        if time.time() >= deadline:
            print("[autoresearch] wall-clock budget reached — closing block.", flush=True)
            break
        if len(rows_this_block) >= args.max_experiments:
            print("[autoresearch] max-experiments reached — closing block.", flush=True)
            break
        if budgets["spent_usd"] >= budgets["max_usd"]:
            print("[autoresearch] API budget reached — closing block.", flush=True)
            break

        # Reload roadmap each iteration so external edits get picked up.
        try:
            roadmap = load_roadmap()
        except Exception as e:
            print(f"[autoresearch] roadmap load failed: {e}", flush=True)
            break

        done = completed_ids(ledger_path)
        pending = [cfg for cfg in roadmap if experiment_id(cfg) not in done]

        if not pending:
            print("[autoresearch] roadmap exhausted — closing block "
                  "(LLM proposer not wired yet in v0).", flush=True)
            break

        cfg = pending[0]
        est = estimate_cost(cfg)
        if budgets["spent_usd"] + est > budgets["max_usd"]:
            print(f"[autoresearch] next exp would exceed API budget "
                  f"(est ${est:.2f}); skipping and trying the next one.", flush=True)
            done.add(experiment_id(cfg))  # mark as "skip due to budget" in-memory only
            continue

        iter_idx += 1
        print(f"[autoresearch] iter {iter_idx}: {cfg.get('id','?')}  est=${est:.2f}", flush=True)
        try:
            row = run_one(cfg, ledger_path)
        except Exception as e:
            traceback.print_exc()
            row = {"id": experiment_id(cfg), "label": short_label(cfg),
                   "status": "error", "error": f"{type(e).__name__}: {e}",
                   "metrics": {"dev": None, "judge": None}}
        budgets["spent_usd"] += est if row.get("status") == "ok" else 0.0
        rows_this_block.append(row)

    report = write_report(block_start, rows_this_block, budgets)
    print(f"[autoresearch] report written → {report.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
