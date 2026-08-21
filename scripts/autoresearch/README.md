# Autoresearch loop

A 12-hour autonomous experiment runner for CAMEO CIR.

## Quick start

```bash
# 1. (one-time) extend the dev set with Rule-5 LLM-generated queries
python scripts/build_dev_set_rule5.py --target 400 --workers 8
python scripts/autoresearch/merge_dev.py
# → data/queries_dev_full.json  (~829 queries)

# 2. launch a 12-hour block
PYTHONPATH=. nohup python scripts/autoresearch/loop.py \
    --duration 43200 --max-experiments 200 --max-api-usd 20 \
    > scripts/autoresearch/reports/loop.log 2>&1 &

# 3. ~12 hours later, read the report
cat scripts/autoresearch/reports/block_<latest>.md
```

## How it works

The loop is a daemon. Each iteration:
1. Reload `roadmap.yaml` (so external edits land mid-block).
2. Find the next config not yet in `experiments.jsonl`.
3. Run it via `run_one.py`: inference → optional rerank → in-process dev-set eval → optional holdout judge.
4. Append a row to `experiments.jsonl`.
5. Repeat until a budget cap is hit (wall-clock, max experiments, API spend) or the roadmap is exhausted.

At the end of the block, `reports/block_<timestamp>.md` is written with:
- summary stats (experiments run, errors, $ spent),
- best config so far by the **Pareto-aware driver score**,
- the block's per-experiment table,
- best NDCG@10 per dev-rule across all OK rows.

## Driver metric

```
driver_score = dev.overall.ndcg@10  -  λ · max_per_rule_regression
            (λ = 0.5; max_per_rule_regression = best_per_rule_so_far − this_per_rule)
```

This penalizes configs that boost the overall number while crashing one rule —
the failure mode EXP-003 hit (text fusion lifted brand_swap but tanked
claim_swap). Plain `dev.overall.ndcg@10` is also kept in the report for
transparency.

## Files

| File | Purpose |
|---|---|
| `schema.py` | Config schema + deterministic config hash → experiment id |
| `ledger.py` | Append-only `experiments.jsonl` ledger |
| `metrics_helper.py` | Per-rule + overall NDCG@k / Recall@k / MAP |
| `run_one.py` | Single-experiment driver (inference → eval → ledger row) |
| `roadmap.yaml` | Ordered list of seed configs the loop will try |
| `gpu_queue.yaml` | Experiments that need a GPU (loop ignores; user batches on Vast.ai) |
| `loop.py` | 12-hour block daemon |
| `merge_dev.py` | Merge `queries_dev.json` + `queries_dev_rule5.json` → `queries_dev_full.json` |
| `experiments.jsonl` | The ledger (created on first run) |
| `reports/` | Per-block markdown reports |

## Cost model

The loop estimates per-experiment API cost from the config (rerank model+topk,
judge enable+model) using a rough `COST_PER_CALL_USD` table in `loop.py`. The
budget is enforced before launching each experiment — if a run would push past
`--max-api-usd`, the loop tries the next config instead.

Defaults: $20/block, 200 experiments, 12 hours. Override on the CLI.

## GPU work

GPU-requiring experiments live in `gpu_queue.yaml`, not `roadmap.yaml`. The
CPU loop never picks them up. When the queue has ≥2 items, rent a Vast.ai
instance and run them in one batch (see `scripts/exp004/README.md` for the
recipe). Then add CPU-side roadmap entries that point at the produced artifacts
(e.g. `oti_tokens: data/checkpoints/oti_cameo_GPTNeo27B/`).

## Extending the loop

- **Add an experiment:** append a YAML doc to `roadmap.yaml`. The loop picks it up next iteration.
- **Add a new fusion kind:** `run_one.py` currently knows `none` and `name`.
  Add a `brand_only` / `structured` branch (and the corresponding flags in
  `run_searle.py`) when needed.
- **Change the driver:** edit `PARETO_LAMBDA` in `loop.py`. Past rows aren't
  re-ranked but the next block respects the new metric.
- **LLM proposer:** v0 stops when the roadmap is empty. A follow-up will add
  an LLM that proposes 1–3 new configs based on the ledger when this happens.

## Reading the ledger

```python
from scripts.autoresearch.ledger import load_rows
rows = load_rows()
# sort by composite driver score
from scripts.autoresearch.loop import best_per_rule, driver_score
b = best_per_rule(rows)
ranked = sorted([(driver_score(r, b), r) for r in rows if r["status"] == "ok"],
                key=lambda x: x[0], reverse=True)
for s, r in ranked[:10]:
    print(f"{s:.4f}  {r['id']}  {r['label']}")
```
