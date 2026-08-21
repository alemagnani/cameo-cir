"""
Append-only JSONL ledger of experiment results.

Each row:
  {
    "id": "<sha1[:12]>",
    "label": "<human one-liner>",
    "config": {...},          # full config dict with defaults
    "metrics": {
      "dev": {
        "overall": {"ndcg@10": ..., "ndcg@5": ..., "map": ..., "recall@10": ..., ...},
        "by_rule": {
          "cheaper_same_brand": {...},
          ...
        }
      },
      "judge": null | {
        "ndcg@10": ..., "map": ..., ...
      }
    },
    "cost_usd": 0.0,           # API spend estimate for this run
    "wall_seconds": 0.0,
    "git_sha": "abc1234",
    "started_at": "2026-...",
    "finished_at": "2026-...",
    "status": "ok" | "error",
    "error": null | "...",
  }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

LEDGER_PATH_DEFAULT = Path(__file__).resolve().parent / "experiments.jsonl"


def load_rows(path: Path = LEDGER_PATH_DEFAULT) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def completed_ids(path: Path = LEDGER_PATH_DEFAULT) -> set[str]:
    """Configs we should NOT re-run: both successes (cache) and permanent errors."""
    return {r["id"] for r in load_rows(path) if r.get("status") in ("ok", "error")}


def append_row(row: dict, path: Path = LEDGER_PATH_DEFAULT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def iter_best_by_metric(path: Path = LEDGER_PATH_DEFAULT, metric: str = "ndcg@10") -> Iterator[dict]:
    """Yield rows sorted by dev.overall[metric] descending. Skips errored rows."""
    rows = [r for r in load_rows(path)
            if r.get("status") == "ok"
            and r.get("metrics", {}).get("dev", {}).get("overall", {}).get(metric) is not None]
    rows.sort(key=lambda r: r["metrics"]["dev"]["overall"][metric], reverse=True)
    return iter(rows)
