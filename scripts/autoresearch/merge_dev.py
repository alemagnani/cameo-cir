"""
Merge templated dev set + all Rule-5 passes → data/queries_dev_full.json.

Sources (merged in priority order, later passes can't override earlier):
  - data/queries_dev.json           (429 templated rules; built by build_dev_set.py)
  - data/queries_dev_rule5.json     (original Rule-5; DEV-R5-NNNN)
  - data/queries_dev_rule5_a.json   (pass A; DEV-R5A-NNNN)
  - data/queries_dev_rule5_b.json   (pass B; DEV-R5B-NNNN)
  - ... any data/queries_dev_rule5_*.json with distinct QID prefixes

Re-keys on QID collision (prefixes from --qid-prefix avoid this).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DEV = ROOT / "data" / "queries_dev.json"
OUT = ROOT / "data" / "queries_dev_full.json"


def main():
    base = json.loads(DEV.read_text()) if DEV.exists() else {}
    print(f"base templated: {len(base)}")
    merged = dict(base)

    # Sort Rule-5 files for deterministic merge order
    r5_files = sorted((ROOT / "data").glob("queries_dev_rule5*.json"))
    # Filter out progress and tmp files
    r5_files = [p for p in r5_files if not (p.name.endswith(".progress.json") or p.name.endswith(".tmp"))]

    for p in r5_files:
        try:
            r5 = json.loads(p.read_text())
        except Exception as e:
            print(f"  skip {p.name}: {e}")
            continue
        n_before = len(merged)
        for k, v in r5.items():
            if k in merged:
                k = f"DUP_{p.stem}_{k}"
            merged[k] = v
        print(f"  + {p.name}: {len(r5)}  (merged now {len(merged)}; added {len(merged)-n_before})")

    OUT.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    print(f"\nwrote {len(merged)} merged queries → {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
