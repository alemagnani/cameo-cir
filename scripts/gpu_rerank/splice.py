"""
Phase C helper — splice a reranked top-K head into the Stage-1 run, write a full run.

Input reranked file is {qid: [reordered top-k pids]} (vLLM output) OR a Haiku-style
cache {qid||hash: [indices]} (use --from-cache with --stage1 to resolve indices->pids).
Output is {qid: [reordered head ...] + [stage-1 tail]} ready for eval_by_rule / evaluate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def cand_hash(ids): return hashlib.sha1("|".join(map(str, ids)).encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True, help="Stage-1 run json {qid:[pids]}")
    ap.add_argument("--reranked", required=True, help="reranked heads json, or Haiku cache if --from-cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--from-cache", action="store_true",
                    help="treat --reranked as Haiku cache {qid||hash:[indices]}")
    ap.add_argument("--only-qids", default=None, help="optional queries json; restrict to its qids")
    args = ap.parse_args()

    stage1 = json.loads(Path(args.stage1).read_text())
    rr = json.loads(Path(args.reranked).read_text())
    only = set(json.loads(Path(args.only_qids).read_text())) if args.only_qids else None

    out = {}
    n = 0
    for qid, ranked in stage1.items():
        if only is not None and qid not in only:
            continue
        head = ranked[:args.topk]
        tail = ranked[args.topk:]
        if args.from_cache:
            key = f"{qid}||{cand_hash([str(p) for p in head])}"
            if key not in rr:
                out[qid] = ranked
                continue
            order = rr[key]
            new_head = [head[i] for i in order if i < len(head)]
        else:
            if qid not in rr:
                out[qid] = ranked
                continue
            new_head = [str(p) for p in rr[qid]]
        # ensure completeness
        seen = set(new_head)
        new_head += [p for p in head if p not in seen]
        out[qid] = new_head + tail
        n += 1

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"spliced {n} reranked / {len(out)} total -> {args.out}")


if __name__ == "__main__":
    main()
