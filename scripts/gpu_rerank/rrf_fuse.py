import json, sys
from collections import defaultdict
# rrf_fuse.py out.json k_rrf run1 run2 ...
out, krrf = sys.argv[1], int(sys.argv[2]); runs=[json.load(open(p)) for p in sys.argv[3:]]
qids=set().union(*[set(r) for r in runs])
fused={}
for qid in qids:
    score=defaultdict(float)
    for r in runs:
        for rank,pid in enumerate(r.get(qid,[])):
            score[str(pid)] += 1.0/(krrf+rank)
    fused[qid]=[p for p,_ in sorted(score.items(), key=lambda x:-x[1])]
json.dump(fused, open(out,'w'))
print(f"fused {len(fused)} queries -> {out}")
