"""Self-contained dev eval for the GPU autoresearch loop (reuses src/eval/metrics).
Writes per-rule + overall NDCG@10/@20 + MAP as JSON.
Usage: python ar_eval.py <run.json> <queries.json> <out.json>"""
import json, sys
from collections import defaultdict
sys.path.insert(0, "/workspace/cameo")
from src.eval.metrics import evaluate

run = json.load(open(sys.argv[1]))
q = json.load(open(sys.argv[2]))
out_path = sys.argv[3]

by_rule = defaultdict(dict)
for qid, v in q.items():
    by_rule[v.get("rule", "_unknown")][qid] = v
by_rule["_overall"] = q

out = {}
for rule, sl in by_rule.items():
    qrels = {qid: {str(p): 2 for p in v["answer_product_id"]}
             for qid, v in sl.items() if v.get("answer_product_id")}
    sr = {qid: run[qid] for qid in qrels if qid in run}
    if sr:
        out[rule] = evaluate(qrels, sr, ks=[10, 20])
json.dump(out, open(out_path, "w"), indent=2)
ov = out.get("_overall", {})
print(f"OVERALL ndcg@10={ov.get('ndcg@10',0):.4f} map={ov.get('map',0):.4f}")
