"""LLM proposer for the autoresearch loop. Reads ledger results, asks Claude to propose
new configs (within the allowed knob space) that should improve dev NDCG@10, validates +
dedups against what's already run/queued, appends valid full config lines to the roadmap.
Runs on the box (needs ANTHROPIC_API_KEY). Usage: python ar_propose.py <ledger> <roadmap> <n>"""
import json, os, re, sys

LEDGER, ROADMAP = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 4

BACKBONES = {"vith": ("ViT-H-14", "laion2b_s32b_b79k"),
             "bigg": ("ViT-bigG-14", "laion2b_s39b_b160k")}
MODELS = {"qwen32b": "Qwen/Qwen2.5-VL-32B-Instruct",
          "qwen72b": "Qwen/Qwen2.5-VL-72B-Instruct"}  # internvl3 excluded: fails to run under vLLM here
ALPHAS = {"0", "0.2", "0.4"}
BUS = {"0", "1"}
TOPKS = {"10", "20", "30"}

def tag_of(bb, a, bu, mt, k):
    return f"{bb}_a{a.replace('.','')}_bu{bu}_{mt}_k{k}"

existing = set()
results = []
if os.path.exists(LEDGER):
    for l in open(LEDGER):
        try:
            r = json.loads(l); results.append(r)
        except Exception:
            pass
for l in open(ROADMAP):
    if l.strip() and not l.startswith("#"):
        existing.add(l.split("|")[0].strip())

# results summary for the prompt
summary = "\n".join(
    f"  {r['tag']}: dev_ndcg10={r.get('dev_ndcg10'):.4f} brand@10={r.get('brand_swap_ndcg10') or 0:.4f}"
    for r in sorted(results, key=lambda x: x.get('dev_ndcg10') or 0, reverse=True)) or "  (none yet)"

prompt = f"""You are tuning a composed-image-retrieval pipeline. Knobs:
- backbone: vith | bigg
- alpha (name-text fusion): 0 | 0.2 | 0.4
- brand_union: 0 | 1
- model (reranker): qwen32b | qwen72b
- topk (rerank depth): 10 | 20 | 30

Results so far (dev NDCG@10, higher=better):
{summary}

Propose {N} NEW configs (not already run) likely to maximize dev_ndcg10, mixing exploitation of
the best region with some exploration. Reply ONLY with a JSON list of objects with keys
backbone, alpha, brand_union, model, topk. No prose."""

import anthropic
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
msg = client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                             messages=[{"role": "user", "content": prompt}])
txt = msg.content[0].text
m = re.search(r"\[.*\]", txt, re.S)
proposals = json.loads(m.group(0)) if m else []

added = 0
with open(ROADMAP, "a") as f:
    for p in proposals:
        bb = str(p.get("backbone")); a = str(p.get("alpha")); bu = str(p.get("brand_union"))
        mt = str(p.get("model")); k = str(p.get("topk"))
        if bb not in BACKBONES or a not in ALPHAS or bu not in BUS or mt not in MODELS or k not in TOPKS:
            continue
        tag = tag_of(bb, a, bu, mt, k)
        if tag in existing:
            continue
        existing.add(tag)
        mdl, pre = BACKBONES[bb]
        f.write(f"{tag}|{mdl}|{pre}|{a}|{bu}|{mt}|{MODELS[mt]}|{k}\n")
        added += 1
print(f"PROPOSER added {added} new configs")
