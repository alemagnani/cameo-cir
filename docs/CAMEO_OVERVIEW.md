# CAMEO-CIR — Full Project Overview (for understanding)
*Compiled 2026-06-07. Companion to `docs/PROGRESS_SUMMARY.md` (running log).*

---

# PAGE 1 — Problem, Goal, Deadlines & Submission

## Problem statement
**Composed Image Retrieval (CIR)** for the **NTCIR-19 CAMEO** subtask.
Given a **reference product image** + a **natural-language modification** ("similar but cheaper",
"same format from brand X", "for sensitive skin"), return a **ranked list** of matching products
from a fixed corpus of **4,075 face-cleanser products** (Vietnamese e-commerce, ViEcom-Rec).

- **Zero-shot**: no training labels; test relevance judgments are **withheld until after submission**.
- **Tiny test set** (20 queries) → no room to fit a curve; robustness > leaderboard-chasing.
- **Metrics:** NDCG@K and MAP (TREC-style human judgments by organizers).

## Goal
Build the best zero-shot CIR pipeline (retrieval → rerank), submit up to 5 runs on the 20
official CIR queries, and write the NTCIR-19 workshop paper.

## ⏰ Deadlines (NTCIR-19 general schedule — **confirm exact CAMEO dates** on the task page)
| Milestone | Date (approx.) |
|---|---|
| Task registration | **~July 10, 2026** |
| Dry run window | Jan–Jun 2026 |
| **Formal run (submission) window** | **Mar–Jul 2026** (closes ~July) |
| Evaluation results returned | ~Aug 1, 2026 |
| Participant paper (draft) due | ~Sep 1, 2026 |
| Conference | **Dec 2026, Tokyo** |
*Today is 2026-06-07 → we are inside the formal-run window; ~weeks to submission close, ~3 months to paper.*
Sources: [NTCIR-19 tasks](https://research.nii.ac.jp/ntcir/ntcir-19/tasks.html), [how-to](https://research.nii.ac.jp/ntcir/ntcir-19/howto.html). Verify CAMEO specifics at the task site (cameo-task.com / geniie-lab.github.io/ntcir).

## 📤 Submission requirements (exact)
- **Compete on:** the **20 official CIR queries** (`data/queries_official_cir.json`). The full official
  file `data/queries_official.json` has **30** = 20 CIR + 10 QA (we don't do QA).
- **Format:** follows `data/raw/sample_submission.json` →
  `{ "CIR-0": { "image_path", "product_id", "question", "answer": [ranked product_ids] }, ... }`.
- **How:** **email to `allie.tran@dcu.ie`**, filenames **`retrank_CIR_Run1.json` … `Run5.json`**, **max 5 runs**.
- **Plus:** a **workshop paper** for the NTCIR-19 proceedings.
- We score ourselves with a Claude judge (proxy); **organizers do the real scoring** post-submission.
- Our current submission artifact: `data/submission/submission_bigg_qwen32b.json` (rename to `retrank_CIR_Run1.json` when ready).

---

# PAGE 2 — Data

## Provided by organizers
| Item | Size | Notes |
|---|---|---|
| **Corpus** | **4,075 products** | `data/raw/products__test.parquet`; fields: product_id, brand, price, product_name, translated_name, translated_description, image_path… Images ~1.9 GB (`data/raw/images/`). |
| **Official queries** | **30** (`queries_official.json`) | = **20 CIR** (`queries_official_cir.json`) + 10 QA. Fields: image_path, product_id, question (VN), translated_question (EN), translated_question_rewritten. **No answers** (withheld). |
| Sample submission | 10-entry template | `data/raw/sample_submission.json` — format reference only. |
| Reviews | — | `reviews__test.parquet` — product reviews, currently unused. |

## Created by us (synthetic dev / validation — to iterate cheaply, since test labels are hidden)
| File | Size | What | GT |
|---|---|---|---|
| `queries_dev.json` | **429** | templated rules: cheaper_same_brand (20), pricier_same_brand (20), format_swap (150), brand_swap (89), claim_swap (150). Built by `scripts/build_dev_set.py`. | exact `answer_product_id` |
| `queries_dev_rule5.json` | 400 | Sonnet free-form (seed 42, target rank 5–25) | exact |
| `queries_dev_rule5_a.json` | 646 | Sonnet free-form (seed 44, rank 10–40) — **held out** | exact |
| `queries_dev_rule5_b.json` | 608 | Sonnet free-form (seed 45, rank 30–80) — **held out, harder** | exact |
| `queries_dev_full.json` | **2,083** | merged of all above; every entry has `answer_product_id` + `translated_question_rewritten`; 8 rule tags (5 templated + `llm_free_form_orig/a/b`). | exact |

**Generation cost:** ~$40 of Sonnet (Rule-5/5A/5B + rewriter). Same 4,075 corpus for all.

## ⚠️ Ours vs provided — the critical difference
| | Provided (official-20) | Ours (dev) |
|---|---|---|
| Ground truth | none → **graded judge** (many products relevant) | **exact single answer** (deterministic) |
| Scores look | high (0.87–0.91) — lenient/dense | low (0.04–0.17) — harsh/sparse |
| Use | final proxy metric, overfit-prone (n=20) | cheap recall/iteration, **relative** signal; risk of self-generation bias |
| Phrasing | natural human language | templated / LLM-generated |
**We use validation only.** Real test = organizers' withheld judgments at submission. No leakage.

---

# PAGE 3 — Approaches tried (data → performance, worked/didn't)

## Stage-1 retrieval (recall is the bottleneck) — measured on dev-429 (exact)
| Method | R@20 | NDCG@10 (no rerank) | verdict |
|---|---|---|---|
| iSEARLE/SEARLE-Phi, CLIP ViT-L/14 (EXP-001) | 0.084 | 0.029 | weak baseline |
| + name-text fusion α=0.05 | 0.084 | ~0.03 | marginal |
| SEARLE-**OTI** (GPU textual inversion, EXP-004) | 0.142 | 0.058 | **2× Phi** ✅ |
| open_clip **ViT-H/14** (training-free img+text) | 0.343 | 0.172 | **4× Phi** ✅ |
| **ViT-bigG-14 + name-fusion** | **0.359** | 0.157 | **best visual** ✅ |
| EVA02-L | 0.294 | — | worse |
| SigLIP-2 | — | — | failed to load |
| **Symbolic brand-union** (brand_swap only) | brand R@50 **0.191** (vs 0.04) | brand NDCG@10 0.075 | **7.5× on brand_swap** ✅ |

**Finding:** visual backbones plateau ~0.43 R@50 — bigger ≠ better past ViT-H. Remaining recall
needs **text/brand grounding**, not bigger CLIP.

## Stage-2 reranking — measured on official-20 (Sonnet judge NDCG@10) + dev-429
| Reranker | official-20 | dev-429 | cost/full rerank |
|---|---|---|---|
| **Qwen2.5-VL-72B** | **0.913** | 0.057 | $0 (GPU-hr) |
| InternVL3-38B | 0.907 | 0.051 | $0 |
| Qwen2.5-VL-32B | 0.894 | 0.062 | $0 |
| **Haiku (API baseline)** | 0.872 | 0.044 | **~$11** |
| Pixtral-Large (FP8) | 0.869 | 0.050 | $0 |
| GLM-4.5V | ❌ load fail | — | — |

## Final fixed pipeline (dev-429): bigG+fusion + brand-union → Qwen-32B rerank
| Stage | NDCG@10 | MAP |
|---|---|---|
| Old (Phi + Haiku) | 0.044 | 0.024 |
| bigG+fusion Stage-1 | 0.157 | 0.130 |
| + brand-union | 0.170 | 0.144 |
| + Qwen-32B rerank | **0.165–0.17** | **0.152** |
→ **~3.8–4× over the old pipeline**; brand_swap 0.010 → 0.075.

## ✅ What worked
- **Local open VLM rerank ≥ Haiku at ~$0 marginal cost** (Qwen-72B 0.913). Changes cost structure.
- **Stronger Stage-1 backbone** (ViT-H/bigG, 4–5× recall).
- **SEARLE-OTI** > Phi at Stage-1.
- **Symbolic brand-union** for brand_swap (7.5×).
- Free, deterministic dev harness + cached judge → cheap iteration.

## ❌ What didn't
- Sonnet-**rewritten captions** (block 7) — hurt every metric.
- **Embedding** brand fusion (marginal); **brand-only** fusion (helps 1 rule, loses overall).
- Generic name-fusion at high α — hurts overall (EXP-003).
- **GLM-4.5V** (vLLM load fail), **Pixtral-Large** FP8 (below Haiku).
- Ops waste (~$14.5 of the ~$24 GPU): driver/torch mismatch, over-scoped sessions cut by deadline, over-provisioned disk — all corrected (resumable scripts, reaper, high-rel hosts, tight caps).

## Spend to date
- **Anthropic API ≈ $230** (mostly Haiku reranks during blocks 1–7 + dev generation).
- **GPU (Vast.ai) ≈ $24** (validation + retrieval; the reranker swap pays this back in <2 Haiku reranks).

---

# PAGE 4 — In flight & what's next

## ✅ Broad eval — DONE (frozen pipeline bigG+fusion + brand-union → Qwen-32B)
- **Full-dev 2,083: NDCG@10 0.255, MAP 0.238** (vs old 0.162, **+57%**); held-out splits degrade
  gracefully (orig 0.393 → a 0.300 → b 0.168) — robust, training-free.
- **Official-20 (Sonnet judge): NDCG@10 0.876, NDCG@20 0.946, MAP 0.943, recall@20 1.0.**

## ✅ Already done this round
- Reranker validation (local VLM beats Haiku); retrieval recall sweep; dev-429 fix; brand-union;
  **submission artifact built** (`data/submission/submission_bigg_qwen32b.json`).

## 🚀 What to try to bring it forward (ranked)
1. **LLM brand extractor** — generalize brand-union to *natural* (official) queries (current parser is
   templated-only). Direct path to brand-swap gains on the real test.
2. **Finalize & submit**: pick reranker for the submission run — **Qwen-72B** (best official 0.913) for
   Run1; use the 5 allowed runs for variants (32B, ±brand-union, ±OTI).
3. **Paper-recommended training-data methods** (organizers' reading list): **MagicLens** (ZS CIR, the
   original Stage-1 plan), **TransAgg + UNION** (null-text target augmentation, +2–3 mAP with ~5K triplets),
   **LaSCo/CASE**. These attack the recall ceiling at the source.
4. **Hybrid text retrieval** beyond brand (category/format/claim grounding) — `format_swap`/`claim_swap`
   also benefit from text.
5. **Ensemble Stage-1** (ViT-H ∪ bigG ∪ OTI ∪ brand-union) → higher pooled recall → rerank.
6. **Write-up**: methodology (Pareto + R5A/R5B cross-val), what worked/didn't, overfitting battle.

## Recommended immediate sequence
1. Land Step-1 broad eval → confirm no overfit on R5A/R5B.
2. Build the 5 submission runs (Qwen-72B primary), self-score with judge, email `retrank_CIR_Run{1..5}.json`.
3. In parallel, prototype the LLM brand extractor (cheapest real-test gain) + start the paper.
