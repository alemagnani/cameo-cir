# CAMEO-CIR — Experiment Log

Composed Image Retrieval for the **NTCIR-19 CAMEO** subtask. This README is the
running log of the task, the data, candidate approaches, design decisions,
experiments, and results. Code-level details belong in the source; the *why*
belongs here.

See `CLAUDE.md` for the project brief and stack.

---

## 1. The task

NTCIR-19 CAMEO, **Composed Image Retrieval (CIR)** subtask.

**Input.** A reference product image *I_r* + a natural-language modification
query *t* (in Vietnamese, possibly mixed with English).

**Output.** A ranked list of products from the official corpus that match the
*reference + modification* — i.e., visually similar to *I_r* with the changes
described in *t*.

**Domain.** Face cleansers from Vietnamese e-commerce (ViEcom-Rec).

**Evaluation.** TREC-style human relevance judgments. Primary metrics:
**NDCG@K** and **MAP**. Submission format follows `sample_submission.json`
(downloadable from the task page; saved here as `data/queries_sample.json`).

**Why this is hard for off-the-shelf CIR models:**
- *Niche domain.* Cleansers — small visual differences (size, packaging,
  formulation, "skincare claim" labels) — pre-trained encoders trained on
  generic web images may not separate them well.
- *Vietnamese queries.* Most public CIR models are English-only.
- *Tiny test set* (20 queries). No room for a learned curve — robust
  zero-shot or weakly-supervised wins.
- *Small corpus* (4,075). Exhaustive scoring is trivial; ANN unnecessary.

---

## 2. The data

| Item | Value |
|---|---|
| Source | HuggingFace `allietran/CAMEO` |
| Corpus size | **4,075** products (final; the 2,244 figure was outdated, per Allie Tran 2026-04-13) |
| Domain | Face cleanser, Vietnamese e-commerce |
| Test queries | **20** official CIR queries (+ 10 QA queries we don't compete on) |
| Modalities per product | Product image(s) + Vietnamese title + description + 369k user reviews |
| Official query file | `data/queries_official.json` — `cameo-task.com/static/final_questions_CAMEO.json` (downloaded 2026-05-16) |
| CIR-only subset | `data/queries_official_cir.json` (20 entries, filtered to keys `CIR-*`) |
| Submission | Email to `allie.tran@dcu.ie`, filename `retrank_CIR_Run{1..5}.json`, max 5 runs, plus a workshop paper for NTCIR-19 proceedings |

**Official query schema (different from `queries_sample.json`!):**
```json
{
  "CIR-0": {
    "image_path": "images/14/90/149019141_2282526298.jpg",
    "product_id": "590",
    "question": "Mình tìm loại nào giống này mà rẻ hơn, tầm 40k-50k thôi, vẫn của Pond's thì tốt.",
    "translated_question": "I'm looking for something similar but cheaper, around 40k–50k, preferably still from Pond's."
  }
}
```
- Fields are `image_path`, `product_id`, `question` (Vietnamese), `translated_question` (English) — no `answer_*` fields. Test labels are withheld.
- `queries_sample.json` (5 entries, English, with answers) is *example format only*; do not confuse with the real test set.
- All 20 reference images verified present under `data/raw/images/` ✓.

**Local layout.**
```
data/
  raw/                            # gitignored; HuggingFace dump (images, parquets)
  processed/                      # cached features, embeddings
  eval_cache/
  queries_sample.json             # old 5-query English sample with answers (do not use as test set)
  queries_official.json           # 30 entries: 20 CIR + 10 QA, downloaded from cameo-task.com
  queries_official_cir.json       # 20 CIR-only entries (what we evaluate against)
```

**Key constraints.**
- No labeled CIR training set is provided for this corpus — this is a
  **zero-shot / weakly-supervised** retrieval task.
- Test relevance judgments are **withheld until after submission**. We must
  build our own offline eval (Sonnet vision judge + synthetic dev set) to
  iterate.
- Queries are **Vietnamese first**; an organizer-provided English
  `translated_question` is included, which lets us punt on translation in
  v0 and revisit it as a controlled experiment later.

---

## 3. Reference papers (in `related_papers/`)

The five task-recommended papers, summarized for *what we can borrow*:

| Paper | What it gives us | Pre-trained weights? |
|---|---|---|
| **MagicLens** (Zhang+ ICML'24, `magic_lens.pdf`) | Strongest open ZS-CIR dual-encoder (CoCa/CLIP backbone, 36.7M web-mined triplets). SOTA on CIRCO (mAP@5 = 34.1). | Yes — `google-deepmind/magiclens` |
| **ZSCIR / TransAgg** (Liu+ BMVC'23, `2306.07272v2.pdf`) | Auto-builds CIR triplets from caption pairs via templates/LLMs; transformer fusion + adaptive aggregation over BLIP/CLIP. Recipe for synthetic training data. | Yes — `BUAADreamer/SPN4CIR` |
| **UNION** (Le+ ICDM'25, `S25202.pdf`) | Augments the *target* representation with a null-text-prompted feature (image + null-text → Transformer/MLP). +2–3 mAP on CIRCO with only 5K triplets. Plug-in on top of TransAgg. | Yes — `baohl00/UNION_for_IGROT` |
| **LaSCo / CASE** (Levy+ AAAI'24, `2303.09429v2.pdf`) | 389K-triplet CIR dataset from VQA2.0 + CASE (BLIP early-fusion + reverse-query). Strong supervised baseline; useful for pre-training/fine-tuning. | Yes — `levymsn/LaSCo` |
| **iSEARLE / CIRCO** (Agnolucci+ TPAMI'25, `iSEARLE.pdf`) | Textual inversion: image → pseudo-word in CLIP token space, then text-to-image retrieval with relative caption. **Trains on unlabeled images only** — fits CAMEO perfectly. | Yes — `miccunifi/SEARLE` |

Local code mirrors live in `LaSCo/`, `SEARLE/`, `SPN4CIR/`, `UNION_for_IGROT/`,
and `related_papers/code/magiclens/`.

The organizers also list **PhoBERT** (Vietnamese BERT, `VinAIResearch/PhoBERT`,
arXiv:2003.00744) under the QA subtask — relevant to us as a candidate
Vietnamese text encoder if multilingual CLIP variants underperform.

---

## 4. Candidate approaches

Five concrete options, ordered roughly by effort vs. expected payoff. The plan
is to land **A** as a baseline first, then run **B** and **C** in parallel,
then layer **D** on top of whichever wins, and consider **E** if we have headroom.

### A. MagicLens-L zero-shot (strong baseline)
- **What.** Encode the 4,075 products with the released MagicLens-L target
  encoder; for each query encode (*I_r*, *t*) with the multimodal encoder;
  rank products by cosine similarity.
- **Why first.** Best-in-class open ZS-CIR. CIRCO mAP@5 = 34.1 with no
  training. Single forward pass — fastest path to a leaderboard number.
- **Risks.** English-only; cleanser-domain unseen; modification-text style
  may not match training distribution.

### B. iSEARLE / textual inversion on the CAMEO corpus
- **What.** Pre-train iSEARLE's textual-inversion network *on the 4,075 product
  images* (no labels needed). At inference: image *I_r* → pseudo-word *S*\* →
  feed `"a photo of S* {relative caption}"` through CLIP text encoder →
  text-to-image retrieval over product image features.
- **Why.** Specifically designed for ZS-CIR. Domain adaptation comes for free
  by training the inversion net on cleanser images. Light compute.
- **Risks.** CLIP text encoder is English-only by default → either translate
  Vietnamese queries to English (NLLB / Claude / GPT) or swap CLIP for a
  multilingual variant (M-CLIP, SigLIP-mCLIP).

### C. TransAgg + UNION with synthetic CAMEO triplets
- **What.** Generate pseudo-(*I_r*, *t*, *I_t*) triplets from the 4,075
  products: pair products via title/embedding similarity, then prompt an LLM
  to write the modification text bridging them (LaSCo-LLM recipe). Train
  TransAgg with UNION target features on this synthetic set.
- **Why.** Domain-tailored training without manual labeling. UNION's
  null-text-augmented target consistently helps when there are few triplets
  (UNION reports best CIRCO mAP@10 = 34.5 with only 5K triplets).
- **Risks.** Synthetic triplet quality is the bottleneck; needs careful filter.

### D. Two-stage retrieve-then-rerank
- **What.** Stage 1: any of A/B/C produces top-K (K ≈ 50). Stage 2: rerank
  with one of:
  - **CASE** (BLIP early-fusion, cross-attention) — strong supervised reranker
    even when only zero-shot transferred.
  - **VLM-as-judge** — give a vision-language model (Claude / GPT-4V) the
    reference image, modification text, and each candidate, ask for a
    relevance score. Slow but powerful for 20 queries × 50 candidates.
- **Why.** Stage 2 cross-encoders / VLMs catch fine-grained mismatches
  (color, count, claim text) that bi-encoders miss.
- **Risks.** API cost / latency; prompt design for VLM judge.

### E. Vietnamese handling (cross-cutting, not a standalone option)
- **Translate-to-English** the modification text (cheapest; LLM or NLLB).
- **Use multilingual encoder** (M-CLIP, SigLIP) — replaces English CLIP in B/C.
- **Use product metadata.** The corpus has Vietnamese titles/descriptions —
  can feed both image and Vietnamese caption into the encoder for richer
  product representation.

### Decision pending
Which of A/B/C lands first depends on whether the released MagicLens
checkpoint is straightforward to set up locally; if not, B (iSEARLE) becomes
the baseline.

---

## 5. Decisions & rationale

Each entry: **Date — Decision — Rationale — Status.**

- *2026-05-09* — Use this README as the single source of truth for experiments and decisions. *Rationale:* keeps design intent and result history co-located with the code, reviewable in git history. *Status:* active.
- *2026-05-09* — Pull the five task-recommended papers (MagicLens, ZSCIR, UNION, LaSCo/CASE, CIRCO/iSEARLE) into `related_papers/`. *Rationale:* organizers' explicit reading list — covers ZS-CIR SOTA, training-data construction, target representations, and the matching benchmark methodology. *Status:* active.
- *2026-05-09* — Plan staged rollout: A (MagicLens ZS) → B (iSEARLE) and C (TransAgg+UNION) in parallel → D (rerank) on the winner. *Rationale:* A is the cheapest credible number; B and C address the two main gaps (domain, training-data); D layers on the winner without committing to it early. *Status:* active.

### Open questions
- Translate Vietnamese queries to English vs. use a multilingual encoder? (Probably try both — translation is cheap; multilingual is more elegant.)
- Stage-1 candidate cutoff for reranking — K=50 seems standard; revisit after seeing recall@K curves.
- Do we have access to per-product textual metadata (Vietnamese titles, descriptions) at inference? If so, fuse it into the product representation.

---

## 6. Architecture

Two-stage retrieval (target shape — concrete model TBD by experiments):

1. **Stage 1 — Candidate scoring.** Bi-encoder (MagicLens / iSEARLE / TransAgg+UNION) over the full 4,075 corpus. Exhaustive — no ANN needed.
2. **Stage 2 — Reranking.** Cross-encoder (CASE) or VLM-as-judge over top-K stage-1 candidates.

Architecture changes should be logged in §5 with rationale before code changes land.

---

## 7. Experiments

Each experiment: **ID, date, hypothesis, setup, result, takeaway.**

| ID | Date | Approach | Backbone | NDCG@10 (dev) | MAP (dev) | NDCG@10 (judge) | MAP (judge) | Notes |
|----|------|----------|----------|---------------|-----------|-----------------|-------------|-------|
| EXP-001 | 2026-05-27 | iSEARLE zero-shot | CLIP ViT-L/14 | 0.029 | 0.024 | 0.682 | 0.751 | dev=429 hard labels; judge=20 official w/ Haiku |
| EXP-005 | 2026-05-30 | EXP-001 + Sonnet listwise rerank top-20 | CLIP ViT-L/14 + Sonnet 4.6 vision | 0.029 | 0.024 | 0.703 | 0.789 | +13% NDCG@5 on judge eval; null on dev set (see takeaway) |
| EXP-003 | 2026-05-31 | EXP-001 + index text-fusion α·CLIP-text(name) + (1−α)·CLIP-image | CLIP ViT-L/14 | 0.023 / 0.023 / 0.015 | 0.016 / 0.014 / 0.011 | — | — | α=0.3/0.5/0.7 all hurt overall; brand_swap +0.009 at α=0.5, claim_swap collapses |

### 7.1 Dev set (`data/queries_dev.json`)

Built by `scripts/build_dev_set.py` because the official test set ships
without `answer_product_id` — we need hard labels to iterate.

- **429 queries** across five templated rules (per-rule slices in
  `data/queries_dev_by_rule/{rule}.json` for sliced eval):
  - `cheaper_same_brand` (20): ref = most-expensive in brand, target = cheapest;
    "same brand but cheaper, around {tgt_price_k}k VND".
  - `pricier_same_brand` (20): inverse direction.
  - `format_swap` (150): within brand, swap product format (foam ↔ gel ↔
    cream ↔ oil ↔ micellar ↔ powder ↔ clay). Keyword-mined from VN+EN
    names+descriptions; only products with exactly one format tag.
  - `brand_swap` (89): cross-brand pair at ±15% price, same (format, claim).
    Saturates at 89 from corpus structure.
  - `claim_swap` (150): within brand, swap skin-concern claim
    (oily / dry / acne / sensitive / whitening / anti-aging).
- **Uniqueness constraint** per rule:
  - price-swap: no other in-brand product within ±25% of target price.
  - format/claim-swap: target is the only in-brand exemplar of the target
    tag.
  - brand-swap: target is the only product in target-brand matching that
    (format, claim) within ±25% of ref price.
  Together these make NDCG@1 meaningful on every rule.
- **Bilingual:** every entry has English `translated_question` (template
  output) and Vietnamese `question` (Sonnet-translated, batched, deterministic
  seed). Decouples "model handles Vietnamese" from "model handles CIR".
- **Sanity checks:** all 429 reference images resolve under `data/raw/`;
  all answer product IDs exist in the 4,075-product corpus; no ref==tgt
  collisions.

What the dev set is **not**: a stand-in for the organizers' relevance
judgments. It probes price / brand / format / claim reasoning under
controlled templates. The Sonnet vision judge on the official 20 queries
remains the second eval signal.

Pending extension (Rule 5, deferred): **LLM free-form mods** — OpenCLIP-embed
the 4,075 corpus, sample ref → target at neighbor ranks 5–20, Sonnet judges
each pair + writes the modification text; keep confidence ≥ 4. Target
~150–200 queries. Adds realism missing from the templated rules. To land
once the deterministic dev set has been used against a baseline.

### Detailed entries

<!-- Add per-experiment write-ups below. Keep the table above as the index. -->

#### EXP-000 (template)
- **Hypothesis:**
- **Setup:** model, data split, hyperparameters, seed, hardware.
- **Command:** exact CLI invocation.
- **Result:** metric values; link to logs / output dir.
- **Takeaway:** what we learned; what changes next.

#### EXP-001 — iSEARLE-XL zero-shot, English `translated_question`
- **Hypothesis:** SEARLE Phi (trained on Open Images concepts) + CLIP ViT-L/14
  gives a credible ZS-CIR floor without any CAMEO-domain adaptation. Using
  the organizer-provided English translation lets us punt on Vietnamese in v0.
- **Setup:** SEARLE-XL via `torch.hub.load('miccunifi/SEARLE', backbone='ViT-L/14')`;
  OpenAI CLIP ViT-L/14; prompt `"a photo of $ that {translated_question}"`;
  index = 4,075 corpus images encoded by CLIP image encoder (`.mp4`
  products encoded from first frame; 3 corrupt `.jpg`s use a gray placeholder);
  reference image excluded from retrieved results. CPU, batch=8. Seed = SEARLE
  default. Hardware: this box (CPU only).
- **Command:**
  ```
  python scripts/run_searle.py --queries data/queries_dev.json          --out data/runs/searle_xl_dev.json      --backbone ViT-L/14 --batch-size 8
  python scripts/run_searle.py --queries data/queries_official_cir.json --out data/runs/searle_xl_official.json --backbone ViT-L/14 --batch-size 8
  python scripts/eval_by_rule.py --run data/runs/searle_xl_dev.json
  python scripts/evaluate.py --run data/runs/searle_xl_official.json --queries data/queries_official_cir.json \
      --use-judge --judge-model haiku --judge-k 20 --k 5 10 20
  ```
- **Result — dev set, hard labels (`data/queries_dev.json`, 429 queries):**

  | rule                 | N   | NDCG@1 | NDCG@10 | R@10   | MAP    |
  |----------------------|-----|--------|---------|--------|--------|
  | cheaper_same_brand   | 20  | 0.0000 | 0.0158  | 0.0500 | 0.0124 |
  | pricier_same_brand   | 20  | 0.1000 | 0.1315  | 0.1500 | 0.1284 |
  | format_swap          | 150 | 0.0000 | 0.0237  | 0.0533 | 0.0184 |
  | brand_swap           | 89  | 0.0000 | 0.0000  | 0.0000 | 0.0012 |
  | claim_swap           | 150 | 0.0067 | 0.0390  | 0.0800 | 0.0302 |
  | **overall**          | 429 | 0.0070 | 0.0288  | 0.0559 | 0.0238 |

- **Result — official 20 queries, Sonnet-Haiku judge (top-20 per query):**
  - NDCG@5 = 0.5647, NDCG@10 = 0.6817, NDCG@20 = 0.7871
  - Recall@5 = 0.27, Recall@10 = 0.57, Recall@20 = 1.00
  - MAP = 0.7513
  - 400 judge calls cached at `data/eval_cache/`.

- **Takeaway:**
  1. The two eval signals **strongly diverge** — judge metrics ~25× higher than
     hard-label metrics. Expected: dev set asks "find the *one* exact target",
     while the permissive judge marks any visually-similar same-brand cleanser
     as relevant. Both are real signals on different things; keep both.
  2. `brand_swap` is at the floor (NDCG@10 = 0). SEARLE's CLIP text encoder has
     no grounding on Vietnamese e-commerce brand names ("Wonjin Effect",
     "BYVIBES WONDER BATH") — pseudo-token + brand mention in the prompt is
     not enough.
  3. `pricier_same_brand` (NDCG@10 = 0.13) beats `cheaper_same_brand`
     (NDCG@10 = 0.016) by 8×. Hypothesis: when the *reference* is the
     cheapest in-brand product, its visual identity (and pseudo-token) carries
     more brand signal toward upmarket variants than the reverse direction.
     Worth re-checking with brand metadata fused into the product representation.
  4. Recall@20 = 1.0 on the judge eval means top-20 always contains *something*
     the judge accepts — but NDCG@5 is far below 1, so ranking inside top-20 is
     the bottleneck, not candidate generation. Reranking (approach D) is the
     natural next move.

- **Next experiments:**
  - **EXP-002**: replace English `translated_question` with the original
    Vietnamese `question` via a multilingual CLIP variant (M-CLIP / SigLIP)
    to isolate the translation-loss effect.
  - **EXP-003**: fuse the Vietnamese product name (`translated_name` or
    `product_name`) into the product representation — concatenate text+image
    embeddings — to give brand-swap a chance.
  - **EXP-004**: pre-train SEARLE Phi on the 4,075 product images using
    SEARLE's own OTI-based distillation recipe (Stage 1 of iSEARLE) — true
    domain adaptation.
  - **EXP-005**: layer a Sonnet vision reranker over top-20 (Stage 2) to attack
    the in-top-20 ranking gap visible on the judge eval.

#### EXP-005 — Sonnet listwise vision reranker on EXP-001 top-20
- **Hypothesis:** EXP-001 has Recall@20 = 1.0 on the judge eval but NDCG@5 only
  0.56 — relevant items *are* in the top-20, but ranking inside top-20 is bad.
  A vision-capable LLM should fix the ranking by seeing the reference, the
  modification, and all 20 candidate images at once, then emitting a list.
- **Setup:** `scripts/rerank_sonnet.py` — for each query, send Sonnet 4.6 the
  reference image, the English `translated_question`, and the top-20 candidate
  images labeled (A)..(T) with their English product names. Sonnet returns a
  20-letter ordering; we splice it into positions 0..19 of the run and keep
  20..49 as-is. Per-query cache keyed by (qid, candidate-set hash, model). All
  images downscaled to ≤384px to bound input cost.
- **Command:**
  ```
  python scripts/rerank_sonnet.py --run data/runs/searle_xl_official.json \
      --queries data/queries_official_cir.json \
      --out data/runs/searle_xl_official.rerank_sonnet.json --topk 20 --model sonnet
  python scripts/rerank_sonnet.py --run data/runs/searle_xl_dev.json \
      --queries data/queries_dev.json \
      --out data/runs/searle_xl_dev.rerank_sonnet.json --topk 20 --model sonnet
  python scripts/evaluate.py --run data/runs/searle_xl_official.rerank_sonnet.json \
      --queries data/queries_official_cir.json --use-judge --judge-model haiku \
      --judge-k 20 --k 5 10 20
  python scripts/eval_by_rule.py --run data/runs/searle_xl_dev.rerank_sonnet.json
  ```
- **Result — official 20, Haiku text judge (vs EXP-001):**

  | metric    | EXP-001 | EXP-005 | Δ        |
  |-----------|---------|---------|----------|
  | NDCG@5    | 0.5647  | 0.6404  | +0.076  (+13.4%) |
  | NDCG@10   | 0.6817  | 0.7026  | +0.021  (+3.1%)  |
  | NDCG@20   | 0.7871  | 0.8303  | +0.043  (+5.5%)  |
  | MAP       | 0.7513  | 0.7889  | +0.038  (+5.0%)  |
  | Recall@20 | 1.0000  | 1.0000  | unchanged (same candidate set) |

  Big per-query wins: CIR-13 0.19→1.00, CIR-8 0.48→0.87, CIR-4 0.74→0.85.
  Regressions: CIR-1 0.63→0.52, CIR-18 0.24→0.00, CIR-5 0.76→0.49.

- **Result — dev set 429, hard labels (vs EXP-001):**

  | rule                 | N   | NDCG@10 (EXP-001) | NDCG@10 (EXP-005) | Δ      |
  |----------------------|-----|-------------------|-------------------|--------|
  | cheaper_same_brand   | 20  | 0.0158            | 0.0178            | +0.002 |
  | pricier_same_brand   | 20  | 0.1315            | 0.1315            | 0      |
  | format_swap          | 150 | 0.0237            | 0.0237            | 0      |
  | brand_swap           | 89  | 0.0000            | 0.0000            | 0      |
  | claim_swap           | 150 | 0.0390            | 0.0390            | 0      |
  | **overall**          | 429 | **0.0288**        | **0.0289**        | ~0     |

  Last query (DEV-0428) fell back to stage-1 order — API credits hit zero on
  call 429 — but the cached 428 reorderings are correct and the metric impact
  of one query is negligible.

- **Takeaway — the rerank gap is real and informative:**
  1. **Reranking only helps when stage-1 already puts relevant candidates in the
     head.** On the permissive judge eval, EXP-001 had Recall@20 = 1.0
     — the targets *were* in the top-20, just in the wrong order; Sonnet fixed
     that and bought +13% NDCG@5. On the strict dev set, brand_swap top-20
     contains the exact target ~1% of the time (R@20 = 0.011) and no rerank
     can fix what isn't there.
  2. **The dev set's null result is therefore a candidate-generation diagnosis,
     not a rerank failure.** Stage-1 is the bottleneck for fine-grained
     discrimination on this corpus. Reranking is a second-order optimization
     and should be paused until stage-1 puts the right candidates in the head.
  3. The official-20 win is genuine (independent: Sonnet reranker, Haiku text
     judge — different models, different modalities) and is what we'd submit
     today. But the dev-set null tells us we won't beat it by stacking more
     rerank passes; the leverage is upstream.

- **Next move:** focus on stage-1 improvements that put the right products in
  top-20:
  - **EXP-002** (multilingual CLIP / SigLIP) — addresses brand/text grounding
    in Vietnamese.
  - **EXP-003** (fuse product name + image into the index representation) —
    directly attacks brand_swap and format_swap, which need text grounding to
    rank correctly.
  - **EXP-004** (SEARLE Phi pre-train on 4,075 product images) — gives the
    inversion network cleanser-domain priors.

  Rerank (EXP-005) should be re-run on top of whichever of these moves the
  dev-set Recall@20 needle.

#### EXP-003 — index text-fusion (α-weighted CLIP-text(product name) + CLIP-image)
- **Hypothesis:** EXP-005 said stage-1 is the bottleneck. The corpus's English
  product names contain explicit brand / format / claim tokens. Fusing
  CLIP-text(translated_name) into the index representation should give the
  query-side CLIP text encoder something to match against, especially for
  `brand_swap` (which is at floor) and `format_swap`.
- **Setup:** extend `scripts/run_searle.py` with `--fuse-text α`. For each
  product: index = α · norm(CLIP-text(translated_name)) + (1−α) ·
  norm(CLIP-image(product_image)), re-normalized. Same query path as EXP-001.
  Cached image and text feature tensors per backbone (`data/processed/clip_vit-l14_index_text.pt`).
  Swept α ∈ {0.0, 0.3, 0.5, 0.7}; α=0 reproduces EXP-001 exactly.
- **Command:**
  ```
  for a in 0.3 0.5 0.7; do
    python scripts/run_searle.py --queries data/queries_dev.json \
        --out data/runs/searle_xl_dev_fuse${a}.json \
        --backbone ViT-L/14 --batch-size 8 --fuse-text $a
    python scripts/eval_by_rule.py --run data/runs/searle_xl_dev_fuse${a}.json
  done
  ```
- **Result — dev set 429, hard labels:**

  | rule \ α              | 0.0 (EXP-001) | 0.3    | 0.5    | 0.7    |
  |-----------------------|---------------|--------|--------|--------|
  | cheaper_same_brand    | 0.0158        | 0.0178 | 0.0145 | 0.0000 |
  | pricier_same_brand    | 0.1315        | 0.1315 | 0.1340 | 0.0545 |
  | format_swap           | 0.0237        | 0.0242 | 0.0257 | 0.0200 |
  | brand_swap            | 0.0000        | 0.0032 | **0.0089** | 0.0056 |
  | claim_swap            | **0.0390**    | 0.0178 | 0.0138 | 0.0124 |
  | **overall NDCG@10**   | **0.0288**    | 0.0223 | 0.0226 | 0.0150 |
  | **overall MAP**       | **0.0238**    | 0.0162 | 0.0141 | 0.0110 |

  Recall@20 (overall): 0.079 → 0.077 → 0.063 → 0.047 — strictly monotonically
  worse with more text fusion.

- **Takeaway — null result, but informative:**
  1. Naive product-name fusion **hurts overall** at every α tested. There is
     no Pareto-improving α in {0.3, 0.5, 0.7}.
  2. The damage is concentrated on **`claim_swap`** (0.039 → 0.014, −65% at
     α=0.5). Mechanism: queries mention "for sensitive skin" / "brightening"
     etc.; the indexed product names mention the same generic skin claims
     across hundreds of products. CLIP text↔text matching then drags many
     wrong products into the head.
  3. The single rule that *benefits* is **`brand_swap`** (0 → 0.009 at α=0.5),
     consistent with the diagnosis: brand_swap queries name a brand
     explicitly, and the indexed name carries the brand string literally.
  4. So the right text signal is **brand**, not full name. Full name embeds
     too much noise.

- **Next move(s):**
  - **EXP-003b** (cheap follow-up): fuse with CLIP-text(`"a photo of a {brand}
    facial cleanser"`) — strip everything but the brand from the product side.
    If brand_swap lifts further without crashing claim_swap, that's the right
    signal to keep. Or fuse a structured "brand={brand} format={fmt} claim={claim}"
    using the same keyword tables already in `build_dev_set.py`.
  - **EXP-002**: swap CLIP for a multilingual encoder (M-CLIP / SigLIP) so the
    query side can use the original Vietnamese — addresses the text-encoder
    grounding problem at the source instead of patching the index.
  - **EXP-004**: SEARLE Phi pre-train on the 4,075 cleanser images (OTI
    distillation). True domain adaptation of the inversion network — the
    biggest model-side move available.

---

## 8. Results summary

Best run so far: *none*.

Leaderboard-style table tracking only the runs we'd actually submit:

| Rank | Run ID | Description | NDCG@10 | MAP | Date |
|------|--------|-------------|---------|-----|------|
| —    | —      | —           | —       | —   | —    |

---

## 9. Reproducibility

- Data download: `python scripts/download_data.py`
- Baseline run: `python scripts/run_baseline.py`
- Eval: `python scripts/evaluate.py`
- Raw data is gitignored under `data/raw/`.
- Per experiment, record: seed, model + checkpoint hash, HuggingFace dataset
  revision, GPU, exact command. Put the output JSON under
  `data/<run_id>.json` and the eval JSON under `data/<run_id>.eval.json`.

---

## 10. References & vendored repos

- `LaSCo/` — LaSCo dataset & CASE baseline (Levy+ AAAI'24).
- `SEARLE/` — iSEARLE textual inversion (Agnolucci+ TPAMI'25).
- `SPN4CIR/` — TransAgg / ZSCIR (Liu+ BMVC'23) and downstream.
- `UNION_for_IGROT/` — UNION target representation (Le+ ICDM'25).
- `related_papers/code/magiclens/` — MagicLens (Zhang+ ICML'24).
- `related_papers/` — the five PDFs above.
