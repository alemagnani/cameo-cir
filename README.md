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
| Corpus size | **4,075** products (final; the 2,244 figure on the task page was outdated) |
| Domain | Face cleanser, Vietnamese e-commerce |
| Test queries | **20** official CIR queries |
| Modalities per product | Product image(s) + textual metadata (title, description in Vietnamese) |
| Submission template | `data/queries_sample.json` (must be downloaded from task page) |

**Local layout.**
```
data/
  raw/          # gitignored; HuggingFace dump
  processed/    # cached features, embeddings
  eval_cache/
  queries_sample.json   # official submission template (20 queries)
  test_run.json
  test_run.eval.json
```

**Key constraint.** No labeled CIR training set is provided for this corpus —
this is essentially a **zero-shot / weakly-supervised** retrieval task, which
shapes the approach choices below.

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

| ID | Date | Approach | Backbone | NDCG@10 | MAP | Notes |
|----|------|----------|----------|---------|-----|-------|
| —  | —    | —        | —        | —       | —   | no runs yet |

### Detailed entries

<!-- Add per-experiment write-ups below. Keep the table above as the index. -->

#### EXP-000 (template)
- **Hypothesis:**
- **Setup:** model, data split, hyperparameters, seed, hardware.
- **Command:** exact CLI invocation.
- **Result:** metric values; link to logs / output dir.
- **Takeaway:** what we learned; what changes next.

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
