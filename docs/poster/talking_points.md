# Poster talking points — team retrank, NTCIR-19 CAMEO CIR

Short version. The full script, poster tours, decision log and Q&A are in
`docs/rehearsal/rehearsal_packet.pdf`. Every number matches the submitted paper.

## 30-second pitch
- Across Dev and the aggregate official results, getting the right products into the candidate pool was the bigger lever. Individual topics can still be quite sensitive to the reranker.
- Structured metadata, price above all, brings back products a visual encoder can't see.
- Local open-weight VLM rerankers are competitive with the hosted API at near-zero marginal cost, once a GPU host is rented.
- Best run: 0.740 / 0.714 nDCG@5/@10, 3rd of 5 teams at both cutoffs. At nDCG@10 the top four teams are within 0.019, and we don't infer an order among 2nd–4th.

## Walk the poster (top to bottom, left to right)
1. **The Task:** a reference image plus a shopper request goes in; a ranking of 4,086 Vietnamese skincare products comes out. Many requests are about price, brand or format, which live in the metadata, not the pixels.
2. **Approach:**
   - A training-free CLIP ViT-bigG branch, plus a metadata branch where Claude Haiku extracts brand, price, format and claim from the request.
   - Products of that brand are scored by price proximity and keyword hits. The branch abstains if no brand is confidently named.
   - Metadata hits go first, then the visual list. Qwen2.5-VL-72B, run locally, reranks the top 20.
3. **Three Evaluation Regimes:**
   - Dev-429: synthetic, one exact answer per query.
   - Official-20 proxy: a cached Claude Sonnet vision judge over our runs. 20 queries, one pass, noisy, not calibrated against human assessors.
   - Official: pooled human judgments.
   - These can't be compared with each other.
4. **Finding 1:** Dev R@50 goes 0.154 (SEARLE-Phi) → 0.436 (bigG + name fusion) → 0.527 (metadata-fused). With one gold answer, no reranker can beat the pool's recall.
5. **Finding 2:**
   - Metadata-only with gold fields: brand only 0.084, brand + price 0.232, all four terms 0.211.
   - Extraction loses nothing on the templated price rules, but hasn't been validated on natural queries.
   - On the full 2,083-query Dev set (79% free-form), fusion is 0.115 vs 0.119 for bigG alone: roughly neutral.
6. **Finding 3:**
   - Same SEARLE-Phi pool, proxy judge: Qwen-72B 0.913, InternVL3-38B 0.907, Qwen-32B 0.894, Haiku API 0.872, Pixtral 0.869.
   - Three of four local models score above the API, with the gaps within noise.
   - An extra pass costs about $0 at the margin, vs about $11 on the API.
7. **Implementation & Cost:** about $235 in hosted-LLM API calls plus about $60 of GPU.
8. **Official Results:**
   - Same pool, three rerankers: 0.030 apart (Friedman p≈0.52); per-topic ranges reach 0.22.
   - Same reranker, new Stage-1 pool: +0.078 (Run2→Run1) and +0.103 (Run4→Run5).
   - Those comparisons swap the whole Stage-1 family, and Run5's brand-union added no official candidates. So metadata's own effect is shown on Dev, not here.
9. **Where It Still Fails:**
   - Brand swap is the weakest Dev rule (0.117).
   - On Run1, the 11 official price-band topics scored 0.586 nDCG@10 vs 0.870 for the other 9. Their graded-relevant recall wasn't lower, which points to a top-rank ordering problem. This is descriptive, not causal.
10. **What Did Not Work:**
    - RRF of CLIP backbones; query rewriting.
    - Putting brand into the embedding.
    - Using format/claim as hard filters.
    - Visually reranking the metadata pool: Dev 0.200 → 0.173. Treat as a caution; the mechanism is unverified.
    - MagicLens: blocked by its environment.
11. **Takeaways / next:** better brand resolution; repeat the metadata-pool rerank audit; validate the extractor on natural official queries.

## Likely questions
- **Is the ordering among teams 2–4 established at nDCG@10?** No. Their point estimates are within 0.019 on 20 topics, and we ran no significance or equivalence tests, so we don't infer an ordering.
- **Does metadata explain the official gain?** Not by itself. The official contrasts change the whole Stage-1, and Run5's brand-union added no candidates. The metadata attribution comes from the Dev ablation.
- **Why is Dev so low and the proxy so high?** They use different ground truths: one exact answer versus a lenient graded judge.
