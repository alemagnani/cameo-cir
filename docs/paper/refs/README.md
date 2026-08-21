# Reference library — CAMEO paper

Rule: every `\bibitem` in `../ntcir19_cameo_retrank.tex` must have its full paper here,
downloaded and read in full, and its entry verified against the PDF title page before citing.
(See memory `feedback-cite-only-after-reading`.)

| bibkey | PDF | Title / authors (verified from title page) | Venue, year | Read | Cited for |
|--------|-----|--------------------------------------------|-------------|------|-----------|
| `searle` | searle_baldrati_2303.15247.pdf | Zero-Shot Composed Image Retrieval with Textual Inversion — Baldrati, Agnolucci (equal), Bertini, Del Bimbo | ICCV 2023 (arXiv:2303.15247v2) | full | ZS-CIR textual-inversion baseline (SEARLE/OTI) |
| `pic2word` | pic2word_saito_2302.03084.pdf | Pic2Word: Mapping Pictures to Words for Zero-shot CIR — Saito, Sohn, Zhang, Li, Lee, Saenko, Pfister | CVPR 2023 (arXiv:2302.03084v2) | full | ZS-CIR pseudo-token line of work |
| `magiclens` | magiclens_zhang_2403.19651.pdf | MagicLens: Self-Supervised Image Retrieval with Open-Ended Instructions — Zhang, Luan, Hu, Lee, Qiao, Chen, Su, Chang | ICML 2024 (arXiv:2403.19651v2) | full | purpose-built multimodal CIR encoder |
| `openclip` | (Zenodo software record) | OpenCLIP — Ilharco et al. | Zenodo 2021, doi:10.5281/zenodo.5143773 | software cite | ViT-bigG-14 / ViT-H encoders |
| `qwenvl` | qwen25vl_bai_2502.13923.pdf | Qwen2.5-VL Technical Report — Qwen Team / S. Bai et al. | arXiv:2502.13923v1, 2025 | full | local VLM listwise reranker |
| `cameo` | tran2026cameo_overview.pdf | Overview of CAMEO — A. Tran, Q.-L. Tran, H.-B. Le, B. T. Nguyen, A. L. P. De Ocampo | NTCIR-19, Tokyo, 2026 (draft) | full (6 pp.) | task overview, metrics, leaderboard, sec 6.1 |
| `viecomrec` | viecomrec_tran_csonet2023.pdf | ViEcomRec: A Dataset for Recommendation in Vietnamese E-Commerce — Q.-L. Tran, B. T. Nguyen, G. J. F. Jones, C. Gurrin | CSoNet 2023, LNCS 14479, pp. 74–82, Springer 2024, doi:10.1007/978-981-97-0669-3_7 | full (9 pp.) | origin of our corpus (2,244 face cleansers, attribute extraction) |
| `union` | union_le_2511.22253.pdf | UNION: A Lightweight Target Representation for Efficient Zero-Shot Image-Guided Retrieval with Optional Textual Queries — H.-B. Le, **A. Tran**, B. T. Nguyen, L. Zhou, C. Gurrin | arXiv:2511.22253v1, 27 Nov 2025 | full | organizers' group's CIR/IGROT direction |

Read status (as of 2026-08-20): ALL eight read in full and every cited claim
verified against the full text (a dedicated agent read searle/pic2word/
magiclens/qwenvl/union cover-to-cover on 2026-08-20; cameo+viecomrec read
directly). Verdicts: all SUPPORTED as worded.

Caveat on `qwenvl`: the cited Qwen2.5-VL Technical Report (arXiv:2502.13923,
Feb 2025) documents only 3B/7B/72B — it does NOT cover Qwen2.5-VL-32B, which
was released later (March 2025) and which we use in Run4/Run5. We cite the
report only for the model family (Related Work); the 32B appears only as a
model name in result tables, so no claim depends on the report covering it.

Claim-scope notes (verified against the text actually read):
- `magiclens`: abstract states training on 36.7M (query image, instruction, target image)
  triplets — supports the "large image–instruction–image data" characterization.
- `union`: abstract/intro state it unifies CIR and sketch-based retrieval under
  "image-guided retrieval with optional text (IGROT)" via a lightweight target
  representation — supports the Related Work sentence.

Notes:
- `viecomrec` and `union` added 2026-08-17 (Allie Tran is a co-author of `union` and `cameo`).
- `qwenvl` title page credits "Qwen Team, Alibaba Group"; arXiv author metadata lists S. Bai et al. — we cite the latter form, which is standard.
