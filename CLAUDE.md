# CAMEO-CIR: Composed Image Retrieval for NTCIR-19 CAMEO Task

## Project goal
Participate in the CAMEO composed image retrieval subtask (NTCIR-19, Dec 2026, Tokyo).
Given a reference product image + natural language modification query, retrieve a ranked 
list of matching products from a corpus of 4,075 face cleanser products (ViEcom-Rec dataset).

## Dataset
- Source: HuggingFace `allietran/CAMEO`
- Raw data lives in `data/raw/` (gitignored)
- 4,075 products, face cleanser category, Vietnamese e-commerce
- Evaluation: NDCG@K and MAP, TREC-style human relevance judgments

## Stack
- Python 3.11+, PyTorch, open_clip_torch, transformers, datasets (HuggingFace)
- Arch Linux, PyCharm, Claude Code

## Architecture plan
- Stage 1: MagicLens zero-shot CIR scoring over full 4,075 corpus (exhaustive, no ANN)
- Stage 2: Cross-encoder reranking over candidates

## Key files
- `src/data/cameo_dataset.py` — dataset loading and preprocessing
- `src/models/` — retrieval models
- `src/eval/metrics.py` — NDCG@K, MAP
- `scripts/download_data.py` — pulls from HuggingFace
- `scripts/explore_data.py` — EDA

## Commands
- Download data: `python scripts/download_data.py`
- Run baseline: `python scripts/run_baseline.py`
```

**5. Create `requirements.txt`**
```
torch
torchvision
open_clip_torch
transformers
datasets
pillow
numpy
pandas
tqdm
faiss-cpu
scikit-learn
matplotlib
jupyter
