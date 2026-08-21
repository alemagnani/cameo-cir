# EXP-004 — running SEARLE OTI on a Vast.ai (or any CUDA Linux) GPU box

We pre-invert each of the 4,075 CAMEO product images on a GPU using SEARLE's
OTI (Optimization-based Textual Inversion). The output is a small bundle of
~4 MB of pseudo-token tensors + names, which we pull back and use at
inference time via `run_searle.py --oti-tokens <dir>`.

## What this folder ships

- `setup_remote.sh` — installs deps, downloads GPT phrases + Open Images V7
  class descriptions on a fresh GPU box.
- `run_remote.sh` — runs image-concept association then OTI inversion. End
  state: `SEARLE/data/oti_pseudo_tokens/cameo/test/cameo_GPTNeo27B/` is filled.

## Cost / wall-clock estimate

| GPU              | OTI steps | Wall-clock | Vast cost ($) |
|------------------|-----------|------------|---------------|
| RTX 3090 (24 GB) | 350       | ~45 min    | ~$0.30        |
| A100 (40 GB)     | 350       | ~15 min    | ~$0.20        |

## Recipe

1. **Rent a Vast.ai instance** with a PyTorch image (e.g. `pytorch/pytorch:2.x-cuda12.x-cudnn-runtime`).
   Pick ≥24 GB GPU RAM and a Linux template that ships with `curl`, `git`,
   `unzip`. Note the SSH host/port.

2. **Push the repo + data to the instance.** From your local box:

   ```bash
   # set this once
   export VAST_HOST=root@<host>
   export VAST_PORT=<port>
   export REMOTE=/workspace/cameo-retrank

   ssh -p $VAST_PORT $VAST_HOST "mkdir -p $REMOTE"

   # repo (excludes data/raw which we send separately)
   rsync -avz --exclude='data/raw' --exclude='.venv*' --exclude='__pycache__' \
         -e "ssh -p $VAST_PORT" \
         /home/alessandro/workspace/cameo-retrank/ $VAST_HOST:$REMOTE/

   # corpus — ~1.9 GB; skip the zip dump + reviews parquet we don't use here
   rsync -avz --exclude='images.zip' --exclude='reviews__test.parquet' \
         -e "ssh -p $VAST_PORT" \
         /home/alessandro/workspace/cameo-retrank/data/raw/ $VAST_HOST:$REMOTE/data/raw/
   ```

3. **Run setup + OTI on the instance.**

   ```bash
   ssh -p $VAST_PORT $VAST_HOST
   cd /workspace/cameo-retrank
   bash scripts/exp004/setup_remote.sh
   bash scripts/exp004/run_remote.sh    # ~45 min on RTX 3090
   ```

4. **Pull the OTI bundle back to local.**

   ```bash
   rsync -avz -e "ssh -p $VAST_PORT" \
         $VAST_HOST:$REMOTE/SEARLE/data/oti_pseudo_tokens/cameo/test/cameo_GPTNeo27B/ \
         /home/alessandro/workspace/cameo-retrank/data/checkpoints/oti_cameo_GPTNeo27B/
   ```

5. **Destroy the instance.** (Vast.ai console — important so it stops billing.)

6. **Run inference locally with OTI tokens.**

   ```bash
   PYTHONPATH=. python scripts/run_searle.py \
       --queries data/queries_dev.json \
       --out data/runs/searle_xl_dev_oti.json \
       --backbone ViT-L/14 \
       --oti-tokens data/checkpoints/oti_cameo_GPTNeo27B/

   PYTHONPATH=. python scripts/run_searle.py \
       --queries data/queries_official_cir.json \
       --out data/runs/searle_xl_official_oti.json \
       --backbone ViT-L/14 \
       --oti-tokens data/checkpoints/oti_cameo_GPTNeo27B/

   PYTHONPATH=. python scripts/eval_by_rule.py --run data/runs/searle_xl_dev_oti.json
   PYTHONPATH=. python scripts/evaluate.py --run data/runs/searle_xl_official_oti.json \
       --queries data/queries_official_cir.json \
       --use-judge --judge-model haiku --judge-k 20 --k 5 10 20
   ```

## Knobs in `run_remote.sh`

| Env var       | Default        | Notes                                                                 |
|---------------|----------------|-----------------------------------------------------------------------|
| `BACKBONE`    | `ViT-L/14`     | Must match the backbone used at inference (`run_searle.py --backbone`). |
| `OTI_STEPS`   | `350`          | SEARLE paper default. 100 is faster but worse; 350 is safe.            |
| `BATCH_SIZE`  | `32`           | Bump on A100; drop on smaller GPU.                                     |
| `EXP_NAME`    | `cameo_GPTNeo27B` | Output directory name. Change for ablations.                        |

## What's adapted upstream

- `SEARLE/src/cameo_dataset.py` — `CAMEODataset` class (mirrors
  `ImageNetDataset`'s `{image, image_name}` interface). Handles `.mp4`
  first-frame + corrupt-jpg → gray placeholder.
- `SEARLE/src/image_concepts_association.py` + `SEARLE/src/oti_inversion.py`
  — additive `cameo` branch in the dataset switch; everything else unchanged.

Phi training (`train_phi.py`) is **skipped** for EXP-004. We use the
SEARLE-OTI inference variant: every query reference image is in the trained
corpus, so we look up its OTI pseudo-token directly instead of predicting via
Phi. The paper reports SEARLE-OTI ≥ SEARLE-Phi on CIRCO.
