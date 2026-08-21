#!/usr/bin/env bash
#
# EXP-004 — Vast.ai (or any CUDA Linux box) setup.
#
# Assumes:
#   * You have ssh'd into a fresh GPU instance with CUDA 11/12 already installed
#     (Vast.ai PyTorch images include CUDA + a working torch).
#   * The CAMEO repo has been git-pushed to the instance at /workspace/cameo-retrank
#     (rsync command in scripts/exp004/README.md).
#
# After this script runs, the instance is ready to run scripts/exp004/run_remote.sh.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/workspace/cameo-retrank}
SEARLE_ROOT=${REPO_ROOT}/SEARLE

echo "=== EXP-004 setup ==="
echo "  REPO_ROOT=${REPO_ROOT}"
echo "  SEARLE_ROOT=${SEARLE_ROOT}"

cd "${REPO_ROOT}"

echo "--- python + cuda ---"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

echo "--- pip install ---"
pip install --quiet \
    "git+https://github.com/openai/CLIP.git" \
    ftfy regex tqdm pandas pyarrow opencv-python-headless pillow \
    "numpy<2"

echo "--- download GPT phrases (SEARLE release, ~few MB) ---"
mkdir -p "${SEARLE_ROOT}/data/GPT_phrases"
cd "${SEARLE_ROOT}/data/GPT_phrases"
if [ ! -f GPTNeo27B/concept_to_phrases.pkl ]; then
    curl -L -o GPTNeo27B.zip \
        https://github.com/miccunifi/SEARLE/releases/download/weights/GPTNeo27B.zip
    unzip -q -o GPTNeo27B.zip
    rm GPTNeo27B.zip
fi
ls GPTNeo27B/ | head

echo "--- download Open Images V7 class descriptions ---"
cd "${SEARLE_ROOT}/data"
if [ ! -f oidv7-class-descriptions.csv ]; then
    curl -L -o oidv7-class-descriptions.csv \
        https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv
fi
wc -l oidv7-class-descriptions.csv

echo "--- verify CAMEO data layout ---"
test -f "${REPO_ROOT}/data/raw/products__test.parquet" \
    || { echo "FATAL: missing ${REPO_ROOT}/data/raw/products__test.parquet"; exit 1; }
python -c "
import pandas as pd
df = pd.read_parquet('${REPO_ROOT}/data/raw/products__test.parquet').drop_duplicates('product_id')
print('products:', len(df))
"

echo "=== setup OK — now run scripts/exp004/run_remote.sh ==="
