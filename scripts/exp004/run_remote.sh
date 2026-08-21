#!/usr/bin/env bash
#
# EXP-004 — Vast.ai run pipeline.
#
#   1. image-concept association  → SEARLE/data/similar_concepts/cameo/test/*.csv
#   2. OTI inversion             → SEARLE/data/oti_pseudo_tokens/cameo/test/cameo_GPTNeo27B/
#         * image_names.pkl
#         * ema_oti_pseudo_tokens.pt
#         * oti_pseudo_tokens.pt
#         * hyperparameters.json
#
# The OTI outputs are what we pull back to local; inference at home uses them
# via run_searle.py --oti-tokens <dir>.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/workspace/cameo-retrank}
SEARLE_ROOT=${REPO_ROOT}/SEARLE
EXP_NAME=${EXP_NAME:-cameo_GPTNeo27B}
BACKBONE=${BACKBONE:-ViT-L/14}
OTI_STEPS=${OTI_STEPS:-350}
BATCH_SIZE=${BATCH_SIZE:-32}

cd "${SEARLE_ROOT}"

echo "=== 1. image-concept association ==="
echo "    backbone=${BACKBONE}"
python src/image_concepts_association.py \
    --clip-model-name "${BACKBONE}" \
    --dataset cameo \
    --dataset-path "${REPO_ROOT}" \
    --split test \
    --dataset-mode classic \
    --batch-size "${BATCH_SIZE}" \
    --num-workers 4 \
    --preprocess-type targetpad

echo "=== 2. OTI inversion ==="
echo "    exp_name=${EXP_NAME}  steps=${OTI_STEPS}  backbone=${BACKBONE}"
python src/oti_inversion.py \
    --exp-name "${EXP_NAME}" \
    --clip-model-name "${BACKBONE}" \
    --dataset cameo \
    --dataset-path "${REPO_ROOT}" \
    --split test \
    --gpt-exp-name GPTNeo27B \
    --batch-size "${BATCH_SIZE}" \
    --preprocess-type targetpad \
    --oti-steps "${OTI_STEPS}" \
    --save-frequency 50

echo "=== done ==="
echo "Outputs to pull back to local:"
echo "  ${SEARLE_ROOT}/data/oti_pseudo_tokens/cameo/test/${EXP_NAME}/"
du -sh "${SEARLE_ROOT}/data/oti_pseudo_tokens/cameo/test/${EXP_NAME}/" || true
