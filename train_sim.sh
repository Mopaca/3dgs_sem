#!/bin/bash

set -e

# ============================================================
# 3DGS experiment pipeline
#
# Usage:
# ./train.sh <dataset_path> <image_folder> <output_path>
#
# Example:
# ./train.sh \
#   datasets/NeRF360/flowers \
#   images_8 \
#   output/flowers8_scale20_sampling03
# ============================================================


PROJECT_DIR="/workspace/3dgs_sem"

cd "$PROJECT_DIR"

# ------------------------------------------------------------
# Arguments
# ------------------------------------------------------------

DATASET_PATH="$1"
IMAGE_FOLDER="$2"
OUTPUT_PATH="$3"


# ------------------------------------------------------------
# Check arguments
# ------------------------------------------------------------

if [ -z "$DATASET_PATH" ] || [ -z "$IMAGE_FOLDER" ] || [ -z "$OUTPUT_PATH" ]; then

    echo ""
    echo "Usage:"
    echo "./train.sh <dataset_path> <image_folder> <output_path>"
    echo ""
    echo "Example:"
    echo "./train.sh datasets/NeRF360/flowers images_8 output/flowers8_scale20_sampling03"
    echo ""

    exit 1
fi


# ------------------------------------------------------------
# Persistent Torch / DINO cache
# ------------------------------------------------------------

export TORCH_HOME=/workspace/.cache/torch


echo ""
echo "============================================================"
echo " 3DGS Experiment"
echo "============================================================"
echo ""
echo "Dataset      : $DATASET_PATH"
echo "Images       : $IMAGE_FOLDER"
echo "Output       : $OUTPUT_PATH"
echo "Torch cache  : $TORCH_HOME"
echo ""


# ------------------------------------------------------------
# GPU check
# ------------------------------------------------------------

echo "============================================================"
echo "[1/4] GPU Check"
echo "============================================================"

python - <<'PY'
import torch

print("Torch :", torch.__version__)
print("CUDA  :", torch.version.cuda)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

print("GPU   :", torch.cuda.get_device_name(0))
PY


# ------------------------------------------------------------
# Train
# ------------------------------------------------------------

echo ""
echo "============================================================"
echo "[2/4] TRAIN"
echo "============================================================"
echo ""

python train_sim.py \
    -s "$DATASET_PATH" \
    -i "$IMAGE_FOLDER" \
    --eval \
    -m "$OUTPUT_PATH"


echo ""
echo "============================================================"
echo "Training completed."
echo "============================================================"


# ------------------------------------------------------------
# Render
# ------------------------------------------------------------

echo ""
echo "============================================================"
echo "[3/4] RENDER"
echo "============================================================"
echo ""

python render.py \
    -m "$OUTPUT_PATH"


echo ""
echo "============================================================"
echo "Rendering completed."
echo "============================================================"


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

echo ""
echo "============================================================"
echo "[4/4] METRICS"
echo "============================================================"
echo ""

python metrics.py \
    -m "$OUTPUT_PATH"


echo ""
echo "============================================================"
echo " ALL DONE"
echo "============================================================"
echo ""
echo "Dataset : $DATASET_PATH"
echo "Images  : $IMAGE_FOLDER"
echo "Output  : $OUTPUT_PATH"
echo ""