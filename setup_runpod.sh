#!/bin/bash

set -e

echo "=================================================="
echo " 3DGS + DINOv2 RunPod Environment Setup"
echo " Target: NVIDIA A40 / CUDA 12.8 / Torch 2.8.0"
echo "=================================================="

PROJECT_DIR="/workspace/3dgs_sem"
TORCH_CACHE="/workspace/.cache/torch"

cd "$PROJECT_DIR"

# --------------------------------------------------
# 0. Persistent cache
# --------------------------------------------------

echo ""
echo "[0/8] Setting persistent Torch cache..."

mkdir -p "$TORCH_CACHE"
export TORCH_HOME="$TORCH_CACHE"

echo "TORCH_HOME=$TORCH_HOME"


# --------------------------------------------------
# 1. System check
# --------------------------------------------------

echo ""
echo "[1/8] Checking GPU / CUDA / Python..."

nvidia-smi || true

echo ""
echo "Python:"
python --version

echo ""
echo "NVCC:"
nvcc --version || true


# --------------------------------------------------
# 2. Basic Python build tools
# --------------------------------------------------

echo ""
echo "[2/8] Installing build tools..."

python -m pip install --upgrade pip setuptools wheel

python -m pip install \
    ninja \
    packaging


# --------------------------------------------------
# 3. Install PyTorch 2.8.0 + CUDA 12.8
# --------------------------------------------------

echo ""
echo "[3/8] Checking PyTorch..."

NEED_TORCH_INSTALL=0

python - <<'PY' || NEED_TORCH_INSTALL=1
import torch

print("Current Torch :", torch.__version__)
print("Torch CUDA    :", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

assert torch.__version__.startswith("2.8.0"), \
    f"Expected Torch 2.8.0, got {torch.__version__}"

assert torch.version.cuda == "12.8", \
    f"Expected CUDA 12.8 build, got {torch.version.cuda}"

assert torch.cuda.is_available(), \
    "CUDA is not available"
PY

if [ "$NEED_TORCH_INSTALL" -eq 1 ]; then

    echo "Installing Torch 2.8.0 + cu128..."

    python -m pip uninstall -y \
        torch torchvision torchaudio xformers 2>/dev/null || true

    python -m pip install \
        torch==2.8.0 \
        torchvision==0.23.0 \
        torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu128

else
    echo "Correct PyTorch already installed."
fi


# --------------------------------------------------
# 4. Common 3DGS dependencies
# --------------------------------------------------

echo ""
echo "[4/8] Installing 3DGS dependencies..."

python -m pip install \
    numpy \
    scipy \
    pillow \
    tqdm \
    plyfile \
    opencv-python \
    tensorboard \
    matplotlib==3.8.4


# --------------------------------------------------
# 5. DINOv2 dependencies
# --------------------------------------------------

echo ""
echo "[5/8] Installing DINOv2 dependencies..."

# IMPORTANT:
# Do NOT install dinov2/requirements.txt
# It pins an old Torch/CUDA environment.

python -m pip install \
    timm \
    einops \
    omegaconf \
    fvcore \
    iopath

# # --------------------------------------------------
# # Dataset Setup - Mip-NeRF 360
# # --------------------------------------------------

# echo ""
# echo "[Dataset] Setting up Mip-NeRF 360 dataset..."

# DATASET_ROOT="$PROJECT_DIR/datasets"
# NERF360_DIR="$DATASET_ROOT/NeRF360"

# mkdir -p "$DATASET_ROOT"
# mkdir -p "$NERF360_DIR"

# cd "$NERF360_DIR"


# # --------------------------------------------------
# # Download helper
# # --------------------------------------------------

# download_file () {

#     URL="$1"
#     OUTPUT="$2"

#     if [ -f "$OUTPUT" ]; then
#         echo "$OUTPUT already exists - skipping download."
#         return
#     fi

#     echo ""
#     echo "Downloading $OUTPUT..."

#     wget \
#         --show-progress \
#         --continue \
#         "$URL" \
#         -O "$OUTPUT"
# }


# # --------------------------------------------------
# # 1. Standard Mip-NeRF 360 scenes
# #
# # bicycle
# # bonsai
# # counter
# # garden
# # kitchen
# # room
# # stump
# # --------------------------------------------------

# BASE_ZIP="360_v2.zip"

# download_file \
#     "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip" \
#     "$BASE_ZIP"


# echo ""
# echo "Extracting standard Mip-NeRF 360 scenes..."

# unzip -q -o "$BASE_ZIP"

# echo "Standard scenes extracted."


# # --------------------------------------------------
# # 2. Extra scenes
# #
# # flowers
# # treehill
# # --------------------------------------------------

# EXTRA_ZIP="360_extra_scenes.zip"

# download_file \
#     "https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip" \
#     "$EXTRA_ZIP"


# echo ""
# echo "Extracting extra scenes..."

# unzip -q -o "$EXTRA_ZIP"

# echo "Extra scenes extracted."


# # --------------------------------------------------
# # Remove zip files after successful extraction
# # --------------------------------------------------

# echo ""
# echo "Removing downloaded ZIP files..."

# rm -f "$BASE_ZIP"
# rm -f "$EXTRA_ZIP"


# # --------------------------------------------------
# # Dataset verification
# # --------------------------------------------------

# echo ""
# echo "======================================"
# echo "Mip-NeRF 360 Dataset"
# echo "======================================"

# EXPECTED_SCENES=(
#     bicycle
#     bonsai
#     counter
#     flowers
#     garden
#     kitchen
#     room
#     stump
#     treehill
# )

# for SCENE in "${EXPECTED_SCENES[@]}"; do

#     if [ -d "$NERF360_DIR/$SCENE" ]; then

#         IMAGE_COUNT=0

#         if [ -d "$NERF360_DIR/$SCENE/images" ]; then
#             IMAGE_COUNT=$(find "$NERF360_DIR/$SCENE/images" \
#                 -type f | wc -l)
#         fi

#         echo "[OK] $SCENE - images: $IMAGE_COUNT"

#     else

#         echo "[MISSING] $SCENE"

#     fi

# done


# echo ""
# echo "Dataset directory:"
# echo "$NERF360_DIR"

# cd "$PROJECT_DIR"

# --------------------------------------------------
# 6. Check / patch custom rasterizer config
# --------------------------------------------------

echo ""
echo "[6/8] Checking custom rasterizer..."

RASTER_DIR="$PROJECT_DIR/submodules/diff-gaussian-rasterization-feature"
CONFIG_FILE="$RASTER_DIR/cuda_rasterizer/config.h"
IMPL_HEADER="$RASTER_DIR/cuda_rasterizer/rasterizer_impl.h"

if [ ! -d "$RASTER_DIR" ]; then
    echo "ERROR: Rasterizer directory does not exist:"
    echo "$RASTER_DIR"
    exit 1
fi


# ---------- Ensure cstdint exists ----------

if ! grep -q "#include <cstdint>" "$IMPL_HEADER"; then

    echo "Adding #include <cstdint> to rasterizer_impl.h..."

    sed -i '1i #include <cstdint>' "$IMPL_HEADER"

else
    echo "<cstdint> already present."
fi


# ---------- Set RGB channels = 3 ----------

echo "Setting NUM_CHANNELS = 3..."

sed -i \
    's/#define NUM_CHANNELS .*/#define NUM_CHANNELS 3 \/\/ RGB/' \
    "$CONFIG_FILE"


# ---------- Set semantic channels = 3 ----------

echo "Setting NUM_SEMANTIC_CHANNELS = 3..."

sed -i \
    's/#define NUM_SEMANTIC_CHANNELS .*/#define NUM_SEMANTIC_CHANNELS 3 \/\/ similarity_aux/' \
    "$CONFIG_FILE"


echo ""
echo "Current config:"
grep -E "NUM_CHANNELS|NUM_SEMANTIC_CHANNELS" "$CONFIG_FILE"


# --------------------------------------------------
# 7. Build CUDA extensions
# --------------------------------------------------

echo ""
echo "[7/8] Building CUDA rasterizer..."

cd "$RASTER_DIR"

python -m pip uninstall -y diff-gaussian-rasterization 2>/dev/null || true

rm -rf build
rm -rf dist
rm -rf *.egg-info

python -m pip install --no-build-isolation .


# ---------- simple-knn ----------

SIMPLE_KNN="$PROJECT_DIR/submodules/simple-knn"

if [ -d "$SIMPLE_KNN" ]; then

    echo ""
    echo "Building simple-knn..."

    cd "$SIMPLE_KNN"

    python -m pip uninstall -y simple-knn 2>/dev/null || true

    rm -rf build
    rm -rf dist
    rm -rf *.egg-info

    python -m pip install --no-build-isolation .

else

    echo "simple-knn directory not found - skipping."

fi


# --------------------------------------------------
# 8. Final verification
# --------------------------------------------------

echo ""
echo "[8/8] Final environment check..."

cd "$PROJECT_DIR"

python - <<'PY'

import torch

print()
print("======================================")
print("PyTorch")
print("======================================")

print("Torch version :", torch.__version__)
print("CUDA version  :", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU            :", torch.cuda.get_device_name(0))


print()
print("======================================")
print("Rasterizer")
print("======================================")

import diff_gaussian_rasterization

print("diff_gaussian_rasterization: OK")


try:
    import simple_knn
    print("simple_knn: OK")
except Exception as e:
    print("simple_knn check:", e)


print()
print("======================================")
print("DINO dependencies")
print("======================================")

import timm
import einops
import omegaconf
import fvcore
import iopath

print("DINO dependencies: OK")

print()
print("Environment setup complete.")

PY


echo ""
echo "=================================================="
echo " SETUP COMPLETE"
echo "=================================================="
echo ""
echo "Project : $PROJECT_DIR"
echo "Torch cache : $TORCH_CACHE"
echo ""
echo "To train:"
echo ""
echo "cd /workspace/3dgs_sem"
echo "export TORCH_HOME=/workspace/.cache/torch"
echo ""
echo "python train_sim.py \\"
echo "  -s datasets/NeRF360/flowers \\"
echo "  -i images_8 \\"
echo "  --eval \\"
echo "  -m output/flowers8_scale20_sampling03"
echo ""