#!/bin/bash
# ============================================================
# Decisive experiment runner (review item 1, plus controls for items 3 and 4)
#
#   scenes x arms x seeds  ->  train_sim.py -> render.py -> metrics.py -> eval_maps.py
#
# Output: $OUT_ROOT/<scene>/<arm>/seed<k>/
#   cfg_args, sem_run_config.json, train_stats.json, sem_log.jsonl, sem_snapshots/,
#   test/ours_30000/{renders,gt}, results.json, per_view.json, eval_maps.json, *.log, command.txt
# Each stage is skipped when its output already exists, so the script can simply be re-run
# after an interruption (e.g. a RunPod restart).
#
# Usage (from anywhere):
#   bash run_decisive.sh
#   SCENES="flowers" SEEDS="0" ARMS="dino_l03 ssim_l03" bash run_decisive.sh
#   M360_DIR=/workspace/datasets/NeRF360 TANDT_DIR=/workspace/datasets/tandt bash run_decisive.sh
#
# Afterwards:  python collect_results.py --root "$OUT_ROOT"
# ============================================================

set -u

cd "$(dirname "$0")"

# ---------------- paths ----------------
M360_DIR=${M360_DIR:-datasets/NeRF360}
TANDT_DIR=${TANDT_DIR:-datasets/tandt}
DB_DIR=${DB_DIR:-datasets/db}
OUT_ROOT=${OUT_ROOT:-output/decisive}
export TORCH_HOME=${TORCH_HOME:-/root/.cache/torch}

# ---------------- fixed hyperparameters ----------------
# Passed explicitly so that the defaults in arguments/__init__.py do not matter.
TAU_G=${TAU_G:-0.0002}   # positional-gradient threshold (3DGS default)
RHO=${RHO:-0.3}          # stratified sampling ratio: SET TO THE VALUE USED FOR THE REPORTED "Ours"
SPLIT=${SPLIT:-adc}      # densify_and_split_by_selected_mask (used for the reported "Ours")

# ---------------- what to run ----------------
SCENES=${SCENES:-"flowers treehill garden truck"}
SEEDS=${SEEDS:-"0"}
ARMS=${ARMS:-"dino_l03 ssim_l03 l1_l03 dino_l0 ssim_l0 l1_l0 rand_matched radius_matched"}

ITER=30000
COMMON="--eval --disable_viewer --test_iterations -1 --save_iterations $ITER \
--densify_grad_threshold $TAU_G --sem_sample_ratio $RHO --sem_split $SPLIT --sem_log"

M360_OUTDOOR="bicycle flowers garden stump treehill"
M360_INDOOR="room counter kitchen bonsai"
TANDT_SCENES="truck train"
DB_SCENES="drjohnson playroom"

contains() { case " $1 " in *" $2 "*) return 0 ;; *) return 1 ;; esac; }

scene_source() {   # prints "<source_path> <image_folder>"
    local s=$1
    if contains "$M360_OUTDOOR" "$s"; then echo "$M360_DIR/$s images_8"
    elif contains "$M360_INDOOR" "$s"; then echo "$M360_DIR/$s images_2"
    elif contains "$TANDT_SCENES" "$s"; then echo "$TANDT_DIR/$s images"
    elif contains "$DB_SCENES" "$s"; then echo "$DB_DIR/$s images"
    else return 1; fi
}

arm_flags() {
    case "$1" in
        # ---- 1st round: signal x scale weight, and selection controls ----
        dino_l03)       echo "--sem_signal dino --sem_shadow ssim" ;;   # proposed method (+ SSIM shadow for logging)
        ssim_l03)       echo "--sem_signal ssim" ;;
        l1_l03)         echo "--sem_signal l1" ;;
        dino_l0)        echo "--sem_signal dino --sem_lambda_scale 0" ;;
        ssim_l0)        echo "--sem_signal ssim --sem_lambda_scale 0" ;;
        l1_l0)          echo "--sem_signal l1 --sem_lambda_scale 0" ;;
        rand_matched)   echo "--sem_signal dino --sem_select random_matched" ;;   # same count, random
        radius_matched) echo "--sem_signal dino --sem_select radius_matched" ;;   # same count, largest radius
        dino_pow1)      echo "--sem_signal dino --sem_scale_mode power --sem_lambda_scale 1" ;;   # S^(1+R)
        dino_pow2)      echo "--sem_signal dino --sem_scale_mode power --sem_lambda_scale 2" ;;   # S^(1+2R)
        # ---- 2nd round (optional) ----
        l1blur_l03)     echo "--sem_signal l1_blur" ;;
        dino_viewmin)   echo "--sem_signal dino --sem_view_agg min" ;;              # view mean vs worst view
        r9_ssim)        echo "--sem_mode r9 --sem_signal ssim --sem_lambda_scale 0" ;;  # [9]-style rule, 1-SSIM
        r9_dino)        echo "--sem_mode r9 --sem_signal dino --sem_lambda_scale 0" ;;  # [9]-style rule, 1-cos
        r9_dino_mean)   echo "--sem_mode r9 --sem_signal dino --sem_lambda_scale 0 --sem_view_norm mean" ;;  # weighted mean vs sum
        *) return 1 ;;
    esac
}

for scene in $SCENES; do
    src_img=$(scene_source "$scene") || { echo "[decisive] unknown scene: $scene"; exit 1; }
    read -r SRC IMG <<< "$src_img"
    if [ ! -d "$SRC" ]; then echo "[decisive] dataset not found: $SRC"; exit 1; fi

    for arm in $ARMS; do
        flags=$(arm_flags "$arm") || { echo "[decisive] unknown arm: $arm"; exit 1; }

        for seed in $SEEDS; do
            out="$OUT_ROOT/$scene/$arm/seed$seed"
            mkdir -p "$out"
            cmd="python train_sim.py -s $SRC -i images_8 -m $out --seed $seed $COMMON $flags"
            echo "$cmd" > "$out/command.txt"
            echo "============================================================"
            echo "[decisive] $scene / $arm / seed$seed"

            if [ ! -f "$out/train_stats.json" ]; then
                $cmd > "$out/train.log" 2>&1 || { echo "[decisive] TRAIN FAILED: $out (see train.log)"; continue; }
            else
                echo "  train: done, skipped"
            fi

            if [ ! -d "$out/test/ours_$ITER/renders" ]; then
                python render.py -m "$out" --iteration $ITER --skip_train > "$out/render.log" 2>&1 \
                    || { echo "[decisive] RENDER FAILED: $out"; continue; }
            fi

            if [ ! -f "$out/results.json" ]; then
                python metrics.py -m "$out" > "$out/metrics.log" 2>&1 \
                    || { echo "[decisive] METRICS FAILED: $out"; continue; }
            fi

            if [ ! -f "$out/eval_maps.json" ]; then
                python eval_maps.py -m "$out" > "$out/eval_maps.log" 2>&1 \
                    || echo "[decisive] EVAL_MAPS FAILED: $out (metrics are still valid)"
            fi
        done
    done
done

python collect_results.py --root "$OUT_ROOT"
