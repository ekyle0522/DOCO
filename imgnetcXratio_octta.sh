#!/bin/bash

set -e
trap 'echo "Error at line $LINENO: $BASH_COMMAND"; exit 1' ERR

export HF_HUB_OFFLINE=1

CONDA_ENV_NAME="doco"
CUDA_DEVICE_ID="2"
DATA_DIR="${DATA_ROOT:-/mnt/d/stamp_lib/datasets}"

BATCH_SIZE=64

METHODS=(
    "source"
    "eatacome"
    "eata"
    "dpcore"
    "doco"
    "tent"
    "eataunient"
    "sar"
    "sarcome"
    "cotta"
    "ostta"
    "vida"
    "stamp"
)

OOD_DATASETS=(
    "places365"
    "textures"
    "inaturalist"
    "sun"
    "ssb-hard"
    "ninco_ood_classes"
)

# Outer loop: ratio
OOD_PROPORTIONS=(
    50
    # 10
    # 20
    # 30
    # 40
)

NUM_ID_SAMPLES=5000
SAVE_DIR_BASE="../output_imgnetc"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Error: Conda initialization script not found."
    exit 1
fi

conda activate "$CONDA_ENV_NAME"
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_ID

echo "========================================================"
echo "Starting experiment suite at $(date)"
echo "========================================================"

cd imagenet || { echo "Failed to enter the imagenet directory."; exit 1; }
mkdir -p "$SAVE_DIR_BASE"

for ood_ratio_percent in "${OOD_PROPORTIONS[@]}"
do
    if [ "$ood_ratio_percent" -eq 0 ]; then
        num_ood_samples=0
    else
        num_ood_samples=$(awk -v p="$ood_ratio_percent" -v n="$NUM_ID_SAMPLES" 'BEGIN {
        r = p / 100.0
        printf "%.0f", (r * n) / (1 - r)
    }')
    fi

    echo ""
    echo "########################################################"
    echo "Current OOD ratio: ${ood_ratio_percent}%"
    echo "Computed OOD samples: $num_ood_samples"
    echo "########################################################"

    for method in "${METHODS[@]}"
    do
        CONFIG_FILE="./cfgs/ood_${method}.yaml"
        if [ ! -f "$CONFIG_FILE" ]; then
            echo "Warning: Config file $CONFIG_FILE not found. Skipping method $method."
            continue
        fi

        for dataset in "${OOD_DATASETS[@]}"
        do
            RUN_SAVE_DIR="${SAVE_DIR_BASE}/ratio_${ood_ratio_percent}/${method}/${dataset}"
            mkdir -p "$RUN_SAVE_DIR"

            echo ""
            echo "--------------------------------------------------------"
            echo "Running: ratio=${ood_ratio_percent}% / method=${method} / dataset=${dataset}"
            echo "--------------------------------------------------------"

            python octta_main.py \
                --cfg "$CONFIG_FILE" \
                --data_dir "$DATA_DIR" \
                RNG_SEED 1 \
                CORRUPTION.ID_BENCHMARK "imagenet_c" \
                CORRUPTION.OOD_DATASET "$dataset" \
                CORRUPTION.NUM_OOD_SAMPLES "$num_ood_samples" \
                CORRUPTION.NUM_EX "$NUM_ID_SAMPLES" \
                TEST.BATCH_SIZE "$BATCH_SIZE" \
                OPTIM.DOCO_BETA 0.5 \
                OPTIM.DOCO_PROMPT_NUM 8 \
                SAVE_DIR "$RUN_SAVE_DIR"

            echo "Done: ratio=${ood_ratio_percent}% / method=${method} / dataset=${dataset}"
        done
    done
done

echo ""
echo "========================================================"
echo "All experiments completed at $(date)"
echo "========================================================"
